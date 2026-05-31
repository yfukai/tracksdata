"""
Module to compose attribute expressions for attribute filtering or value evaluation.

Attributes are used to query content of nodes and edges through their names as columns in a data frame.

Users will mostly interact with [NodeAttr][tracksdata.attrs.NodeAttr] and [EdgeAttr][tracksdata.attrs.EdgeAttr]
which are thin wrappers around [Attr][tracksdata.attrs.Attr] to distinguish between node and edge attributes
in ambiguous cases.

They can be used to filter elements in the graph as:
```python
graph.filter(NodeAttr("t") == 1).subgraph()
```

Boolean combinations of comparisons can be expressed with `|` (or), `^` (xor),
`&` (and) and `~` (not). Comparisons passed as multiple positional arguments to
`filter()` are still implicitly AND-ed together.
```python
graph.filter((NodeAttr("t") == 1) | (NodeAttr("t") == 2)).subgraph()
graph.filter(~(NodeAttr("t") == 0)).subgraph()
```

Or to create complex expression when solving the tracking problem:
```python
NearestNeighborsSolver(-Attr("iou") * (-Attr("distance") / 30.0).exp())
```
"""

import functools
import math
import operator
from collections.abc import Callable, Sequence
from typing import Any, TypeGuard, Union, overload

import numpy as np
import polars as pl
from polars import DataFrame, Expr, Series

from tracksdata._filter import _FilterCompound, _FilterLeaf, _FilterNode, walk_leaves

Scalar = int | float | str | bool | complex | np.number
ExprInput = Union[str, Scalar, "Attr", Expr]
MembershipExprInput = Sequence[Scalar] | np.ndarray

# A filter-shaped `Attr` (one whose `_filter` is set). Backends introspect
# `Attr._filter` to translate compound boolean filters into SQL / polars /
# Python-dict predicates.
FilterInput = "Attr"


__all__ = [
    "Attr",
    "EdgeAttr",
    "NodeAttr",
    "attr_comps_to_strs",
    "polars_reduce_attr_comps",
    "split_attr_comps",
]


def _is_in_op(lhs: Any, values: MembershipExprInput) -> Any:
    """
    Backend-aware membership operator that works for Polars expressions, SQLAlchemy columns, and Python scalars.
    """
    if isinstance(lhs, pl.Expr):
        return lhs.is_in(values)
    if hasattr(lhs, "in_"):
        return lhs.in_(values)
    return lhs in values


_OPS_MATH_SYMBOLS: dict[Callable, str] = {
    operator.add: "+",
    operator.sub: "-",
    operator.mul: "*",
    operator.truediv: "/",
    operator.floordiv: "//",
    operator.mod: "%",
    operator.pow: "**",
    operator.and_: "&",
    operator.or_: "|",
    operator.xor: "^",
    operator.eq: "==",
    operator.ne: "!=",
    operator.lt: "<",
    operator.le: "<=",
    operator.gt: ">",
    operator.ge: ">=",
    _is_in_op: "in",
}


_FILTER_OP_SYMBOLS = {"and": "&", "or": "|", "xor": "^", "not": "~"}
_BOOLEAN_OP_FUNCS: dict[str, Callable] = {
    "and": operator.and_,
    "or": operator.or_,
    "xor": operator.xor,
}


def _is_membership_expr_input(x: Any) -> TypeGuard[MembershipExprInput]:
    if isinstance(x, Attr | pl.Expr):
        return False
    if isinstance(x, Scalar):
        return False
    if isinstance(x, np.ndarray):
        return getattr(x, "ndim", 1) >= 1
    return isinstance(x, Sequence)


def _cast_membership(values: MembershipExprInput) -> list:
    if isinstance(values, np.ndarray):
        return values.tolist()
    return list(values)


def _cast_scalar(value: Any) -> Any:
    # numpy scalars are problematic for sqlalchemy; unwrap to Python types
    if isinstance(value, np.ndarray):
        return value.item()
    return value


def _filter_repr(node: _FilterNode) -> str:
    if isinstance(node, _FilterLeaf):
        return f"{node.kind.__name__}({node.column}) {_OPS_MATH_SYMBOLS[node.op]} {node.other}"
    if node.op == "not":
        return f"~{_filter_repr(node.operands[0])}"
    sep = f" {_FILTER_OP_SYMBOLS[node.op]} "
    return "(" + sep.join(_filter_repr(o) for o in node.operands) + ")"


class Attr:
    """
    A class to compose an attribute expression for attribute filtering or value evaluation.

    Parameters
    ----------
    value : ExprInput
        The value to compose the attribute expression from.

    Examples
    --------
    ```python
    Attr("t") == 1  # filter for time point 1
    Attr("iou").log()  # log the iou
    Attr(1.0)  # constant value
    Attr((1 - Attr("iou")) * Attr("distance"))  # complex expression
    ```
    """

    expr: Expr
    _filter: _FilterNode | None

    def __init__(self, value: ExprInput) -> None:
        self._inf_exprs: list[Attr] = []  # expressions multiplied by +inf
        self._neg_inf_exprs: list[Attr] = []  # expressions multiplied by -inf
        self._filter = None

        if isinstance(value, str):
            self.expr = pl.col(value)
        elif isinstance(value, Attr):
            self.expr = value.expr
            # Copy infinity tracking; intentionally do NOT copy `_filter` — wrapping
            # an Attr in another Attr is for rebinding/aliasing, not duplicating
            # filter identity (operators set `_filter` on the new instance).
            self._inf_exprs = value.inf_exprs
            self._neg_inf_exprs = value.neg_inf_exprs
        elif isinstance(value, Expr):
            self.expr = value
        else:
            self.expr = pl.lit(value)

    @classmethod
    def _leaf(
        cls,
        column: str,
        op: Callable,
        other: Any,
        kind: type["Attr"] | None = None,
    ) -> "Attr":
        """Construct an `Attr` representing a single leaf filter `column op other`.

        Provided for tests and internal helpers that need to build filter nodes
        without going through Python's operator dispatch.
        """
        is_membership = _is_membership_expr_input(other)
        if is_membership and op is not _is_in_op:
            raise ValueError(
                f"Membership values can only be used with the 'is_in' method. Found '{_OPS_MATH_SYMBOLS[op]}'."
            )
        if not is_membership and op is _is_in_op:
            raise ValueError(
                f"Cannot use 'is_in' method with non-membership values. Found '{other}' of type {type(other)}."
            )
        leaf_kind = kind if kind is not None else cls
        other_cast = _cast_membership(other) if is_membership else _cast_scalar(other)

        if is_membership:
            expr = pl.col(column).is_in(other_cast)
        else:
            expr = op(pl.col(column), other_cast)

        result = leaf_kind(expr)
        result._filter = _FilterLeaf(column=column, op=op, other=other_cast, kind=leaf_kind)
        return result

    def _wrap(self, expr: ExprInput) -> Union["Attr", Any]:
        if isinstance(expr, Expr):
            result = type(self)(expr)
            # Propagate infinity tracking
            result._inf_exprs = self._inf_exprs.copy()
            result._neg_inf_exprs = self._neg_inf_exprs.copy()
            return result
        return expr

    def _result_kind(self, other: "ExprInput") -> type["Attr"]:
        """Pick the result class for a binary op so NodeAttr/EdgeAttr is preserved.

        The base `Attr` defers to a more specific operand; two specific kinds
        must match — mixing `NodeAttr` and `EdgeAttr` in one expression raises
        because they target different graph tables.
        """
        self_kind = type(self)
        if not isinstance(other, Attr):
            return self_kind
        other_kind = type(other)
        if self_kind is Attr:
            return other_kind
        if other_kind is Attr or other_kind is self_kind:
            return self_kind
        raise ValueError(
            f"Cannot combine {self_kind.__name__} and {other_kind.__name__} "
            "in a single expression — they target different graph tables."
        )

    def _delegate_operator(self, other: ExprInput, op: Callable[[Expr, Expr], Expr], reverse: bool = False) -> "Attr":
        """
        Delegate a binary numeric/bitwise operator to the polars expression.

        Arithmetic and pure-bitwise operations always clear `_filter` (the result
        is no longer a filter-shaped Attr), so callers that need to combine
        filter compounds must go through `_delegate_boolean_operator` instead.
        """
        cls = self._result_kind(other)

        # Special handling for multiplication with infinity
        if op == operator.mul:
            # Check if we're multiplying with infinity scalar
            # In both reverse and non-reverse cases, 'other' is the infinity value
            # and 'self' is the AttrExpr we want to track
            if isinstance(other, int | float) and math.isinf(other):
                result = cls(pl.lit(0))  # Clean expression is zero (infinity term removed)

                # Copy existing infinity tracking
                result._inf_exprs = self._inf_exprs.copy()
                result._neg_inf_exprs = self._neg_inf_exprs.copy()

                # Add the expression to appropriate infinity list
                if other > 0:
                    result._inf_exprs.append(self)
                else:
                    result._neg_inf_exprs.append(self)

                return result

        # Regular operation - no infinity involved
        left = Attr(other).expr if reverse else self.expr
        right = self.expr if reverse else Attr(other).expr
        result = cls(op(left, right))

        # Combine infinity tracking from both operands
        if isinstance(other, Attr):
            result._inf_exprs = self._inf_exprs + other._inf_exprs
            result._neg_inf_exprs = self._neg_inf_exprs + other._neg_inf_exprs

            # Special handling for subtraction: flip signs of the second operand's infinity terms
            if op == operator.sub and not reverse:
                # self - other: other's positive infinity becomes negative, negative becomes positive
                result._inf_exprs = self._inf_exprs + other._neg_inf_exprs
                result._neg_inf_exprs = self._neg_inf_exprs + other._inf_exprs
            elif op == operator.sub and reverse:
                # other - self: self's positive infinity becomes negative, negative becomes positive
                result._inf_exprs = other._inf_exprs + self._neg_inf_exprs
                result._neg_inf_exprs = other._neg_inf_exprs + self._inf_exprs
        else:
            result._inf_exprs = self._inf_exprs.copy()
            result._neg_inf_exprs = self._neg_inf_exprs.copy()

        return result

    def _delegate_comparison_operator(self, other: ExprInput, op: Callable) -> "Attr":
        """
        Build a leaf-filter `Attr` for `self <op> other` when possible.

        If `other` is itself an `Attr`, the result is a non-filter Attr that
        evaluates as a polars boolean expression. If `self` has infinity
        tracking, comparison is rejected as semantically meaningless.
        Multi-column / literal LHS also falls back to a non-filter result —
        such filters can't be pushed down to SQL and must be evaluated by
        polars only.
        """
        if self.has_inf():
            raise ValueError("Comparison operators are not supported for expressions with infinity.")

        if isinstance(other, Attr):
            return self._delegate_operator(other, op)

        columns = self.expr_columns
        if len(columns) != 1:
            # Can't form a leaf — fall back to a non-filter Attr.
            return self._delegate_operator(other, op)

        other_cast = _cast_scalar(other)
        expr = op(self.expr, other_cast)
        result = type(self)(expr)
        result._filter = _FilterLeaf(column=columns[0], op=op, other=other_cast, kind=type(self))
        return result

    def _delegate_boolean_operator(self, other: "ExprInput", op_name: str, reverse: bool = False) -> "Attr":
        """
        Combine two `Attr`s with a boolean op (`& | ^`).

        If both have `_filter` set, build a compound filter (auto-flattening
        nested same-op compounds). If neither has `_filter`, fall through to
        plain bitwise polars evaluation. Mixing a filter-shaped Attr with a
        non-filter operand raises: implicit pushdown loss is too easy to miss.
        """
        op_func = _BOOLEAN_OP_FUNCS[op_name]
        self_has = self._filter is not None
        other_has = isinstance(other, Attr) and other._filter is not None

        if self_has != other_has:
            symbol = _FILTER_OP_SYMBOLS[op_name]
            raise TypeError(
                f"Cannot apply '{symbol}' between a filter-shaped Attr and a non-filter operand. "
                "Both operands must be filter-shaped (built from comparisons) or both non-filter."
            )

        if not self_has:
            # Neither has filter structure — pure bitwise op, no `_filter` on result.
            return self._delegate_operator(other, op_func, reverse=reverse)

        # Both have filter — combine into a compound, auto-flattening associative ops.
        cls = self._result_kind(other)
        first, second = (other, self) if reverse else (self, other)
        operands: list[_FilterNode] = []
        for op_attr in (first, second):
            f = op_attr._filter
            if isinstance(f, _FilterCompound) and f.op == op_name:
                operands.extend(f.operands)
            else:
                operands.append(f)

        if reverse:
            expr = op_func(other.expr, self.expr)
        else:
            expr = op_func(self.expr, other.expr)

        result = cls(expr)
        result._filter = _FilterCompound(op_name, tuple(operands))
        return result

    def alias(self, name: str) -> "Attr":
        result = type(self)(self.expr.alias(name))
        result._inf_exprs = self._inf_exprs.copy()
        result._neg_inf_exprs = self._neg_inf_exprs.copy()
        return result

    def evaluate(self, df: DataFrame) -> Series:
        """
        Evaluate the expression on a DataFrame returning a numeric result.

        Parameters
        ----------
        df : DataFrame
            The DataFrame to evaluate the expression on.

        Returns
        -------
        Series
            The evaluated expression.
        """
        return df.select(self.expr).to_series()

    @property
    def columns(self) -> list[str]:
        return list(dict.fromkeys(self.expr_columns + self.inf_columns + self.neg_inf_columns))

    @property
    def inf_exprs(self) -> list["Attr"]:
        """Get the expressions multiplied by positive infinity."""
        return self._inf_exprs.copy()

    @property
    def neg_inf_exprs(self) -> list["Attr"]:
        """Get the expressions multiplied by negative infinity."""
        return self._neg_inf_exprs.copy()

    @property
    def expr_columns(self) -> list[str]:
        """Get the names of columns in the expression."""
        return list(dict.fromkeys(self.expr.meta.root_names()))

    @property
    def inf_columns(self) -> list[str]:
        """Get the names of columns multiplied by positive infinity."""
        columns: list[str] = []
        for attr_expr in self._inf_exprs:
            columns.extend(attr_expr.columns)
        return list(dict.fromkeys(columns))

    @property
    def neg_inf_columns(self) -> list[str]:
        """Get the names of columns multiplied by negative infinity."""
        columns: list[str] = []
        for attr_expr in self._neg_inf_exprs:
            columns.extend(attr_expr.columns)
        return list(dict.fromkeys(columns))

    def has_inf(self) -> bool:
        """
        Check if any column in the expression is multiplied by infinity or negative infinity.
        """
        return self.has_pos_inf() or self.has_neg_inf()

    def has_pos_inf(self) -> bool:
        return len(self._inf_exprs) > 0

    def has_neg_inf(self) -> bool:
        return len(self._neg_inf_exprs) > 0

    def is_in(self, values: MembershipExprInput) -> "Attr":
        """
        Create a membership filter `self in values`.

        Returns a filter-shaped `Attr` suitable for `graph.filter()` and for
        composition with `&`, `|`, `^`, `~`.

        Parameters
        ----------
        values : Iterable[Scalar] | Sequence[Scalar] | np.ndarray
            Values the attribute should belong to.
        """
        if not _is_membership_expr_input(values):
            raise ValueError(
                f"Cannot use 'is_in' method with non-membership values. Found '{values}' of type {type(values)}."
            )
        if self.has_inf():
            raise ValueError("Comparison operators are not supported for expressions with infinity.")
        columns = self.expr_columns
        if len(columns) != 1:
            raise ValueError(f"'is_in' is only supported for single-column expressions. Found columns {columns}.")
        values_cast = _cast_membership(values)
        expr = self.expr.is_in(values_cast)
        result = type(self)(expr)
        result._filter = _FilterLeaf(column=columns[0], op=_is_in_op, other=values_cast, kind=type(self))
        return result

    def __invert__(self) -> "Attr":
        result = type(self)(~self.expr)
        result._inf_exprs = self._inf_exprs.copy()
        result._neg_inf_exprs = self._neg_inf_exprs.copy()
        if self._filter is not None:
            result._filter = _FilterCompound("not", (self._filter,))
        return result

    def __neg__(self) -> "Attr":
        result = type(self)(-self.expr)
        # `-(x * inf)` is `x * -inf`: swap positive and negative trackers.
        result._inf_exprs = self._neg_inf_exprs.copy()
        result._neg_inf_exprs = self._inf_exprs.copy()
        return result

    def __pos__(self) -> "Attr":
        return self

    def __abs__(self) -> "Attr":
        result = type(self)(abs(self.expr))
        result._inf_exprs = self._inf_exprs.copy()
        result._neg_inf_exprs = self._neg_inf_exprs.copy()
        return result

    def __getattr__(self, attr: str) -> Any:
        # Don't delegate our internal attributes to the expr
        if attr.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

        # To auto generate operator methods such as `.log()``
        expr_attr = getattr(self.expr, attr)
        if callable(expr_attr):

            @functools.wraps(expr_attr)
            def _wrapped(*args, **kwargs):
                return self._wrap(expr_attr(*args, **kwargs))

            return _wrapped
        return expr_attr

    def __repr__(self) -> str:
        if self._filter is not None:
            return _filter_repr(self._filter)
        # Non-filter Attrs always render as `Attr(<expr>)` regardless of subclass —
        # the kind is meaningful for filter dispatch, not for arbitrary expressions.
        return f"Attr({self.expr})"

    # Binary arithmetic operators (auto-generated by `_setup_ops`)
    def __add__(self, other: ExprInput) -> "Attr": ...
    def __sub__(self, other: ExprInput) -> "Attr": ...
    def __mul__(self, other: ExprInput) -> "Attr": ...
    def __truediv__(self, other: ExprInput) -> "Attr": ...
    def __floordiv__(self, other: ExprInput) -> "Attr": ...
    def __mod__(self, other: ExprInput) -> "Attr": ...
    def __pow__(self, other: ExprInput) -> "Attr": ...

    # Boolean / bitwise operators (auto-generated by `_setup_ops`)
    def __and__(self, other: ExprInput) -> "Attr": ...
    def __or__(self, other: ExprInput) -> "Attr": ...
    def __xor__(self, other: ExprInput) -> "Attr": ...

    # Reverse arithmetic operators
    def __radd__(self, other: Scalar) -> "Attr": ...
    def __rsub__(self, other: Scalar) -> "Attr": ...
    def __rmul__(self, other: Scalar) -> "Attr": ...
    def __rtruediv__(self, other: Scalar) -> "Attr": ...
    def __rfloordiv__(self, other: Scalar) -> "Attr": ...
    def __rmod__(self, other: Scalar) -> "Attr": ...
    def __rpow__(self, other: Scalar) -> "Attr": ...
    def __rand__(self, other: Scalar) -> "Attr": ...
    def __ror__(self, other: Scalar) -> "Attr": ...
    def __rxor__(self, other: Scalar) -> "Attr": ...

    # Comparison operators with overloads (auto-generated by `_setup_ops`).
    # No reflected `__r{eq,ne,lt,le,gt,ge}__` — Python uses the symmetric / opposite
    # operator on the swapped operand instead, so those dunders are never invoked.
    @overload
    def __eq__(self, other: "Attr") -> "Attr": ...
    @overload
    def __eq__(self, other: Scalar) -> "Attr": ...
    def __eq__(self, other: ExprInput) -> "Attr": ...

    @overload
    def __ne__(self, other: "Attr") -> "Attr": ...
    @overload
    def __ne__(self, other: Scalar) -> "Attr": ...
    def __ne__(self, other: ExprInput) -> "Attr": ...

    @overload
    def __lt__(self, other: "Attr") -> "Attr": ...
    @overload
    def __lt__(self, other: Scalar) -> "Attr": ...
    def __lt__(self, other: ExprInput) -> "Attr": ...

    @overload
    def __le__(self, other: "Attr") -> "Attr": ...
    @overload
    def __le__(self, other: Scalar) -> "Attr": ...
    def __le__(self, other: ExprInput) -> "Attr": ...

    @overload
    def __gt__(self, other: "Attr") -> "Attr": ...
    @overload
    def __gt__(self, other: Scalar) -> "Attr": ...
    def __gt__(self, other: ExprInput) -> "Attr": ...

    @overload
    def __ge__(self, other: "Attr") -> "Attr": ...
    @overload
    def __ge__(self, other: Scalar) -> "Attr": ...
    def __ge__(self, other: ExprInput) -> "Attr": ...


def _setup_ops() -> None:
    """Auto-generate dunder methods on `Attr` from operator tables.

    Arithmetic ops use `_delegate_operator` (clears `_filter`); comparison ops
    use `_delegate_comparison_operator` (sets `_filter` leaf when possible);
    boolean ops use `_delegate_boolean_operator` (builds compounds).
    """
    arith_ops = {
        "add": operator.add,
        "sub": operator.sub,
        "mul": operator.mul,
        "truediv": operator.truediv,
        "floordiv": operator.floordiv,
        "mod": operator.mod,
        "pow": operator.pow,
    }
    bool_ops = ("and", "or", "xor")
    comp_ops = {
        "eq": operator.eq,
        "ne": operator.ne,
        "lt": operator.lt,
        "le": operator.le,
        "gt": operator.gt,
        "ge": operator.ge,
    }

    for name, func in arith_ops.items():
        setattr(Attr, f"__{name}__", functools.partialmethod(Attr._delegate_operator, op=func, reverse=False))
        setattr(Attr, f"__r{name}__", functools.partialmethod(Attr._delegate_operator, op=func, reverse=True))

    for name in bool_ops:
        setattr(
            Attr, f"__{name}__", functools.partialmethod(Attr._delegate_boolean_operator, op_name=name, reverse=False)
        )
        setattr(
            Attr, f"__r{name}__", functools.partialmethod(Attr._delegate_boolean_operator, op_name=name, reverse=True)
        )

    for name, func in comp_ops.items():
        setattr(Attr, f"__{name}__", functools.partialmethod(Attr._delegate_comparison_operator, op=func))


_setup_ops()


class NodeAttr(Attr):
    """
    Wrapper of [Attr][tracksdata.attrs.Attr] to represent a node attribute.

    See Also
    --------
    [Attr][tracksdata.attrs.Attr]:
        The base class for all attributes.
    """


class EdgeAttr(Attr):
    """
    Wrapper of [Attr][tracksdata.attrs.Attr] to represent an edge attribute.

    See Also
    --------
    [Attr][tracksdata.attrs.Attr]:
        The base class for all attributes.
    """


def _filter_attr_kind(node: _FilterNode) -> type[Attr]:
    """Return the leaf-attribute kind (`NodeAttr` / `EdgeAttr` / `Attr`) of a filter node.

    Raises `ValueError` if the filter mixes `NodeAttr` and `EdgeAttr` leaves.
    The base `Attr` kind defers to any more specific kind present.
    """
    kinds = {leaf.kind for leaf in walk_leaves(node)}
    specific = {k for k in kinds if k is not Attr}
    if len(specific) > 1:
        raise ValueError(
            "A single compound filter cannot mix NodeAttr and EdgeAttr comparisons. "
            "Combine node and edge filters via separate positional arguments to graph.filter()."
        )
    return specific.pop() if specific else Attr


def split_attr_comps(
    attr_comps: Sequence["Attr"],
) -> tuple[list["Attr"], list["Attr"]]:
    """
    Split a list of filter-shaped Attrs into node and edge groups based on the
    kind of their leaf comparisons.

    Parameters
    ----------
    attr_comps : Sequence[Attr]
        The filter-shaped Attrs to split. Each must have `_filter` set (i.e.
        be built from comparisons + boolean ops).

    Returns
    -------
    tuple[list[Attr], list[Attr]]
        A tuple of lists of node and edge filters.
    """
    node_attr_comps: list[Attr] = []
    edge_attr_comps: list[Attr] = []

    for attr_comp in attr_comps:
        if not isinstance(attr_comp, Attr) or attr_comp._filter is None:
            raise ValueError(f"Expected a filter-shaped Attr (built from comparisons), got {type(attr_comp).__name__}.")
        kind = _filter_attr_kind(attr_comp._filter)
        if kind is NodeAttr:
            node_attr_comps.append(attr_comp)
        elif kind is EdgeAttr:
            edge_attr_comps.append(attr_comp)
        else:
            raise ValueError(f"Expected comparisons of 'NodeAttr' or 'EdgeAttr' objects, got {kind.__name__}.")

    return node_attr_comps, edge_attr_comps


def attr_comps_to_strs(attr_comps: Sequence["Attr"]) -> list[str]:
    """
    Convert a list of filter-shaped Attrs to the list of column names they
    reference, deduplicated while preserving order.
    """
    out: list[str] = []
    for attr_comp in attr_comps:
        if attr_comp._filter is None:
            continue
        for leaf in walk_leaves(attr_comp._filter):
            out.append(leaf.column)
    return list(dict.fromkeys(out))


def polars_reduce_attr_comps(
    df: pl.DataFrame,
    attr_comps: Sequence["Attr"],
    reduce_op: Callable[[Expr, Expr], Expr],
) -> pl.Expr:
    """
    Reduce a list of filter-shaped Attrs into a single polars expression,
    combined with `reduce_op` at the top level (AND-ed by default in callers).

    Parameters
    ----------
    df : pl.DataFrame
        Present for API compatibility; unused — each Attr already carries a
        fully-formed polars expression in `attr.expr`.
    attr_comps : Sequence[Attr]
        The filters to reduce.
    reduce_op : Callable[[Expr, Expr], Expr]
        The operation to reduce the top-level filters with.
    """
    if not attr_comps:
        raise ValueError("No attribute comparisons provided.")
    del df  # unused; kept for backward-compatible signature
    return pl.reduce(reduce_op, [a.expr for a in attr_comps])
