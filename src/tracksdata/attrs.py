import functools
import math
import operator
from collections.abc import Callable, Sequence
from typing import Any, Union

import polars as pl
from polars import DataFrame, Expr, Series

Scalar = int | float | str | bool
ExprInput = Union[str, Scalar, "Attr", Expr, "AttrComparison"]


__all__ = [
    "AttrComparison",
    "EdgeAttr",
    "NodeAttr",
    "attr_comps_to_strs",
    "polars_reduce_attr_comps",
    "split_attr_comps",
]


class AttrComparison:
    def __init__(self, attr: "Attr", op: Callable, other: Any) -> None:
        if attr.has_inf():
            raise ValueError("Comparison operators are not supported for expressions with infinity.")

        if isinstance(other, Attr):
            raise ValueError(f"Does not support comparison between expressions. Found {other} and {attr}.")

        columns = attr.expr_columns

        if len(columns) == 0:
            raise ValueError("Comparison operators are not supported for empty expressions.")

        elif len(columns) > 1:
            raise ValueError(f"Comparison operators are not supported for multiple columns. Found {columns}.")

        self.attr = attr
        self.column = columns[0]
        self.op = op
        self.other = other

    def __repr__(self) -> str:
        return f"Attr({self.column}) '{self.op.__name__}' {self.other}"

    def to_attr(self) -> "Attr":
        return Attr(self.op(pl.col(self.column), self.other))

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.to_attr(), attr)

    def _delegate_operator(self, other: ExprInput, op: Callable[[Expr, Expr], Expr], reverse: bool = False) -> "Attr":
        return self.to_attr()._delegate_operator(other, op, reverse)


class Attr:
    """
    A class to compose an attribute expression for graph attributes.

    Parameters
    ----------
    value : ExprInput
        The value to compose the attribute expression from.

    Examples
    --------
    >>> `AttrExpr("iou").log()`
    >>> `AttrExpr(1.0)`
    >>> `AttrExpr((1 - AttrExpr("iou")) * AttrExpr("distance"))`
    """

    expr: Expr

    def __init__(self, value: ExprInput) -> None:
        self._inf_exprs = []  # expressions multiplied by +inf
        self._neg_inf_exprs = []  # expressions multiplied by -inf

        if isinstance(value, str):
            self.expr = pl.col(value)
        elif isinstance(value, Attr):
            self.expr = value.expr
            # Copy infinity tracking from the other AttrExpr
            self._inf_exprs = value.inf_exprs
            self._neg_inf_exprs = value.neg_inf_exprs
        elif isinstance(value, AttrComparison):
            attr = value.to_attr()
            self.expr = attr.expr
            self._inf_exprs = attr.inf_exprs
            self._neg_inf_exprs = attr.neg_inf_exprs
        elif isinstance(value, Expr):
            self.expr = value
        else:
            self.expr = pl.lit(value)

    def _wrap(self, expr: Expr | Any) -> Union["Attr", Any]:
        if isinstance(expr, Expr):
            result = Attr(expr)
            # Propagate infinity tracking
            result._inf_exprs = self._inf_exprs.copy()
            result._neg_inf_exprs = self._neg_inf_exprs.copy()
            return result
        return expr

    def _delegate_operator(self, other: ExprInput, op: Callable[[Expr, Expr], Expr], reverse: bool = False) -> "Attr":
        # Special handling for multiplication with infinity
        if op == operator.mul:
            # Check if we're multiplying with infinity scalar
            # In both reverse and non-reverse cases, 'other' is the infinity value
            # and 'self' is the AttrExpr we want to track
            if isinstance(other, int | float) and math.isinf(other):
                result = Attr(pl.lit(0))  # Clean expression is zero (infinity term removed)

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
        result = Attr(op(left, right))

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

    def _delegate_comparison_operator(
        self,
        other: Any,
        op: Callable,
        reverse: bool = False,
    ) -> "AttrComparison | Attr":
        if isinstance(other, Attr):
            return self._delegate_operator(other, op, reverse)

        if reverse:
            lhs = other
            rhs = self
        else:
            lhs = self
            rhs = other
        return AttrComparison(lhs, op, rhs)

    def alias(self, name: str) -> "Attr":
        result = Attr(self.expr.alias(name))
        result._inf_exprs = self._inf_exprs.copy()
        result._neg_inf_exprs = self._neg_inf_exprs.copy()
        return result

    def evaluate(self, df: DataFrame) -> Series:
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
        columns = []
        for attr_expr in self._inf_exprs:
            columns.extend(attr_expr.columns)
        return list(dict.fromkeys(columns))

    @property
    def neg_inf_columns(self) -> list[str]:
        """Get the names of columns multiplied by negative infinity."""
        columns = []
        for attr_expr in self._neg_inf_exprs:
            columns.extend(attr_expr.columns)
        return list(dict.fromkeys(columns))

    def has_inf(self) -> bool:
        """
        Check if any column in the expression is multiplied by infinity or negative infinity.

        Returns
        -------
        bool
            True if any column is multiplied by infinity, False otherwise.
        """
        return self.has_pos_inf() or self.has_neg_inf()

    def has_pos_inf(self) -> bool:
        """
        Check if any column in the expression is multiplied by positive infinity.
        """
        return len(self._inf_exprs) > 0

    def has_neg_inf(self) -> bool:
        """
        Check if any column in the expression is multiplied by negative infinity.
        """
        return len(self._neg_inf_exprs) > 0

    def __invert__(self) -> "Attr":
        return Attr(~self.expr)

    def __neg__(self) -> "Attr":
        return Attr(-self.expr)

    def __pos__(self) -> "Attr":
        return Attr(+self.expr)

    def __abs__(self) -> "Attr":
        return Attr(abs(self.expr))

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
        return f"AttrExpr({self.expr})"


# Auto-generate operator methods using functools.partialmethod
def _add_operator(
    cls: type[Attr] | type[AttrComparison],
    name: str,
    op: Callable,
    reverse: bool = False,
) -> None:
    method = functools.partialmethod(cls._delegate_operator, op=op, reverse=reverse)
    setattr(cls, name, method)


def _add_comparison_operator(
    name: str,
    op: Callable,
    reverse: bool = False,
) -> None:
    method = functools.partialmethod(Attr._delegate_comparison_operator, op=op, reverse=reverse)
    setattr(Attr, name, method)


def _setup_ops() -> None:
    """
    Setup the operator methods for the AttrExpr class.
    """
    bin_ops = {
        "add": operator.add,
        "sub": operator.sub,
        "mul": operator.mul,
        "truediv": operator.truediv,
        "floordiv": operator.floordiv,
        "mod": operator.mod,
        "pow": operator.pow,
        "and": operator.and_,
        "or": operator.or_,
        "xor": operator.xor,
    }

    comp_ops = {
        "eq": operator.eq,
        "ne": operator.ne,
        "lt": operator.lt,
        "le": operator.le,
        "gt": operator.gt,
        "ge": operator.ge,
    }

    for op_name, op_func in bin_ops.items():
        _add_operator(Attr, f"__{op_name}__", op_func, reverse=False)
        _add_operator(Attr, f"__r{op_name}__", op_func, reverse=True)
        _add_operator(AttrComparison, f"__{op_name}__", op_func, reverse=False)
        _add_operator(AttrComparison, f"__r{op_name}__", op_func, reverse=True)

    for op_name, op_func in comp_ops.items():
        _add_comparison_operator(f"__{op_name}__", op_func, reverse=False)
        _add_comparison_operator(f"__r{op_name}__", op_func, reverse=True)

        # attrr_comparision uses normal delegate_operator
        _add_operator(AttrComparison, f"__{op_name}__", op_func, reverse=False)
        _add_operator(AttrComparison, f"__r{op_name}__", op_func, reverse=True)


_setup_ops()


class NodeAttr(Attr):
    """
    A class to represent a node attribute.
    """


class EdgeAttr(Attr):
    """
    A class to represent an edge attribute.
    """


def split_attr_comps(attr_comps: Sequence[AttrComparison]) -> tuple[list[AttrComparison], list[AttrComparison]]:
    """
    Split a list of attribute comparisons into node and edge attribute comparisons.

    """
    node_attr_comps = []
    edge_attr_comps = []

    for attr_comp in attr_comps:
        if isinstance(attr_comp.attr, NodeAttr):
            node_attr_comps.append(attr_comp)
        elif isinstance(attr_comp.attr, EdgeAttr):
            edge_attr_comps.append(attr_comp)
        else:
            raise ValueError(f"Expected comparisons of 'NodeAttr' or 'EdgeAttr' objects, got {type(attr_comp.attr)}")

    return node_attr_comps, edge_attr_comps


def attr_comps_to_strs(attr_comps: list[AttrComparison]) -> list[str]:
    """
    Convert a list of attribute comparisons to a list of strings.
    """
    return [str(attr_comp.column) for attr_comp in attr_comps]


def polars_reduce_attr_comps(df: pl.DataFrame, attr_comps: list[AttrComparison]) -> pl.Expr:
    """
    Reduce a list of attribute comparisons to a single polars expression.

    Parameters
    ----------
    df : pl.DataFrame
        The dataframe to reduce the attribute comparisons on.
    attr_comps : list[AttrComparison]
        The attribute comparisons to reduce.

    Returns
    -------
    pl.Expr
        The reduced polars expression.
    """
    return pl.reduce(
        lambda x, y: x & y, [attr_comp.op(df[str(attr_comp.column)], attr_comp.other) for attr_comp in attr_comps]
    )
