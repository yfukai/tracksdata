import functools
import math
import operator
from collections.abc import Callable
from typing import Any, Union

import polars as pl
from polars import DataFrame, Expr, Series

Scalar = int | float | str | bool
ExprInput = Union[str, Scalar, "AttrExpr", Expr]


class AttrExpr:
    """
    A class to compose attribute expressions for graph attributes.

    Supports mathematical operations, comparisons, and polars-style transformations
    while automatically delegating method calls to the underlying polars expression.

    Parameters
    ----------
    value : ExprInput
        The value to compose the expression from:
        - str: Column name to reference from a DataFrame
        - Scalar: Literal value (int, float, str, bool)
        - AttrExpr: Another AttrExpr instance to copy
        - Expr: A polars expression to wrap

    Attributes
    ----------
    expr : Expr
        The underlying polars expression.

    Examples
    --------
    Create column references and literals:

    >>> expr = AttrExpr("iou")
    >>> expr = AttrExpr(1.0)

    Compose complex expressions:

    >>> expr = AttrExpr("iou").log()
    >>> expr = (1 - AttrExpr("iou")) * AttrExpr("distance")
    >>> expr = AttrExpr("score") > 0.5
    """

    expr: Expr

    def __init__(self, value: ExprInput) -> None:
        self._inf_exprs = []  # expressions multiplied by +inf
        self._neg_inf_exprs = []  # expressions multiplied by -inf

        if isinstance(value, str):
            self.expr = pl.col(value)
        elif isinstance(value, AttrExpr):
            self.expr = value.expr
            # Copy infinity tracking from the other AttrExpr
            self._inf_exprs = value.inf_exprs
            self._neg_inf_exprs = value.neg_inf_exprs
        elif isinstance(value, Expr):
            self.expr = value
        else:
            self.expr = pl.lit(value)

    def _wrap(self, expr: Expr | Any) -> Union["AttrExpr", Any]:
        """Wrap a polars expression in AttrExpr if it's an Expr, otherwise return as-is."""
        if isinstance(expr, Expr):
            result = AttrExpr(expr)
            # Propagate infinity tracking
            result._inf_exprs = self._inf_exprs.copy()
            result._neg_inf_exprs = self._neg_inf_exprs.copy()
            return result
        return expr

    def _delegate_operator(
        self, other: ExprInput, op: Callable[[Expr, Expr], Expr], reverse: bool = False
    ) -> "AttrExpr":
        """Delegate binary operations to polars expressions."""
        # Special handling for multiplication with infinity
        if op == operator.mul:
            # Check if we're multiplying with infinity scalar
            # In both reverse and non-reverse cases, 'other' is the infinity value
            # and 'self' is the AttrExpr we want to track
            if isinstance(other, int | float) and math.isinf(other):
                result = AttrExpr(pl.lit(0))  # Clean expression is zero (infinity term removed)

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
        left = AttrExpr(other).expr if reverse else self.expr
        right = self.expr if reverse else AttrExpr(other).expr
        result = AttrExpr(op(left, right))

        # Combine infinity tracking from both operands
        if isinstance(other, AttrExpr):
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

    def alias(self, name: str) -> "AttrExpr":
        """
        Alias the expression with a new name.

        Parameters
        ----------
        name : str
            The new name for the expression.

        Returns
        -------
        AttrExpr
            New AttrExpr with the aliased expression.
        """
        result = AttrExpr(self.expr.alias(name))
        result._inf_exprs = self._inf_exprs.copy()
        result._neg_inf_exprs = self._neg_inf_exprs.copy()
        return result

    def evaluate(self, df: DataFrame) -> Series:
        """
        Evaluate the expression against a DataFrame.

        Parameters
        ----------
        df : DataFrame
            The polars DataFrame to evaluate against.

        Returns
        -------
        Series
            The result as a polars Series.
        """
        return df.select(self.expr).to_series()

    @property
    def columns(self) -> list[str]:
        """
        Get the column names referenced by this expression.

        Returns
        -------
        list[str]
            List of column names that this expression depends on.
        """
        return list(dict.fromkeys(self.expr_columns + self.inf_columns + self.neg_inf_columns))

    @property
    def inf_exprs(self) -> list["AttrExpr"]:
        """Get the expressions multiplied by positive infinity."""
        return self._inf_exprs.copy()

    @property
    def neg_inf_exprs(self) -> list["AttrExpr"]:
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

    def __invert__(self) -> "AttrExpr":
        return AttrExpr(~self.expr)

    def __neg__(self) -> "AttrExpr":
        return AttrExpr(-self.expr)

    def __pos__(self) -> "AttrExpr":
        return AttrExpr(+self.expr)

    def __abs__(self) -> "AttrExpr":
        return AttrExpr(abs(self.expr))

    def __getattr__(self, attr: str) -> Any:
        """
        Delegate attribute access to the underlying polars expression.

        Enables access to all polars expression methods while maintaining
        the AttrExpr interface for method chaining.

        Parameters
        ----------
        attr : str
            The attribute name to access.

        Returns
        -------
        Any
            The requested attribute, with callable attributes wrapped to
            return AttrExpr instances when appropriate.
        """
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


def _add_operator(name: str, op: Callable, reverse: bool = False) -> None:
    """Add a binary operator method to the AttrExpr class."""
    method = functools.partialmethod(AttrExpr._delegate_operator, op=op, reverse=reverse)
    setattr(AttrExpr, name, method)


def _setup_ops() -> None:
    """
    Set up all binary operator methods for the AttrExpr class.

    Automatically creates all standard Python binary operators
    (__add__, __sub__, __mul__, etc.) and their reverse variants by
    delegating to the corresponding polars expression operators.
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
        "eq": operator.eq,
        "ne": operator.ne,
        "lt": operator.lt,
        "le": operator.le,
        "gt": operator.gt,
        "ge": operator.ge,
    }

    for op_name, op_func in bin_ops.items():
        _add_operator(f"__{op_name}__", op_func, reverse=False)
        _add_operator(f"__r{op_name}__", op_func, reverse=True)


_setup_ops()
