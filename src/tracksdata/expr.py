import functools
import operator
from collections.abc import Callable
from typing import Any, Union

import polars as pl
from polars import DataFrame, Expr, Series

Scalar = int | float | str | bool
ExprInput = Union[str, Scalar, "AttrExpr", Expr]


class AttrExpr:
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
        import math

        # Special handling for multiplication with infinity
        if op == operator.mul:
            # Check if we're multiplying with infinity
            inf_value = None
            expr_to_multiply = None

            if not reverse:
                # self * other
                if isinstance(other, (int, float)) and math.isinf(other):
                    inf_value = other
                    expr_to_multiply = self
                elif isinstance(other, AttrExpr):
                    # Check if other is infinity literal
                    other_val = self._extract_literal_value(other)
                    if other_val is not None and math.isinf(other_val):
                        inf_value = other_val
                        expr_to_multiply = self
            else:
                # other * self
                if isinstance(other, (int, float)) and math.isinf(other):
                    inf_value = other
                    expr_to_multiply = self

            # If we detected infinity multiplication, track it and return zero expression
            if inf_value is not None and expr_to_multiply is not None:
                result = AttrExpr(pl.lit(0))  # Clean expression is zero (infinity term removed)

                # Copy existing infinity tracking
                result._inf_exprs = self._inf_exprs.copy()
                result._neg_inf_exprs = self._neg_inf_exprs.copy()

                # Add the expression to appropriate infinity list
                if inf_value > 0:
                    result._inf_exprs.append(expr_to_multiply)
                else:
                    result._neg_inf_exprs.append(expr_to_multiply)

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

    def _extract_literal_value(self, expr: "AttrExpr") -> float | None:
        """Extract the literal value from an expression if it's a simple literal."""
        try:
            if expr.expr.meta.is_literal():
                # Try to evaluate it to get the value
                import polars as pl

                test_df = pl.DataFrame({"dummy": [1]})
                result = test_df.select(expr.expr).to_series()
                return result[0] if len(result) > 0 else None
        except:
            pass
        return None

    def alias(self, name: str) -> "AttrExpr":
        result = AttrExpr(self.expr.alias(name))
        result._inf_exprs = self._inf_exprs.copy()
        result._neg_inf_exprs = self._neg_inf_exprs.copy()
        return result

    def evaluate(self, df: DataFrame) -> Series:
        return df.select(self.expr).to_series()

    def column_names(self) -> list[str]:
        return self.expr.meta.root_names()

    @property
    def inf_exprs(self) -> list["AttrExpr"]:
        """Get the expressions multiplied by positive infinity."""
        return self._inf_exprs.copy()

    @property
    def neg_inf_exprs(self) -> list["AttrExpr"]:
        """Get the expressions multiplied by negative infinity."""
        return self._neg_inf_exprs.copy()

    @property
    def inf_columns(self) -> list[str]:
        """Get the names of columns multiplied by positive infinity."""
        columns = []
        for attr_expr in self._inf_exprs:
            try:
                if attr_expr.expr.meta.is_column():
                    columns.extend(attr_expr.column_names())
            except:
                pass
        return list(set(columns))

    @property
    def neg_inf_columns(self) -> list[str]:
        """Get the names of columns multiplied by negative infinity."""
        columns = []
        for attr_expr in self._neg_inf_exprs:
            try:
                if attr_expr.expr.meta.is_column():
                    columns.extend(attr_expr.column_names())
            except:
                pass
        return list(set(columns))

    def has_infinity_multiplication(self) -> bool:
        """
        Check if any column in the expression is multiplied by infinity.

        Returns
        -------
        bool
            True if any column is multiplied by infinity, False otherwise.
        """
        return len(self._inf_exprs) > 0 or len(self._neg_inf_exprs) > 0

    def get_columns_multiplied_by_infinity(self) -> list[str]:
        """
        Get the names of columns that are multiplied by infinity.

        Returns
        -------
        list[str]
            List of column names that are multiplied by any infinity (positive or negative).
        """
        return list(set(self.inf_columns + self.neg_inf_columns))

    def __invert__(self) -> "AttrExpr":
        return AttrExpr(~self.expr)

    def __neg__(self) -> "AttrExpr":
        return AttrExpr(-self.expr)

    def __pos__(self) -> "AttrExpr":
        return AttrExpr(+self.expr)

    def __abs__(self) -> "AttrExpr":
        return AttrExpr(abs(self.expr))

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


def _add_operator(name: str, op: Callable, reverse: bool = False) -> None:
    method = functools.partialmethod(AttrExpr._delegate_operator, op=op, reverse=reverse)
    setattr(AttrExpr, name, method)


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
