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
        if isinstance(value, str):
            self.expr = pl.col(value)
        elif isinstance(value, AttrExpr):
            self.expr = value.expr
        elif isinstance(value, Expr):
            self.expr = value
        else:
            self.expr = pl.lit(value)

    def _wrap(self, expr: Expr | Any) -> Union["AttrExpr", Any]:
        return AttrExpr(expr) if isinstance(expr, Expr) else expr

    def _delegate_operator(
        self, other: ExprInput, op: Callable[[Expr, Expr], Expr], reverse: bool = False
    ) -> "AttrExpr":
        left = AttrExpr(other).expr if reverse else self.expr
        right = self.expr if reverse else AttrExpr(other).expr
        return AttrExpr(op(left, right))

    def alias(self, name: str) -> "AttrExpr":
        return AttrExpr(self.expr.alias(name))

    def evaluate(self, df: DataFrame) -> Series:
        return df.select(self.expr).to_series()

    def column_names(self) -> list[str]:
        return self.expr.meta.root_names()

    def __invert__(self) -> "AttrExpr":
        return AttrExpr(~self.expr)

    def __neg__(self) -> "AttrExpr":
        return AttrExpr(-self.expr)

    def __pos__(self) -> "AttrExpr":
        return AttrExpr(+self.expr)

    def __abs__(self) -> "AttrExpr":
        return AttrExpr(abs(self.expr))

    def __getattr__(self, attr: str) -> Any:
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
    method = functools.partialmethod(
        AttrExpr._delegate_operator, op=op, reverse=reverse
    )
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
