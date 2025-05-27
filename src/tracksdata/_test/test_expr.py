import operator
from collections.abc import Callable

import polars as pl
import pytest

from tracksdata.expr import AttrExpr


def test_attr_expr_init_with_string() -> None:
    expr = AttrExpr("test")
    assert isinstance(expr.expr, pl.Expr)
    assert expr.expr.meta.root_names() == ["test"]


def test_attr_expr_init_with_scalar() -> None:
    expr = AttrExpr(1.0)
    assert isinstance(expr.expr, pl.Expr)
    # Literal expressions don't have root names
    assert expr.expr.meta.root_names() == []


def test_attr_expr_init_with_attr_expr() -> None:
    expr1 = AttrExpr("test")
    expr2 = AttrExpr(expr1).sqrt()
    assert isinstance(expr2.expr, pl.Expr)
    assert expr2.expr.meta.root_names() == ["test"]


def test_attr_expr_init_with_polars_expr() -> None:
    pl_expr = pl.col("test")
    expr = AttrExpr(pl_expr)
    assert isinstance(expr.expr, pl.Expr)
    assert expr.expr.meta.root_names() == ["test"]


def test_attr_expr_evaluate() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    expr = AttrExpr("a") + AttrExpr("b")
    result = expr.evaluate(df)
    assert isinstance(result, pl.Series)
    assert result.to_list() == [5, 7, 9]


def test_attr_expr_column_names() -> None:
    expr = AttrExpr("test")
    assert expr.column_names() == ["test"]


@pytest.mark.parametrize(
    "op,func",
    [
        (operator.neg, lambda x: -x),
        (operator.pos, lambda x: +x),
        (operator.abs, abs),
        (operator.invert, lambda x: ~x),
    ],
)
def test_attr_expr_unary_operators(op: Callable, func: Callable) -> None:
    df = pl.DataFrame({"a": [-1, 2, -3]})
    expr = op(AttrExpr("a"))
    result = expr.evaluate(df)
    expected = pl.Series([func(x) for x in df["a"]])
    assert result.to_list() == expected.to_list()


@pytest.mark.parametrize(
    "op,func",
    [
        (operator.add, lambda x, y: x + y),
        (operator.sub, lambda x, y: x - y),
        (operator.mul, lambda x, y: x * y),
        (operator.truediv, lambda x, y: x / y),
        (operator.floordiv, lambda x, y: x // y),
        (operator.mod, lambda x, y: x % y),
        (operator.pow, lambda x, y: x**y),
        (operator.eq, lambda x, y: x == y),
        (operator.ne, lambda x, y: x != y),
        (operator.lt, lambda x, y: x < y),
        (operator.le, lambda x, y: x <= y),
        (operator.gt, lambda x, y: x > y),
        (operator.ge, lambda x, y: x >= y),
    ],
)
def test_attr_expr_binary_operators(op: Callable, func: Callable) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    expr = op(AttrExpr("a"), AttrExpr("b"))
    result = expr.evaluate(df)
    expected = pl.Series([func(x, y) for x, y in zip(df["a"], df["b"], strict=False)])
    assert result.to_list() == expected.to_list()


def test_attr_expr_alias() -> None:
    expr = AttrExpr("test").alias("new_name")
    assert isinstance(expr, AttrExpr)
    # Note: alias doesn't change root names
    assert expr.column_names() == ["test"]


def test_attr_expr_method_delegation() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    expr = AttrExpr("a").log(2)
    result = expr.evaluate(df)
    expected = df.select(pl.col("a").log(2)).to_series()
    assert result.to_list() == expected.to_list()


def test_attr_expr_complex_expression() -> None:
    df = pl.DataFrame({"iou": [0.5, 0.7, 0.9], "distance": [10, 20, 30]})
    expr = (1 - AttrExpr("iou")) * AttrExpr("distance")
    result = expr.evaluate(df)
    expected = [(1 - iou) * dist for iou, dist in zip(df["iou"], df["distance"], strict=False)]
    assert result.to_list() == expected


def test_attr_expr_scalar_operations() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    expr = AttrExpr("a") * 2
    result = expr.evaluate(df)
    assert result.to_list() == [2, 4, 6]

    expr = 2 * AttrExpr("a")  # Test reverse operation
    result = expr.evaluate(df)
    assert result.to_list() == [2, 4, 6]


def test_attr_expr_boolean_operations() -> None:
    df = pl.DataFrame({"a": [True, False, True], "b": [False, True, True]})

    expr = AttrExpr("a") & AttrExpr("b")  # and
    result = expr.evaluate(df)
    assert result.to_list() == [False, False, True]

    expr = AttrExpr("a") | AttrExpr("b")  # or
    result = expr.evaluate(df)
    assert result.to_list() == [True, True, True]
