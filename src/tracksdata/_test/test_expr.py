import math
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
    assert expr.columns() == ["test"]


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
    assert expr.columns() == ["test"]


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


def test_attr_expr_with_infinite() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    expr = (AttrExpr("a") == 1) * math.inf - math.inf * (AttrExpr("b") > 4) + AttrExpr("c")

    result = expr.evaluate(df)
    assert result.to_list() == [7, 8, 9]
    assert expr.column_names() == ["c"]

    assert len(expr.inf_exprs) == 1
    assert expr.inf_exprs[0].column_names() == ["a"]
    assert expr.inf_exprs[0].evaluate(df).to_list() == [True, False, False]

    assert len(expr.neg_inf_exprs) == 1
    assert expr.neg_inf_exprs[0].column_names() == ["b"]
    assert expr.neg_inf_exprs[0].evaluate(df).to_list() == [False, True, True]


def test_attr_expr_multiple_positive_infinity() -> None:
    """Test expression with multiple positive infinity terms."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    expr = AttrExpr("a") * math.inf + math.inf * AttrExpr("b") + AttrExpr("c")

    result = expr.evaluate(df)
    assert result.to_list() == [7, 8, 9]  # Only finite term remains
    assert expr.expr_columns == ["c"]

    assert len(expr.inf_exprs) == 2
    assert len(expr.neg_inf_exprs) == 0

    # Check both expressions are tracked
    inf_columns = set()
    for inf_expr in expr.inf_exprs:
        inf_columns.update(inf_expr.columns)
    assert inf_columns == {"a", "b"}


def test_attr_expr_multiple_negative_infinity() -> None:
    """Test expression with multiple negative infinity terms."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    expr = AttrExpr("a") * (-math.inf) - math.inf * AttrExpr("b") + AttrExpr("c")

    result = expr.evaluate(df)
    assert result.to_list() == [7, 8, 9]  # Only finite term remains
    assert expr.expr_columns == ["c"]

    assert len(expr.inf_exprs) == 0
    assert len(expr.neg_inf_exprs) == 2

    # Check both expressions are tracked
    neg_inf_columns = set()
    for neg_inf_expr in expr.neg_inf_exprs:
        neg_inf_columns.update(neg_inf_expr.columns)
    assert neg_inf_columns == {"a", "b"}


def test_attr_expr_only_infinity_terms() -> None:
    """Test expression with only infinity terms (no finite terms)."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    expr = AttrExpr("a") * math.inf - math.inf * AttrExpr("b")

    result = expr.evaluate(df)
    assert result.to_list() == [0]  # All infinity terms become literal zero
    assert expr.expr_columns == []  # No finite columns

    assert len(expr.inf_exprs) == 1
    assert len(expr.neg_inf_exprs) == 1
    assert expr.inf_exprs[0].columns == ["a"]
    assert expr.neg_inf_exprs[0].columns == ["b"]


def test_attr_expr_complex_infinity_expressions() -> None:
    """Test infinity with more complex expressions (not just boolean comparisons)."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    expr = (AttrExpr("a") + AttrExpr("b")) * math.inf - math.inf * (AttrExpr("c") / 2) + AttrExpr("a")

    result = expr.evaluate(df)
    assert result.to_list() == [1, 2, 3]  # Only AttrExpr("a") remains
    assert expr.columns == ["a"]

    assert len(expr.inf_exprs) == 1
    assert len(expr.neg_inf_exprs) == 1

    # Test that complex expressions are properly tracked
    inf_expr_result = expr.inf_exprs[0].evaluate(df)
    assert inf_expr_result.to_list() == [5, 7, 9]  # a + b
    assert set(expr.inf_exprs[0].columns) == {"a", "b"}

    neg_inf_expr_result = expr.neg_inf_exprs[0].evaluate(df)
    assert neg_inf_expr_result.to_list() == [3.5, 4.0, 4.5]  # c / 2
    assert expr.neg_inf_exprs[0].columns == ["c"]


def test_attr_expr_nested_infinity_operations() -> None:
    """Test nested operations involving infinity."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # Create intermediate expressions with infinity
    inf_expr = AttrExpr("a") * math.inf
    finite_expr = AttrExpr("b") + 1

    # Combine them
    combined = inf_expr + finite_expr

    result = combined.evaluate(df)
    assert result.to_list() == [5, 6, 7]  # Only (b + 1) remains
    assert combined.expr_columns == ["b"]
    assert set(combined.columns) == {"a", "b"}

    assert len(combined.inf_exprs) == 1
    assert len(combined.neg_inf_exprs) == 0
    assert combined.inf_exprs[0].columns == ["a"]


def test_attr_expr_infinity_column_name_tracking() -> None:
    """Test that inf_columns and neg_inf_columns properties work correctly."""
    df = pl.DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})

    # Simpler test: separate infinity operations that are easier to track
    expr = (
        AttrExpr("x") * math.inf  # positive infinity
        + AttrExpr("y") * math.inf  # positive infinity
        - math.inf * AttrExpr("z")  # negative infinity
        + AttrExpr("x")  # finite term
    )

    result = expr.evaluate(df)
    assert result.to_list() == [1, 2]  # Only finite AttrExpr("x") remains

    # Test the convenience properties
    assert set(expr.inf_columns) == {"x", "y"}  # x and y in positive infinity
    assert set(expr.neg_inf_columns) == {"z"}  # z in negative infinity

    assert expr.has_inf() is True

    # Check individual expressions
    assert len(expr.inf_exprs) == 2  # x and y
    assert len(expr.neg_inf_exprs) == 1  # z


def test_attr_expr_no_infinity_terms() -> None:
    """Test that expressions without infinity work normally."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    expr = AttrExpr("a") * 2 + AttrExpr("b") - 1

    result = expr.evaluate(df)
    assert result.to_list() == [5, 8, 11]  # 2*a + b - 1 = 2*[1,2,3] + [4,5,6] - 1 = [2,4,6] + [4,5,6] - 1 = [5,8,11]

    # No infinity terms should be tracked
    assert len(expr.inf_exprs) == 0
    assert len(expr.neg_inf_exprs) == 0
    assert expr.inf_columns == []
    assert expr.neg_inf_columns == []
    assert expr.has_inf() is False


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


def test_duplicated_columns() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    expr = (AttrExpr("a") == 1) * 10 - 5 * (AttrExpr("a") > 2)
    result = expr.evaluate(df)
    assert result.to_list() == [10, 0, -5]
    assert expr.columns == ["a"]
