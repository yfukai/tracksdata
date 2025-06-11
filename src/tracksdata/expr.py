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

    def has_infinity_multiplication(self) -> bool:
        """
        Check if any column in the expression is multiplied by infinity.

        This method uses Polars' expression tree traversal via the pop() method
        and is_infinite() to detect patterns where a column is multiplied by infinity.

        Returns
        -------
        bool
            True if any column is multiplied by infinity, False otherwise.

        Examples
        --------
        >>> expr1 = AttrExpr("x") + math.inf * AttrExpr("y")
        >>> expr1.has_infinity_multiplication()  # True
        >>> expr2 = AttrExpr("x") + 2.5 * AttrExpr("y")
        >>> expr2.has_infinity_multiplication()  # False
        """
        return len(self.get_columns_multiplied_by_infinity()) > 0

    def get_columns_multiplied_by_infinity(self) -> list[str]:
        """
        Get the names of columns that are multiplied by infinity in the expression.

        This method uses Polars' expression tree traversal via the pop() method
        to recursively examine the expression structure and find columns that are
        involved in multiplication operations with infinity values.

        Returns
        -------
        list[str]
            List of column names that are multiplied by infinity.

        Examples
        --------
        >>> expr = AttrExpr("x") + math.inf * AttrExpr("y") + AttrExpr("z")
        >>> expr.get_columns_multiplied_by_infinity()  # ["y"]
        """
        import polars as pl

        columns_with_inf = []

        # Create a test dataframe for evaluation
        all_columns = self.column_names()
        if not all_columns:
            return columns_with_inf

        test_df = pl.DataFrame({col: [1.0] for col in all_columns})

        # Recursively traverse the expression tree
        self._find_infinity_columns(self.expr, test_df, columns_with_inf)

        # Remove duplicates and return
        return list(set(columns_with_inf))

    def _find_infinity_columns(self, expr, test_df, columns_with_inf):
        """
        Recursively traverse an expression tree to find columns multiplied by infinity.

        Parameters
        ----------
        expr : pl.Expr
            The expression to examine
        test_df : pl.DataFrame
            Test dataframe for evaluation
        columns_with_inf : list
            List to accumulate column names (modified in place)
        """
        try:
            # Get sub-expressions
            sub_expressions = expr.meta.pop()

            # If this expression has exactly 2 sub-expressions, it might be a binary operation
            if len(sub_expressions) == 2:
                left_expr, right_expr = sub_expressions

                # Check if one is infinite and the other is a column
                left_is_inf = self._is_infinite_literal(left_expr, test_df)
                right_is_inf = self._is_infinite_literal(right_expr, test_df)
                left_is_col = left_expr.meta.is_column()
                right_is_col = right_expr.meta.is_column()

                # Pattern 1: infinity * column
                if left_is_inf and right_is_col:
                    column_names = right_expr.meta.root_names()
                    columns_with_inf.extend(column_names)

                # Pattern 2: column * infinity
                elif left_is_col and right_is_inf:
                    column_names = left_expr.meta.root_names()
                    columns_with_inf.extend(column_names)

                # Recursively check sub-expressions
                self._find_infinity_columns(left_expr, test_df, columns_with_inf)
                self._find_infinity_columns(right_expr, test_df, columns_with_inf)

            # For expressions with other numbers of sub-expressions, recurse on all
            elif len(sub_expressions) > 0:
                for sub_expr in sub_expressions:
                    self._find_infinity_columns(sub_expr, test_df, columns_with_inf)

        except Exception:
            # If pop fails, this is likely a leaf node (column or literal)
            # No further traversal needed
            pass

    def _is_infinite_literal(self, expr, test_df):
        """
        Check if an expression is an infinite literal.

        Parameters
        ----------
        expr : pl.Expr
            Expression to check
        test_df : pl.DataFrame
            Test dataframe for evaluation

        Returns
        -------
        bool
            True if the expression is an infinite literal
        """
        try:
            # Check if it's a literal first (for efficiency)
            if not expr.meta.is_literal():
                return False

            # Test if it evaluates to infinity
            result = test_df.select(expr.is_infinite())
            return result.to_series().any()

        except Exception:
            return False

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
