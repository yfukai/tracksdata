import functools
import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    pass


__all__ = [
    "Attr",
    "AttrComparison",
    "EdgeAttr",
    "NodeAttr",
    "attr_comps_to_strs",
    "polars_reduce_attr_comps",
    "split_attr_comps",
]


class AttrComparison:
    def __init__(self, attr: "Attr", op: Callable, other: Any) -> None:
        self.attr = attr
        self.op = op
        self.other = other

    def __repr__(self) -> str:
        return f"Attr({self.attr}) '{self._op.__name__}' {self._other}"


class Attr:
    def __init__(self, attr: str):
        self._attr = attr

    def __repr__(self) -> str:
        return self._attr

    def __str__(self) -> str:
        return self._attr

    def _delegate_operator(self, other: Any, op: Callable, reverse: bool = False) -> AttrComparison:
        if reverse:
            return AttrComparison(other, op, self)
        else:
            return AttrComparison(self, op, other)


def _add_operator(name: str, op: Callable, reverse: bool = False) -> None:
    method = functools.partialmethod(Attr._delegate_operator, op=op, reverse=reverse)
    setattr(Attr, name, method)


def _setup_ops() -> None:
    comp_ops = {
        "eq": operator.eq,
        "ne": operator.ne,
        "lt": operator.lt,
        "le": operator.le,
        "gt": operator.gt,
        "ge": operator.ge,
    }

    for op_name, op_func in comp_ops.items():
        _add_operator(f"__{op_name}__", op_func, reverse=False)
        _add_operator(f"__r{op_name}__", op_func, reverse=True)


_setup_ops()


class NodeAttr(Attr):
    """
    A class to represent a node attribute.
    """


class EdgeAttr(Attr):
    """
    A class to represent an edge attribute.
    """


def split_attr_comps(attr_comps: list[AttrComparison]) -> tuple[list[AttrComparison], list[AttrComparison]]:
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
    return [str(attr_comp.attr) for attr_comp in attr_comps]


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
        lambda x, y: x & y, [attr_comp._op(df[str(attr_comp.attr)], attr_comp._other) for attr_comp in attr_comps]
    )
