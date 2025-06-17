import functools
import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from tracksdata.graph._base_graph import BaseGraph

if TYPE_CHECKING:
    from sqlalchemy.orm import DeclarativeBase, Query


__all__ = ["Attr", "AttrComparison", "AttrsFilter"]


class AttrComparison:
    def __init__(self, attr: "Attr", op: Callable, other: Any) -> None:
        self._attr = attr
        self._op = op
        self._other = other

    def __repr__(self) -> str:
        return f"Attr({self._attr}) '{self._op.__name__}' {self._other}"


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


class AttrsFilter:
    def __init__(self, graph: BaseGraph, attr_ops: list[AttrComparison] | AttrComparison) -> None:
        if isinstance(attr_ops, AttrComparison):
            attr_ops = [attr_ops]
        self._graph = graph
        self._attr_ops = attr_ops

    def dict_filter(self) -> Callable[[dict[str, Any]], bool]:
        def _filter(attrs: dict[str, Any]) -> bool:
            for attr_op in self._attr_ops:
                if not attr_op._op(attrs[str(attr_op._attr)], attr_op._other):
                    return False
            return True

        return _filter

    def query_filter(self, table: type["DeclarativeBase"]) -> Callable[["Query"], "Query"]:
        def _filter(query: "Query") -> "Query":
            query = query.filter(
                *[attr_op._op(getattr(table, str(attr_op._attr)), attr_op._other) for attr_op in self._attr_ops]
            )
            for attr_op in self._attr_ops:
                print(attr_op._op(getattr(table, str(attr_op._attr)), attr_op._other))
            return query

        return _filter
