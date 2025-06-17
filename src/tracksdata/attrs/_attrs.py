from typing import TYPE_CHECKING, Any, Callable, override

from tracksdata.graph._base_graph import BaseGraph

if TYPE_CHECKING:
    from tracksdata.graph._rustworkx_graph import RustWorkXGraph
    from tracksdata.graph._sql_graph import SQLGraph
    from sqlalchemy.orm import Query


class Attr:
    def __init__(self, attr: str):
        self._attr = attr

    @override
    def to_filter(self, graph: RustWorkXGraph) -> Callable[dict[str, Any], bool]: ...

    @override
    def to_filter(self, graph: SQLGraph) -> Callable[[Query], Query]: ...

    def _to_dict_filter(self) -> Callable[[dict[str, Any]], bool]:
        ...

    def _to_query_filter(self) -> Callable[[Query], Query]:
        ...

    def to_filter(self, graph: BaseGraph) -> Callable[[dict[str, Any]], bool] | Callable[[Query], Query]:
        if isinstance(graph, RustWorkXGraph):
            return self._to_dict_filter()
        elif isinstance(graph, SQLGraph):
            return self._to_query_filter()
        else:
            raise ValueError(f"Unsupported graph backend: {type(graph)}")
