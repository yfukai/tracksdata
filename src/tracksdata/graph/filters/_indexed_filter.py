from collections.abc import Sequence
from typing import TYPE_CHECKING, overload

import bidict
import polars as pl

from tracksdata.attrs import AttrComparison
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._rustworkx_graph import (
    IndexedRXGraph,
    RXFilter,
    _create_filter_func,
)
from tracksdata.graph.filters._base_filter import cache_method

if TYPE_CHECKING:
    from tracksdata.graph._graph_view import GraphView


@overload
def map_ids(map: bidict.bidict[int, int], indices: Sequence[int]) -> list[int]: ...


@overload
def map_ids(map: bidict.bidict[int, int], indices: None) -> None: ...


def map_ids(
    map: bidict.bidict[int, int],
    indices: Sequence[int] | None,
) -> list[int] | None:
    if indices is None:
        return None

    if hasattr(indices, "tolist"):
        indices = indices.tolist()

    return [map[node_id] for node_id in indices]


def _map_df_ids(df: pl.DataFrame, map: bidict.bidict[int, int]) -> pl.DataFrame:
    for col in [DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).map_elements(map.__getitem__, return_dtype=pl.Int64).alias(col))
    return df


class IndexRXFilter(RXFilter):
    _graph: "GraphView | IndexedRXGraph"

    def __init__(
        self,
        *attr_comps: AttrComparison,
        graph: "GraphView | IndexedRXGraph",
        to_world_id_map: bidict.bidict[int, int],
        node_ids: Sequence[int] | None = None,
        include_targets: bool = False,
        include_sources: bool = False,
    ) -> None:
        super().__init__(
            *attr_comps,
            graph=graph,
            node_ids=node_ids,
            include_targets=include_targets,
            include_sources=include_sources,
        )
        self._to_world_id_map = to_world_id_map

    @cache_method
    def edge_attrs(self, attr_keys: list[str] | None = None, unpack: bool = False) -> pl.DataFrame:
        return _map_df_ids(
            super().edge_attrs(attr_keys, unpack),
            self._to_world_id_map,
        )

    @cache_method
    def node_ids(self) -> list[int]:
        indices = super().node_ids()
        return map_ids(self._to_world_id_map, indices)

    @cache_method
    def subgraph(
        self,
        node_attr_keys: Sequence[str] | str | None = None,
        edge_attr_keys: Sequence[str] | str | None = None,
    ) -> "GraphView":
        from tracksdata.graph._graph_view import GraphView

        node_ids = self.node_ids()

        rx_graph, node_map = self._graph._rx_subgraph_with_nodemap(node_ids)
        if self._edge_attr_comps:
            _filter_func = _create_filter_func(self._edge_attr_comps)
            for src, tgt, attr in rx_graph.weighted_edge_list():
                if not _filter_func(attr):
                    rx_graph.remove_edge(src, tgt)

        root = self._graph
        if hasattr(self._graph, "_root"):
            root = self._graph._root

        graph_view = GraphView(
            rx_graph,
            node_map_to_root=dict(node_map.items()),
            root=root,
        )

        return graph_view
