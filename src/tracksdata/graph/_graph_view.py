from collections.abc import Callable, Sequence
from typing import Any

import polars as pl
import rustworkx as rx

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.functional._rx import graph_track_ids
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.graph._rustworkx_graph import RustWorkXGraph


def map_ids(
    map: dict[int, int],
    indices: Sequence[int] | None,
) -> list[int] | None:
    if indices is None:
        return None

    if hasattr(indices, "tolist"):
        indices = indices.tolist()

    return [map[node_id] for node_id in indices]


class GraphView(RustWorkXGraph):
    def __init__(
        self,
        rx_graph: rx.PyDiGraph,
        node_map_to_root: dict[int, int],
        root: BaseGraph,
        sync: bool = True,
    ) -> None:
        super().__init__(rx_graph=rx_graph)

        # setting up the time_to_nodes mapping
        for idx in rx_graph.node_indices():
            t = self.rx_graph[idx][DEFAULT_ATTR_KEYS.T]
            if t not in self._time_to_nodes:
                self._time_to_nodes[t] = []
            self._time_to_nodes[t].append(idx)

        self._node_map_to_root = node_map_to_root.copy()
        self._node_map_from_root = {v: k for k, v in node_map_to_root.items()}

        self._edge_map_to_root: dict[int, int] = {
            idx: data[DEFAULT_ATTR_KEYS.EDGE_ID] for idx, (_, _, data) in self.rx_graph.edge_index_map().items()
        }
        self._edge_map_from_root: dict[int, int] = {v: k for k, v in self._edge_map_to_root.items()}

        self._root = root
        self._is_root_rx_graph = isinstance(root, RustWorkXGraph)
        self._sync = sync
        self._out_of_sync = False

        # making sure these are not used
        # they should be accessed through the root graph
        self._node_features_keys = None
        self._edge_features_keys = None

    @property
    def sync(self) -> bool:
        return self._sync

    @sync.setter
    def sync(self, value: bool) -> None:
        if value and not self._sync:
            raise ValueError("Cannot sync a graph view that is not synced\nRe-create the graph view.")
        self._sync = value

    def node_ids(self) -> list[int]:
        indices = self.rx_graph.node_indices()
        return map_ids(self._node_map_to_root, indices)

    def subgraph(
        self,
        *,
        node_ids: Sequence[int] | None = None,
        node_attr_filter: dict[str, Any] | None = None,
        edge_attr_filter: dict[str, Any] | None = None,
        node_feature_keys: Sequence[str] | str | None = None,
        edge_feature_keys: Sequence[str] | str | None = None,
    ) -> "GraphView":
        subgraph = super().subgraph(
            node_ids=map_ids(self._node_map_from_root, node_ids),
            node_attr_filter=node_attr_filter,
            edge_attr_filter=edge_attr_filter,
            node_feature_keys=node_feature_keys,
            edge_feature_keys=edge_feature_keys,
        )

        subgraph._root = self._root

        subgraph._node_map_to_root = {k: self._node_map_to_root[v] for k, v in subgraph._node_map_to_root.items()}
        subgraph._node_map_from_root = {v: k for k, v in subgraph._node_map_to_root.items()}

        subgraph._edge_map_to_root = {k: self._edge_map_to_root[v] for k, v in subgraph._edge_map_to_root.items()}
        subgraph._edge_map_from_root = {v: k for k, v in subgraph._edge_map_to_root.items()}

        return subgraph

    @property
    def node_features_keys(self) -> list[str]:
        return self._root.node_features_keys

    @property
    def edge_features_keys(self) -> list[str]:
        return self._root.edge_features_keys

    def add_node_feature_key(self, key: str, default_value: Any) -> None:
        self._root.add_node_feature_key(key, default_value)
        # because attributes are passed by reference, we need don't need if both are rustworkx graphs
        if self.sync and not self._is_root_rx_graph:
            rx_graph = self.rx_graph
            for node_id in rx_graph.node_indices():
                rx_graph[node_id][key] = default_value
        else:
            self._out_of_sync |= not self._is_root_rx_graph

    def add_edge_feature_key(self, key: str, default_value: Any) -> None:
        self._root.add_edge_feature_key(key, default_value)
        # because attributes are passed by reference, we need don't need if both are rustworkx graphs
        if self.sync and not self._is_root_rx_graph:
            for _, _, edge_attr in self.rx_graph.weighted_edge_list():
                edge_attr[key] = default_value
        else:
            self._out_of_sync |= not self._is_root_rx_graph

    def add_node(
        self,
        attributes: dict[str, Any],
        validate_keys: bool = True,
    ) -> int:
        parent_node_id = self._root.add_node(
            attributes=attributes,
            validate_keys=validate_keys,
        )

        if self.sync:
            node_id = super().add_node(
                attributes=attributes,
                validate_keys=validate_keys,
            )
            self._node_map_to_root[node_id] = parent_node_id
            self._node_map_from_root[parent_node_id] = node_id
        else:
            self._out_of_sync = True

        return parent_node_id

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        attributes: dict[str, Any],
        validate_keys: bool = True,
    ) -> int:
        parent_edge_id = self._root.add_edge(
            source_id=source_id,
            target_id=target_id,
            attributes=attributes,
            validate_keys=validate_keys,
        )
        attributes[DEFAULT_ATTR_KEYS.EDGE_ID] = parent_edge_id

        if self.sync:
            # it does not set the EDGE_ID as attribute as the super().add_edge
            edge_id = self.rx_graph.add_edge(
                self._node_map_from_root[source_id],
                self._node_map_from_root[target_id],
                attributes,
            )
            self._edge_map_to_root[edge_id] = parent_edge_id
            self._edge_map_from_root[parent_edge_id] = edge_id
        else:
            self._out_of_sync = True

        return parent_edge_id

    def _get_neighbors(
        self,
        neighbors_func: Callable[[rx.PyDiGraph, int], rx.NodeIndices],
        node_ids: list[int] | int,
        feature_keys: Sequence[str] | str | None = None,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame:
        single_node = False
        if isinstance(node_ids, int):
            node_ids = [node_ids]
            single_node = True

        node_ids = map_ids(self._node_map_from_root, node_ids)
        neighbors_data = super()._get_neighbors(neighbors_func, node_ids, feature_keys)

        out_data = {}
        for node_id in node_ids:
            df = neighbors_data[node_id]
            out_data[self._node_map_to_root[node_id]] = self._map_to_root_df_node_ids(df)

        if single_node:
            return next(iter(out_data.values()))

        return out_data

    def sucessors(
        self,
        node_ids: list[int] | int,
        feature_keys: Sequence[str] | str | None = None,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame:
        if self._out_of_sync:
            raise RuntimeError("Out of sync graph view cannot be used to get sucessors")
        return super().sucessors(node_ids, feature_keys)

    def predecessors(
        self,
        node_ids: list[int] | int,
        feature_keys: Sequence[str] | str | None = None,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame:
        if self._out_of_sync:
            raise RuntimeError("Out of sync graph view cannot be used to get predecessors")
        return super().predecessors(node_ids, feature_keys)

    def filter_nodes_by_attribute(
        self,
        attributes: dict[str, Any],
    ) -> list[int]:
        node_ids = super().filter_nodes_by_attribute(
            attributes=attributes,
        )
        return map_ids(self._node_map_to_root, node_ids)

    def _map_to_root_df_node_ids(self, df: pl.DataFrame) -> pl.DataFrame:
        if DEFAULT_ATTR_KEYS.NODE_ID in df.columns:
            df = df.with_columns(
                pl.col(DEFAULT_ATTR_KEYS.NODE_ID)
                .map_elements(lambda x: self._node_map_to_root[x], return_dtype=pl.Int64)
                .alias(DEFAULT_ATTR_KEYS.NODE_ID)
            )
        return df

    def node_features(
        self,
        *,
        node_ids: Sequence[int] | None = None,
        feature_keys: Sequence[str] | str | None = None,
    ) -> pl.DataFrame:
        node_dfs = super().node_features(
            node_ids=map_ids(self._node_map_from_root, node_ids),
            feature_keys=feature_keys,
        )
        node_dfs = self._map_to_root_df_node_ids(node_dfs)
        return node_dfs

    def edge_features(
        self,
        *,
        node_ids: Sequence[int] | None = None,
        feature_keys: Sequence[str] | str | None = None,
        include_targets: bool = False,
    ) -> pl.DataFrame:
        edges_df = super().edge_features(
            node_ids=map_ids(self._node_map_from_root, node_ids),
            feature_keys=feature_keys,
            include_targets=include_targets,
        )

        edges_df = edges_df.with_columns(
            *[
                pl.col(key).map_elements(lambda x: self._node_map_to_root[x], return_dtype=pl.Int64).alias(key)
                for key in [DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET]
            ]
        )

        return edges_df

    def update_node_features(
        self,
        node_ids: Sequence[int],
        attributes: dict[str, Any],
    ) -> None:
        self._root.update_node_features(
            node_ids=node_ids,
            attributes=attributes,
        )
        # because attributes are passed by reference, we need don't need if both are rustworkx graphs
        if self.sync and not self._is_root_rx_graph:
            super().update_node_features(
                node_ids=map_ids(self._node_map_from_root, node_ids),
                attributes=attributes,
            )
        else:
            self._out_of_sync |= not self._is_root_rx_graph

    def update_edge_features(
        self,
        edge_ids: Sequence[int],
        attributes: dict[str, Any],
    ) -> None:
        self._root.update_edge_features(
            edge_ids=edge_ids,
            attributes=attributes,
        )
        # because attributes are passed by reference, we need don't need if both are rustworkx graphs
        if self.sync and not self._is_root_rx_graph:
            super().update_edge_features(
                edge_ids=map_ids(self._edge_map_from_root, edge_ids),
                attributes=attributes,
            )
        else:
            self._out_of_sync |= not self._is_root_rx_graph

    def assign_track_ids(
        self,
        output_key: str = DEFAULT_ATTR_KEYS.TRACK_ID,
    ) -> rx.PyDiGraph:
        """
        Compute and assign track ids to nodes.

        Parameters
        ----------
        output_key : str
            The key of the output track id attribute.

        Returns
        -------
        rx.PyDiGraph
            A compressed graph (parent -> child) with track ids lineage relationships.
        """
        try:
            node_ids, track_ids, tracks_graph = graph_track_ids(self.rx_graph)
        except RuntimeError as e:
            raise RuntimeError(
                "Are you sure this graph is a valid lineage graph?\n"
                "This function expects a solved graph.\n"
                "Often used from `graph.filter_nodes_by_attribute({'solution': True})`"
            ) from e

        node_ids = map_ids(self._node_map_to_root, node_ids)

        if output_key not in self.node_features_keys:
            self.add_node_feature_key(output_key, -1)

        self.update_node_features(
            node_ids=node_ids,
            attributes={output_key: track_ids},
        )

        return tracks_graph
