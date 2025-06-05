from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraphBackend
from tracksdata.graph._rustworkx_graph import RustWorkXGraphBackend


def map_ids(
    map: dict[int, int],
    indices: Sequence[int],
) -> list[int]:
    return [map[node_id] for node_id in indices]


class GraphView(RustWorkXGraphBackend):
    def __init__(
        self,
        root: BaseGraphBackend,
        sync: bool = True,
    ) -> None:
        super().__init__()
        self._root = root
        self._sync = sync

        self._node_map_to_root: dict[int, int] = {}
        self._node_map_from_root: dict[int, int] = {}

        self._edge_map_to_root: dict[int, int] = {}
        self._edge_map_from_root: dict[int, int] = {}

        # making sure these are not used
        # they should be accessed through the root graph
        self._node_features_keys = None
        self._edge_features_keys = None

    def node_ids(self) -> np.ndarray:
        indices = self.rx_graph.node_indices()
        return map_ids(self._node_map_to_root, indices)

    @property
    def node_features_keys(self) -> list[str]:
        return self._root.node_features_keys

    @property
    def edge_features_keys(self) -> list[str]:
        return self._root.edge_features_keys

    def add_node_feature_key(self, key: str, default_value: Any) -> None:
        self._root.add_node_feature_key(key, default_value)
        if self._sync:
            for node_id in self._graph.node_indices():
                self._graph[node_id][key] = default_value

    def add_edge_feature_key(self, key: str, default_value: Any) -> None:
        self._root.add_edge_feature_key(key, default_value)
        if self._sync:
            for _, _, edge_attr in self._graph.weighted_edge_list():
                edge_attr[key] = default_value

    def add_node(
        self,
        attributes: dict[str, Any],
        validate_keys: bool = True,
    ) -> int:
        parent_node_id = self._root.add_node(
            attributes=attributes,
            validate_keys=validate_keys,
        )

        if self._sync:
            node_id = super().add_node(
                attributes=attributes,
                validate_keys=validate_keys,
            )
            self._node_map_to_root[node_id] = parent_node_id
            self._node_map_from_root[parent_node_id] = node_id

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

        if self._sync:
            edge_id = super().add_edge(
                source_id=source_id,
                target_id=target_id,
                attributes=attributes,
                validate_keys=validate_keys,
            )
            self._edge_map_to_root[edge_id] = parent_edge_id
            self._edge_map_from_root[parent_edge_id] = edge_id

        return parent_edge_id

    def node_features(
        self,
        *,
        node_ids: Sequence[int] | None = None,
        feature_keys: Sequence[str] | str | None = None,
    ) -> pl.DataFrame:
        return self._root.node_features(
            node_ids=node_ids,
            feature_keys=feature_keys,
        )

    def edge_features(
        self,
        *,
        node_ids: Sequence[int] | None = None,
        feature_keys: Sequence[str] | str | None = None,
        include_targets: bool = False,
    ) -> pl.DataFrame:
        edges_df = super().edge_features(
            node_ids=node_ids,
            feature_keys=feature_keys,
            include_targets=include_targets,
        )

        # for id_key in [DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET]:
        #     edges_df[id_key] = map_ids(self._edge_map_to_root, edges_df[id_key])
        edges_df = edges_df.with_columns(
            {
                key: pl.col(key).map_elements(lambda x: self._edge_map_to_root[x])
                for key in [DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET]
            }
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
        if self._sync:
            super().update_node_features(
                node_ids=map_ids(self._node_map_from_root, node_ids),
                attributes=attributes,
            )

    def update_edge_features(
        self,
        edge_ids: Sequence[int],
        attributes: dict[str, Any],
    ) -> None:
        self._root.update_edge_features(
            edge_ids=edge_ids,
            attributes=attributes,
        )
        if self._sync:
            super().update_edge_features(
                edge_ids=map_ids(self._edge_map_from_root, edge_ids),
                attributes=attributes,
            )
