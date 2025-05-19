from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
import rustworkx as rx

from tracksdata.graph._base_graph import BaseGraphBackend, BaseReadOnlyGraph

# TODO:
# - use a better name for the default graph backend


class RustWorkXReadOnlyGraph(BaseReadOnlyGraph):
    def __init__(
        self,
        graph: rx.PyDiGraph | None = None,
    ) -> None:
        """
        TODO
        """
        if graph is None:
            self._graph = graph
        else:
            self._graph = rx.PyDiGraph()


class RustWorkXGraphBackend(BaseGraphBackend):
    def __init__(self) -> None:
        """
        TODO
        """
        self._graph = rx.PyDiGraph()
        self._time_to_nodes: dict[int, list[int]] = {}
        self._node_features_keys: list[str] = []
        self._edge_features_keys: list[str] = []

    def add_node(
        self,
        *,
        t: int,
        **kwargs: Any,
    ) -> int:
        """
        Add a node to the graph at time t.

        Parameters
        ----------
        t : int
            The time at which to add the node.
        **kwargs : Any
            The attributes of the node to be added.
            The keys of the kwargs will be used as the attributes of the node.
            For example:
            >>> `graph.add_node(t=0, label='A', intensity=100)`
        """
        # avoiding copying kwargs on purpose, it could be a problem in the future
        kwargs["t"] = t
        node_id = self._graph.add_node(kwargs)
        self._time_to_nodes.setdefault(t, []).append(node_id)
        return node_id

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        attributes: dict[str, Any],
    ) -> int:
        """
        Add an edge to the graph.

        Parameters
        ----------
        source_id : int
            The ID of the source node.
        target_id : int
            The ID of the target node.
        attributes : dict[str, Any]
            The attributes of the edge to be added.
            The keys of the attributes will be used as the attributes of the edge.
        """
        for key in attributes.keys():
            if key not in self.edge_features_keys:
                raise ValueError(f"Edge feature key {key} not found")

        edge_id = self._graph.add_edge(source_id, target_id, attributes)
        return edge_id

    def filter_nodes_by_attribute(
        self,
        **kwargs: Any,
    ) -> list[int]:
        # TODO doc

        if "t" in kwargs:
            selected_nodes = self._time_to_nodes.get(kwargs.pop("t"), [])
            if len(kwargs) == 0:
                return selected_nodes

            # FIXME: need to decide what's going to be the readonly graph
            return self.subgraph(selected_nodes).filter_nodes_by_attribute(**kwargs)

        def _filter_func(node_id: int) -> bool:
            for key, value in kwargs.items():
                try:
                    if self._graph[node_id][key] != value:
                        return False
                except KeyError:
                    return False
            return True

        return list(self._graph.filter_nodes(_filter_func))

    def node_ids(self) -> list[int]:
        """
        Get the IDs of all nodes in the graph.
        """
        return list(self._graph.node_indices())

    def subgraph(
        self,
        node_ids: Sequence[int],
    ) -> RustWorkXReadOnlyGraph:
        subgraph = self._graph.subgraph(node_ids)
        return RustWorkXReadOnlyGraph(graph=subgraph)

    def time_points(self) -> list[int]:
        """
        Get the unique time points in the graph.
        """
        return list(self._time_to_nodes.keys())

    @property
    def node_features_keys(self) -> list[str]:
        """
        Get the keys of the features of the nodes.
        """
        return self._node_features_keys

    @property
    def edge_features_keys(self) -> list[str]:
        """
        Get the keys of the features of the edges.
        """
        return self._edge_features_keys

    def add_node_feature_key(self, key: str, default_value: Any) -> None:
        """
        Add a new feature key to the graph.
        All existing nodes will have the default value for the new feature key.

        Parameters
        ----------
        key : str
            The key of the new feature.
        default_value : Any
            The default value for existing nodes for the new feature key.
        """
        if key in self._node_features_keys:
            raise ValueError(f"Feature key {key} already exists")

        self._node_features_keys.append(key)
        for node_id in self._graph.node_indices():
            self._graph[node_id][key] = default_value

    def add_edge_feature_key(self, key: str, default_value: Any) -> None:
        """
        Add a new feature key to the graph.
        All existing edges will have the default value for the new feature key.

        Parameters
        ----------
        key : str
            The key of the new feature.
        default_value : Any
            The default value for existing edges for the new feature key.
        """
        if key in self._edge_features_keys:
            raise ValueError(f"Feature key {key} already exists")

        self._edge_features_keys.append(key)
        for _, _, edge_attr in self._graph.weighted_edge_list():
            edge_attr[key] = default_value

    def node_features(
        self,
        node_ids: Sequence[int] | None = None,
        feature_keys: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """
        Get the features of the nodes as a polars DataFrame.

        Parameters
        ----------
        node_ids : list[int] | None
            The IDs of the nodes to get the features for.
            If None, all nodes are used.
        feature_keys : Sequence[str] | None
            The feature keys to get.
            If None, all the features of the first node are used.

        Returns
        -------
        pl.DataFrame
            A polars DataFrame with the features of the nodes.
        """
        # If no node_ids provided, use all nodes
        if node_ids is None:
            node_ids = list(self._graph.node_indices())

        if len(node_ids) == 0:
            raise ValueError("Empty graph, there are no nodes to get features from")

        if feature_keys is None:
            feature_keys = self.node_features_keys

        # Create columns directly instead of building intermediate dictionaries
        columns = {key: [] for key in feature_keys}

        # Build columns in a vectorized way
        for node_id in node_ids:
            node_data = self._graph[node_id]
            for key in feature_keys:
                columns[key].append(node_data.get(key))

        for key in feature_keys:
            columns[key] = np.asarray(columns[key])

        # Create DataFrame and set node_id as index in one shot
        return pl.DataFrame(columns)

    def edge_features(
        self,
        node_ids: list[int] | None = None,
        feature_keys: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """
        Get the features of the edges as a polars DataFrame.

        Parameters
        ----------
        node_ids : list[int] | None
            The IDs of the subgraph to get the edge features for.
            If None, all edges of the graph are used.
        feature_keys : Sequence[str] | None
            The feature keys to get.
            If None, all features are used.
        """
        if node_ids is None:
            graph = self._graph
        else:
            selected_nodes = set(node_ids)
            for node_id in node_ids:
                neighbors = self._graph.neighbors(node_id)
                selected_nodes.update(neighbors)

            graph = self._graph.subgraph(list(selected_nodes))

        if feature_keys is None:
            feature_keys = self.edge_features_keys

        edge_map = graph.edge_index_map()
        if len(edge_map) == 0:
            return pl.DataFrame(
                {key: [] for key in ["edge_id", "source", "target", *feature_keys]}
            )

        source, target, data = zip(*edge_map.values(), strict=False)

        columns = {key: [] for key in feature_keys}
        columns["edge_id"] = list(edge_map.keys())
        columns["source"] = source
        columns["target"] = target

        for row in data:
            for key in feature_keys:
                columns[key].append(row[key])

        columns = {k: np.asarray(v) for k, v in columns.items()}

        return pl.DataFrame(columns)

    @property
    def num_edges(self) -> int:
        """
        The number of edges in the graph.
        """
        return self._graph.num_edges()

    @property
    def num_nodes(self) -> int:
        """
        The number of nodes in the graph.
        """
        return self._graph.num_nodes()

    def update_node_features(
        self,
        node_ids: Sequence[int],
        attributes: dict[str, Any],
    ) -> None:
        """
        Update the features of the nodes.

        Parameters
        ----------
        node_ids : Sequence[int]
            The IDs of the nodes to update.
        attributes : dict[str, Any]
            The attributes to update.
        """
        for key, value in attributes.items():
            if key not in self.node_features_keys:
                raise ValueError(
                    f"Node feature key '{key}' not found in graph. "
                    f"Expected '{self.node_features_keys}'"
                )

            if not np.isscalar(value) and len(attributes[key]) != len(node_ids):
                raise ValueError(
                    f"Attribute '{key}' has wrong size. "
                    f"Expected {len(node_ids)}, got {len(attributes[key])}"
                )

        for key, value in attributes.items():
            if np.isscalar(value):
                value = np.full(len(node_ids), value)

            for node_id, v in zip(node_ids, value, strict=False):
                self._graph[node_id][key] = v

    def update_edge_features(
        self,
        edge_ids: Sequence[int],
        attributes: dict[str, Any],
    ) -> None:
        """
        Update the features of the edges.

        Parameters
        ----------
        edge_ids : Sequence[int]
            The IDs of the edges to update.
        attributes : dict[str, Any]
            Attributes to be updated.
        """
        size = len(edge_ids)
        for key, value in attributes.items():
            if key not in self.edge_features_keys:
                raise ValueError(
                    f"Edge feature key '{key}' not found in graph. "
                    f"Expected '{self.edge_features_keys}'"
                )

            if np.isscalar(value):
                attributes[key] = np.full(size, value)

            elif len(attributes[key]) != size:
                raise ValueError(
                    f"Attribute '{key}' has wrong size. "
                    f"Expected {size}, got {len(attributes[key])}"
                )

        edge_map = self._graph.edge_index_map()

        for i, edge_id in enumerate(edge_ids):
            edge_attr = edge_map[edge_id][2]  # 0=source, 1=target, 2=attributes
            for key, value in attributes.items():
                edge_attr[key] = value[i]
