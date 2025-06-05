from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import rustworkx as rx
from numpy.typing import ArrayLike

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.utils._logging import LOG

if TYPE_CHECKING:
    from tracksdata.graph._graph_view import GraphView


class RustWorkXGraph(BaseGraph):
    def __init__(self, rx_graph: rx.PyDiGraph | None = None) -> None:
        """
        TODO
        """
        super().__init__()

        if rx_graph is None:
            self._graph = rx.PyDiGraph()
        else:
            self._graph = rx_graph

        self._time_to_nodes: dict[int, list[int]] = {}
        self._node_features_keys: list[str] = [DEFAULT_ATTR_KEYS.T]
        self._edge_features_keys: list[str] = []

    @property
    def rx_graph(self) -> rx.PyDiGraph:
        return self._graph

    def add_node(
        self,
        attributes: dict[str, Any],
        validate_keys: bool = True,
    ) -> int:
        """
        Add a node to the graph at time t.

        Parameters
        ----------
        attributes : Any
            The attributes of the node to be added, must have a "t" key.
            The keys of the attributes will be used as the attributes of the node.
            For example:
            >>> `graph.add_node(dict(t=0, label='A', intensity=100))`
        validate_keys : bool
            Whether to check if the attributes keys are valid.
            If False, the attributes keys will not be checked,
            useful to speed up the operation when doing bulk insertions.
        """
        # avoiding copying attributes on purpose, it could be a problem in the future
        if validate_keys:
            self._validate_attributes(attributes, self.node_features_keys, "node")

            if "t" not in attributes:
                raise ValueError(f"Node attributes must have a 't' key. Got {attributes.keys()}")

        node_id = self.rx_graph.add_node(attributes)
        self._time_to_nodes.setdefault(attributes["t"], []).append(node_id)
        return node_id

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        attributes: dict[str, Any],
        validate_keys: bool = True,
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
        validate_keys : bool
            Whether to check if the attributes keys are valid.
            If False, the attributes keys will not be checked,
            useful to speed up the operation when doing bulk insertions.
        """
        if validate_keys:
            self._validate_attributes(attributes, self.edge_features_keys, "edge")
        edge_id = self.rx_graph.add_edge(source_id, target_id, attributes)
        attributes[DEFAULT_ATTR_KEYS.EDGE_ID] = edge_id
        return edge_id

    def filter_nodes_by_attribute(
        self,
        attributes: dict[str, Any],
    ) -> np.ndarray:
        """
        Filter nodes by attributes.

        Parameters
        ----------
        attributes : dict[str, Any]
            The attributes to filter by, for example:
            >>> `graph.filter_nodes_by_attribute(dict(t=0, label='A'))`

        Returns
        -------
        np.ndarray
            The IDs of the filtered nodes.
        """
        rx_graph = self.rx_graph
        node_map = None
        # entire graph
        if DEFAULT_ATTR_KEYS.T in attributes:
            selected_nodes = self._time_to_nodes.get(attributes.pop(DEFAULT_ATTR_KEYS.T), [])
            if len(attributes) == 0:
                return selected_nodes

            # subgraph of selected nodes
            rx_graph, node_map = rx_graph.subgraph_with_nodemap(selected_nodes)
            # node_map = np.asarray(selected_nodes)

        def _filter_func(node_attr: dict[str, Any]) -> bool:
            for key, value in attributes.items():
                if node_attr[key] != value:
                    return False
            return True

        if node_map is None:
            return list(rx_graph.filter_nodes(_filter_func))
        else:
            return [node_map[n] for n in rx_graph.filter_nodes(_filter_func)]

    def node_ids(self) -> np.ndarray:
        """
        Get the IDs of all nodes in the graph.
        """
        return np.asarray(list(self.rx_graph.node_indices()), dtype=int)

    def subgraph(
        self,
        *,
        node_ids: Sequence[int] | None = None,
        node_attr_filter: dict[str, Any] | None = None,
        edge_attr_filter: dict[str, Any] | None = None,
        node_feature_keys: Sequence[str] | str | None = None,
        edge_feature_keys: Sequence[str] | str | None = None,
    ) -> "GraphView":
        """
        Create a subgraph from the graph from the given node IDs
        or attributes' filters.

        Node IDs or a single attribute filter can be used to create a subgraph.

        Parameters
        ----------
        node_ids : Sequence[int]
            The IDs of the nodes to include in the subgraph.
        node_attr_filter : dict[str, Any] | None
            The attributes to filter the nodes by.
        edge_attr_filter : dict[str, Any] | None
            The attributes to filter the edges by.
        node_feature_keys : Sequence[str] | str | None
            The feature keys to include in the subgraph.
        edge_feature_keys : Sequence[str] | str | None
            The feature keys to include in the subgraph.

        Returns
        -------
        RustWorkXGraph
            A new graph with the specified nodes.
        """
        from tracksdata.graph._graph_view import GraphView

        if node_ids is not None and (node_attr_filter is not None or edge_attr_filter is not None):
            raise ValueError("Node IDs and attributes' filters cannot be used together")

        if node_attr_filter is not None and edge_attr_filter is not None:
            raise ValueError("Node attributes' filters and edge attributes' filters cannot be used together")

        if node_ids is None and node_attr_filter is None and edge_attr_filter is None:
            raise ValueError("Either node IDs or one of the attributes' filters must be provided")

        if edge_attr_filter is not None:
            edges_df = self.edge_features(feature_keys=edge_attr_filter.keys())
            mask = pl.reduce(lambda x, y: x & y, [edges_df[key] == value for key, value in edge_attr_filter.items()])
            node_ids = np.unique(
                edges_df.filter(mask)
                .select(
                    DEFAULT_ATTR_KEYS.EDGE_SOURCE,
                    DEFAULT_ATTR_KEYS.EDGE_TARGET,
                )
                .to_numpy()
            )
        elif node_attr_filter is not None:
            node_ids = self.filter_nodes_by_attribute(node_attr_filter)

        rx_graph, node_map = self.rx_graph.subgraph_with_nodemap(node_ids)

        if edge_attr_filter is not None:
            LOG.info(f"Removing edges without attributes {edge_attr_filter}")
            for source, target, edge_attr in rx_graph.weighted_edge_list():
                for key, value in edge_attr_filter.items():
                    if edge_attr[key] != value:
                        rx_graph.remove_edge(source, target)
                        break

        graph_view = GraphView(
            rx_graph=rx_graph,
            node_map_to_root=dict(node_map.items()),
            root=self,
        )

        return graph_view

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
        if key in self.node_features_keys:
            raise ValueError(f"Feature key {key} already exists")

        self._node_features_keys.append(key)
        rx_graph = self.rx_graph
        for node_id in rx_graph.node_indices():
            rx_graph[node_id][key] = default_value

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
        if key in self.edge_features_keys:
            raise ValueError(f"Feature key {key} already exists")

        self._edge_features_keys.append(key)
        for _, _, edge_attr in self.rx_graph.weighted_edge_list():
            edge_attr[key] = default_value

    def node_features(
        self,
        *,
        node_ids: Sequence[int] | None = None,
        feature_keys: Sequence[str] | str | None = None,
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
        rx_graph = self.rx_graph
        # If no node_ids provided, use all nodes
        if node_ids is None:
            node_ids = list(rx_graph.node_indices())

        if len(node_ids) == 0:
            raise ValueError("Empty graph, there are no nodes to get features from")

        if feature_keys is None:
            feature_keys = self.node_features_keys

        if isinstance(feature_keys, str):
            feature_keys = [feature_keys]

        # Create columns directly instead of building intermediate dictionaries
        columns = {key: [] for key in feature_keys}

        # Build columns in a vectorized way
        for node_id in node_ids:
            node_data = rx_graph[node_id]
            for key in feature_keys:
                columns[key].append(node_data[key])

        for key in feature_keys:
            columns[key] = np.asarray(columns[key])

        # Create DataFrame and set node_id as index in one shot
        return pl.DataFrame(columns)

    def edge_features(
        self,
        *,
        node_ids: list[int] | None = None,
        feature_keys: Sequence[str] | str | None = None,
        include_targets: bool = False,
    ) -> pl.DataFrame:
        """
        Get the features of the edges as a polars DataFrame.

        Parameters
        ----------
        node_ids : list[int] | None
            The IDs of the subgraph to get the edge features for.
            If None, all edges of the graph are used.
        feature_keys : Sequence[str] | str | None
            The feature keys to get.
            If None, all features are used.
        include_targets : bool
            Whether to include edges out-going from the given node_ids even
            if the target node is not in the given node_ids.
        """
        if feature_keys is None:
            feature_keys = self.edge_features_keys

        feature_keys = [DEFAULT_ATTR_KEYS.EDGE_ID, *feature_keys]

        if node_ids is None:
            rx_graph = self.rx_graph
            node_map = None
        else:
            if include_targets:
                selected_nodes = set(node_ids)
                for node_id in node_ids:
                    neighbors = self.rx_graph.neighbors(node_id)
                    selected_nodes.update(neighbors)
                node_ids = list(selected_nodes)

            rx_graph, node_map = self.rx_graph.subgraph_with_nodemap(node_ids)

        edge_map = rx_graph.edge_index_map()
        if len(edge_map) == 0:
            return pl.DataFrame(
                {
                    key: []
                    for key in [
                        *feature_keys,
                        DEFAULT_ATTR_KEYS.EDGE_SOURCE,
                        DEFAULT_ATTR_KEYS.EDGE_TARGET,
                    ]
                }
            )

        source, target, data = zip(*edge_map.values(), strict=False)

        if node_map is not None:
            source = [node_map[s] for s in source]
            target = [node_map[t] for t in target]

        columns = {key: [] for key in feature_keys}

        for row in data:
            for key in feature_keys:
                columns[key].append(row[key])

        columns[DEFAULT_ATTR_KEYS.EDGE_SOURCE] = source
        columns[DEFAULT_ATTR_KEYS.EDGE_TARGET] = target

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
        *,
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
                raise ValueError(f"Node feature key '{key}' not found in graph. Expected '{self.node_features_keys}'")

            if not np.isscalar(value) and len(attributes[key]) != len(node_ids):
                raise ValueError(
                    f"Attribute '{key}' has wrong size. Expected {len(node_ids)}, got {len(attributes[key])}"
                )

        for key, value in attributes.items():
            if np.isscalar(value):
                value = np.full(len(node_ids), value)

            for node_id, v in zip(node_ids, value, strict=False):
                self._graph[node_id][key] = v

    def update_edge_features(
        self,
        *,
        edge_ids: ArrayLike,
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
                raise ValueError(f"Edge feature key '{key}' not found in graph. Expected '{self.edge_features_keys}'")

            if np.isscalar(value):
                attributes[key] = [value] * size

            elif len(attributes[key]) != size:
                raise ValueError(f"Attribute '{key}' has wrong size. Expected {size}, got {len(attributes[key])}")

        edge_map = self._graph.edge_index_map()

        for i, edge_id in enumerate(edge_ids):
            edge_attr = edge_map[edge_id][2]  # 0=source, 1=target, 2=attributes
            for key, value in attributes.items():
                edge_attr[key] = value[i]
