import abc
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import polars as pl
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from tracksdata.graph._graph_view import GraphView


T = TypeVar("T", bound="BaseGraph")


class BaseGraph(abc.ABC):
    """
    Base class for a graph backend.
    """

    @staticmethod
    def _validate_attributes(
        attributes: dict[str, Any],
        reference_keys: list[str],
        mode: str,
    ) -> None:
        """
        Validate the attributes of a node.

        Parameters
        ----------
        attributes : dict[str, Any]
            The attributes to validate.
        reference_keys : list[str]
            The keys to validate against.
        mode : str
            The mode to validate against, for example "node" or "edge".
        """
        for key in attributes.keys():
            if key not in reference_keys:
                raise ValueError(
                    f"{mode} feature key '{key}' not found in existing keys: "
                    f"'{reference_keys}'\nInitialize with "
                    f"`graph.add_{mode}_feature_key(key, default_value)`"
                )

        for ref_key in reference_keys:
            if ref_key not in attributes.keys():
                raise ValueError(
                    f"Attribute '{ref_key}' not found in attributes: "
                    f"'{attributes.keys()}'\nRequested keys: '{reference_keys}'"
                )

    @abc.abstractmethod
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

        TODO: should "t" be it's own parameter?

        Returns
        -------
        int
            The ID of the added node.
        """

    def bulk_add_nodes(
        self,
        nodes: list[dict[str, Any]],
    ) -> None:
        """
        Faster method to add multiple nodes to the graph with less overhead and fewer checks.

        Parameters
        ----------
        nodes : list[dict[str, Any]]
            The data of the nodes to be added.
            The keys of the data will be used as the attributes of the nodes.
            Must have "t" key.
        """
        # this method benefits the SQLGraph backend
        for node in nodes:
            self.add_node(node, validate_keys=False)

    @abc.abstractmethod
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
            Additional attributes for the edge.
        validate_keys : bool
            Whether to check if the attributes keys are valid.
            If False, the attributes keys will not be checked,
            useful to speed up the operation when doing bulk insertions.

        Returns
        -------
        int
            The ID of the added edge.
        """

    def bulk_add_edges(
        self,
        edges: list[dict[str, Any]],
    ) -> None:
        """
        Faster method to add multiple edges to the graph with less overhead and fewer checks.

        Parameters
        ----------
        edges : list[dict[str, Any]]
            The data of the edges to be added.
            The keys of the data will be used as the attributes of the edges.
            Must have "source_id" and "target_id" keys.
            For example:
            >>> `graph.bulk_add_edges([dict(source_id=0, target_id=1, weight=1.0)])`
        """
        # this method benefits the SQLGraph backend
        for edge in edges:
            self.add_edge(
                edge.pop("source_id"),
                edge.pop("target_id"),
                edge,
                validate_keys=False,
            )

    @abc.abstractmethod
    def sucessors(
        self,
        node_ids: list[int] | int,
        feature_keys: Sequence[str] | str | None = None,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame:
        """
        Get the sucessors of a list of nodes.

        Parameters
        ----------
        node_ids : list[int] | int
            The IDs of the nodes to get the sucessors for.
        feature_keys : Sequence[str] | str | None
            The feature keys to get.
            If None, all features are used.

        Returns
        -------
        dict[int, pl.DataFrame] | pl.DataFrame
            The sucessors of the nodes indexed by node ID if a list of nodes is provided.
        """

    @abc.abstractmethod
    def predecessors(
        self,
        node_ids: list[int] | int,
        feature_keys: Sequence[str] | str | None = None,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame:
        """
        Get the predecessors of a list of nodes.

        Parameters
        ----------
        node_ids : list[int] | int
            The IDs of the nodes to get the predecessors for.
        feature_keys : Sequence[str] | str | None
            The feature keys to get.
            If None, all features are used.

        Returns
        -------
        dict[int, pl.DataFrame] | pl.DataFrame
            The predecessors of the nodes indexed by node ID if a list of nodes is provided.
        """

    @abc.abstractmethod
    def filter_nodes_by_attribute(
        self,
        attributes: dict[str, Any],
    ) -> list[int]:
        """
        Filter nodes by attributes.

        Parameters
        ----------
        attributes : dict[str, Any]
            Attributes to filter by, for example:
            >>> `graph.filter_nodes_by_attribute(dict(t=0, label='A'))`

        Returns
        -------
        list[int]
            The IDs of the filtered nodes.
        """

    def _validate_subgraph_args(
        self,
        node_ids: Sequence[int] | None = None,
        node_attr_filter: dict[str, Any] | None = None,
        edge_attr_filter: dict[str, Any] | None = None,
    ) -> None:
        if node_ids is not None and (node_attr_filter is not None or edge_attr_filter is not None):
            raise ValueError("Node IDs and attributes' filters cannot be used together")

        if node_attr_filter is not None and edge_attr_filter is not None:
            raise ValueError("Node attributes' filters and edge attributes' filters cannot be used together")

        if node_ids is None and node_attr_filter is None and edge_attr_filter is None:
            raise ValueError("Either node IDs or one of the attributes' filters must be provided")

    @abc.abstractmethod
    def node_ids(self) -> list[int]:
        """
        Get the IDs of all nodes in the graph.
        """

    @abc.abstractmethod
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
            The feature keys to get.
            If None, all features are used.
        edge_feature_keys : Sequence[str] | str | None
            The feature keys to get.
            If None, all features are used.

        Returns
        -------
        GraphView
            A view of the graph with the specified nodes.
        """

    @abc.abstractmethod
    def time_points(self) -> list[int]:
        """
        Get the unique time points in the graph.
        """

    @abc.abstractmethod
    def node_features(
        self,
        *,
        node_ids: Sequence[int] | None = None,
        feature_keys: Sequence[str] | str | None = None,
    ) -> pl.DataFrame:
        """
        Get the features of the nodes as a pandas DataFrame.

        Parameters
        ----------
        node_ids : list[int] | None
            The IDs of the nodes to get the features for.
            If None, all nodes are used.
        feature_keys : Sequence[str] | str | None
            The feature keys to get.
            If None, all features are used.

        Returns
        -------
        pl.DataFrame
            A polars DataFrame with the features of the nodes.
        """

    @abc.abstractmethod
    def edge_features(
        self,
        *,
        node_ids: list[int] | None = None,
        feature_keys: Sequence[str] | None = None,
        include_targets: bool = False,
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
        include_targets : bool
            Whether to include edges out-going from the given node_ids even
            if the target node is not in the given node_ids.
        """

    @property
    @abc.abstractmethod
    def node_features_keys(self) -> list[str]:
        """
        Get the keys of the features of the nodes.
        """

    @property
    @abc.abstractmethod
    def edge_features_keys(self) -> list[str]:
        """
        Get the keys of the features of the edges.
        """

    @abc.abstractmethod
    def add_node_feature_key(self, key: str, default_value: Any) -> None:
        """
        Add a new feature key to the graph.
        All existing nodes will have the default value for the new feature key.
        """

    @abc.abstractmethod
    def add_edge_feature_key(self, key: str, default_value: Any) -> None:
        """
        Add a new feature key to the graph.
        All existing edges will have the default value for the new feature key.
        """

    @property
    @abc.abstractmethod
    def num_edges(self) -> int:
        """
        The number of edges in the graph.
        """

    @property
    @abc.abstractmethod
    def num_nodes(self) -> int:
        """
        The number of nodes in the graph.
        """

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    @classmethod
    def from_ctc(cls: type[T], data_dir: str | Path, **kwargs) -> T:
        """
        Create a graph from a CTC data directory.

        Parameters
        ----------
        data_dir : str | Path
            The path to the CTC data directory.
        **kwargs : Any
            Additional arguments to pass to the graph constructor.

        Examples
        --------
        >>> graph = BaseGraph.from_ctc("Fluo-N2DL-HeLa/01_GT/TRA")

        See Also
        --------
        tracksdata.io._ctc.load_ctc : Load a CTC ground truth file into a graph.
        RegionPropsNodes : Operator to create nodes from label images.

        Returns
        -------
        BaseGraph
            A graph with the nodes and edges from the CTC data directory.
        """
        from tracksdata.io._ctc import load_ctc

        graph = cls(**kwargs)
        load_ctc(data_dir, graph)
        return graph
