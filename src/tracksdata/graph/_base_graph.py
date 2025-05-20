import abc
from collections.abc import Sequence
from typing import Any

import polars as pl

# NOTE:
# - maybe a single basegraph is better
# - nodes have a t, and space


class BaseReadOnlyGraph(abc.ABC):  # noqa: B024
    """
    Base class for viewing a graph.
    """


class BaseWritableGraph(BaseReadOnlyGraph):
    """
    Base class for writing to a graph.
    """

    # TODO


class BaseGraphBackend(abc.ABC):
    """
    Base class for a graph backend.
    """

    @abc.abstractmethod
    def add_node(
        self,
        *,
        t: int,
        **kwargs: Any,
    ) -> int:
        """
        Add a node to the graph.

        Parameters
        ----------
        t : int
            The time of the node.
        kwargs : Any
            Additional attributes for the node.

        TODO: make add additional attributes
            - x and y
            - z=0
            - mask=None
            - bbox=None
        thoughts?

        Returns
        -------
        int
            The ID of the added node.
        """

    @abc.abstractmethod
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
            Additional attributes for the edge.

        Returns
        -------
        int
            The ID of the added edge.
        """

    @abc.abstractmethod
    def node_ids(self) -> list[int]:
        """
        Get the IDs of all nodes in the graph.
        """

    @abc.abstractmethod
    def filter_nodes_by_attribute(
        self,
        **kwargs: Any,
    ) -> list[int]:
        """
        Filter nodes by attributes.

        Parameters
        ----------
        kwargs : Any
            Attributes to filter by.

        Returns
        -------
        BaseGraphBackend
            A new graph with the filtered nodes.
        """

    @abc.abstractmethod
    def subgraph(
        self,
        node_ids: Sequence[int],
    ) -> "BaseReadOnlyGraph":
        """
        Create a subgraph from the graph from the given node IDs.

        Parameters
        ----------
        node_ids : Sequence[int]
            The IDs of the nodes to include in the subgraph.

        Returns
        -------
        BaseReadOnlyGraph
            A new graph with the specified nodes.
        """

    @abc.abstractmethod
    def time_points(self) -> list[int]:
        """
        Get the unique time points in the graph.
        """

    @abc.abstractmethod
    def node_features(
        self,
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
