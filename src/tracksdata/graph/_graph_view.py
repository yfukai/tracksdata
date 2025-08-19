from collections.abc import Callable, Sequence
from typing import Any, Literal, overload

import bidict
import polars as pl
import rustworkx as rx

from tracksdata.attrs import AttrComparison
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.functional._rx import _assign_track_ids
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.graph._mapped_graph_mixin import MappedGraphMixin
from tracksdata.graph._rustworkx_graph import IndexedRXGraph, RustWorkXGraph, RXFilter
from tracksdata.graph.filters._indexed_filter import IndexRXFilter
from tracksdata.utils._logging import LOG


class GraphView(RustWorkXGraph, MappedGraphMixin):
    """
    A filtered view of a graph that maintains bidirectional mapping to the root graph.

    GraphView provides a lightweight way to work with subsets of a larger graph
    while maintaining the ability to synchronize changes back to the original graph.
    It acts as a view layer that maps between local node/edge IDs and the root
    graph's IDs, enabling efficient subgraph operations with minimal data duplication.

    Parameters
    ----------
    rx_graph : rx.PyDiGraph
        The rustworkx graph object representing the subgraph.
    node_map_to_root : dict[int, int]
        Mapping from local node IDs to root graph node IDs.
    root : BaseGraph
        Reference to the root graph that this view is derived from.
    sync : bool, default True
        Whether to automatically synchronize changes in the view.
        By default only the root graph is updated.

    Attributes
    ----------
    _local_to_external : bidict.bidict[int, int]
        Mapping from local node IDs to external node IDs (via MappedGraphMixin).
    _external_to_local : bidict.bidict[int, int]
        Mapping from external node IDs to local node IDs (via MappedGraphMixin).
    _edge_map_to_root : bidict.bidict[int, int]
        Mapping from view edge IDs to root graph edge IDs.
    _edge_map_from_root : bidict.bidict[int, int]
        Mapping from root graph edge IDs to view edge IDs.

    See Also
    --------
    [RustWorkXGraph][tracksdata.graph.RustWorkXGraph]:
        The base graph implementation that this view extends.

    [SQLGraph][tracksdata.graph.SQLGraph]:
        Database-backed graph implementation for larger datasets.

    Examples
    --------
    Create a subgraph view filtered by time:

    ```python
    from tracksdata.attrs import NodeAttr

    view = graph.filter(NodeAttr("t") == 5).subgraph()
    ```

    Access nodes in the view:

    ```python
    node_ids = view.node_ids()
    node_attrs = view.node_attrs(node_ids=node_ids)
    ```

    The view automatically maps between local and root IDs when needed.
    """

    def __init__(
        self,
        rx_graph: rx.PyDiGraph,
        node_map_to_root: dict[int, int],
        root: BaseGraph,
        sync: bool = True,
    ) -> None:
        # Initialize RustWorkXGraph
        RustWorkXGraph.__init__(self, rx_graph=None)  # rx_graph is not used to avoid initialization
        self._graph = rx_graph

        # Initialize MappedGraphMixin
        MappedGraphMixin.__init__(self, node_map_to_root)

        # Setting up the time_to_nodes mapping (this was removed accidentally)
        for idx in rx_graph.node_indices():
            t = self.rx_graph[idx][DEFAULT_ATTR_KEYS.T]
            self._time_to_nodes.setdefault(t, []).append(idx)

        # Set up edge mapping (nodes handled by mixin)
        self._edge_map_to_root: bidict.bidict[int, int] = bidict.bidict(
            (idx, data[DEFAULT_ATTR_KEYS.EDGE_ID]) for idx, (_, _, data) in self.rx_graph.edge_index_map().items()
        )
        self._edge_map_from_root = self._edge_map_to_root.inverse

        self._root = root
        self._is_root_rx_graph = isinstance(root, RustWorkXGraph)
        self._sync = sync
        self._out_of_sync = False

        # making sure these are not used
        # they should be accessed through the root graph
        self._node_attr_keys = None
        self._edge_attr_keys = None

        # use parent graph overlaps
        self._overlaps = None

    @property
    def supports_custom_indices(self) -> bool:
        return self._root.supports_custom_indices

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
        return self._map_to_external(indices)

    def edge_ids(self) -> list[int]:
        indices = self.rx_graph.edge_indices()
        # Map edge indices using the edge mapping
        if indices is None:
            return None
        return [self._edge_map_to_root[idx] for idx in indices]

    def add_overlap(
        self,
        source_id: int,
        target_id: int,
    ) -> int:
        """
        Add a new overlap to the graph.
        Overlapping nodes are mutually exclusive.

        Parameters
        ----------
        source_id : int
            The ID of the source node.
        target_id : int
            The ID of the target node.

        Returns
        -------
        int
            The ID of the added overlap.
        """
        return self._root.add_overlap(source_id, target_id)

    def bulk_add_overlaps(
        self,
        overlaps: list[list[int, 2]],
    ) -> None:
        """
        Add multiple overlaps to the graph.
        Overlapping nodes are mutually exclusive.

        Parameters
        ----------
        overlaps : list[list[int, 2]]
            The IDs of the nodes to add the overlaps for.

        See Also
        --------
        [add_overlap][tracksdata.graph.GraphView.add_overlap]:
            Add a single overlap to the graph.
        """
        self._root.bulk_add_overlaps(overlaps)

    def overlaps(
        self,
        node_ids: list[int] | None = None,
    ) -> list[list[int, 2]]:
        """
        Get the overlaps between the nodes in `node_ids`.
        If `node_ids` is None, all nodes are used.

        Parameters
        ----------
        node_ids : list[int] | None
            The IDs of the nodes to get the overlaps for.
            If None, all nodes are used.

        Returns
        -------
        list[list[int, 2]]
            The overlaps between the nodes in `node_ids`.
        """
        if node_ids is None:
            node_ids = self.node_ids()
        return self._root.overlaps(node_ids)

    def filter(
        self,
        *attr_filters: AttrComparison,
        node_ids: Sequence[int] | None = None,
        include_targets: bool = False,
        include_sources: bool = False,
    ) -> RXFilter:
        return IndexRXFilter(
            *attr_filters,
            graph=self,
            node_ids=self._map_to_local(node_ids),
            include_targets=include_targets,
            include_sources=include_sources,
        )

    @property
    def node_attr_keys(self) -> list[str]:
        return self._root.node_attr_keys

    @property
    def edge_attr_keys(self) -> list[str]:
        return self._root.edge_attr_keys

    def add_node_attr_key(self, key: str, default_value: Any) -> None:
        self._root.add_node_attr_key(key, default_value)
        # because attributes are passed by reference, we need don't need if both are rustworkx graphs
        if not self._is_root_rx_graph:
            if self.sync:
                rx_graph = self.rx_graph
                for node_id in rx_graph.node_indices():
                    rx_graph[node_id][key] = default_value
            else:
                self._out_of_sync = True

    def add_edge_attr_key(self, key: str, default_value: Any) -> None:
        self._root.add_edge_attr_key(key, default_value)
        # because attributes are passed by reference, we need don't need if both are rustworkx graphs
        if not self._is_root_rx_graph:
            if self.sync:
                for _, _, edge_attr in self.rx_graph.weighted_edge_list():
                    edge_attr[key] = default_value
            else:
                self._out_of_sync = True

    def add_node(
        self,
        attrs: dict[str, Any],
        validate_keys: bool = True,
        index: int | None = None,
    ) -> int:
        parent_node_id = self._root.add_node(
            attrs=attrs,
            validate_keys=validate_keys,
            index=index,
        )

        if self.sync:
            node_id = RustWorkXGraph.add_node(
                self,
                attrs=attrs,
                validate_keys=validate_keys,
            )
            self._add_id_mapping(node_id, parent_node_id)
        else:
            self._out_of_sync = True

        return parent_node_id

    def bulk_add_nodes(self, nodes: list[dict[str, Any]], indices: list[int] | None = None) -> list[int]:
        parent_node_ids = self._root.bulk_add_nodes(nodes, indices=indices)
        if self.sync:
            node_ids = RustWorkXGraph.bulk_add_nodes(self, nodes)
            self._add_id_mappings(list(zip(node_ids, parent_node_ids, strict=True)))
        else:
            self._out_of_sync = True
        return parent_node_ids

    def remove_node(self, node_id: int) -> None:
        """
        Remove a node from the graph.

        This method removes the node from both the view and the root graph,
        along with all connected edges. Also updates the node mappings.

        Parameters
        ----------
        node_id : int
            The ID of the node to remove.

        Raises
        ------
        ValueError
            If the node_id does not exist in the graph.
        """
        if node_id not in self._external_to_local:
            raise ValueError(f"Node {node_id} does not exist in the graph")

        # Remove from root graph first
        self._root.remove_node(node_id)

        if self.sync:
            # Get the local node ID and remove from local graph
            local_node_id = self._external_to_local[node_id]

            super().remove_node(local_node_id)

            # Remove the node mapping
            self._remove_id_mapping(external_id=node_id)

            # Update edge mappings - remove edges involving this node
            edges_to_remove = []
            edge_indices = self.rx_graph.edge_indices()
            for local_edge_id, _ in list(self._edge_map_to_root.items()):
                # Check if this edge is still in the local graph
                if local_edge_id not in edge_indices:
                    edges_to_remove.append(local_edge_id)

            for edge_id in edges_to_remove:
                if edge_id in self._edge_map_to_root:
                    del self._edge_map_to_root[edge_id]
        else:
            self._out_of_sync = True

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        attrs: dict[str, Any],
        validate_keys: bool = True,
    ) -> int:
        parent_edge_id = self._root.add_edge(
            source_id=source_id,
            target_id=target_id,
            attrs=attrs,
            validate_keys=validate_keys,
        )
        attrs[DEFAULT_ATTR_KEYS.EDGE_ID] = parent_edge_id

        if self.sync:
            # it does not set the EDGE_ID as attribute as the super().add_edge
            edge_id = self.rx_graph.add_edge(
                self._map_to_local(source_id),
                self._map_to_local(target_id),
                attrs,
            )
            self._edge_map_to_root.put(edge_id, parent_edge_id)
        else:
            self._out_of_sync = True

        return parent_edge_id

    def bulk_add_edges(self, edges: list[dict[str, Any]], return_ids: bool = False) -> list[int] | None:
        return BaseGraph.bulk_add_edges(self, edges, return_ids=return_ids)

    def _get_neighbors(
        self,
        neighbors_func: Callable[[rx.PyDiGraph, int], rx.NodeIndices],
        node_ids: list[int] | int,
        attr_keys: Sequence[str] | str | None = None,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame:
        single_node = False
        if isinstance(node_ids, int):
            node_ids = [node_ids]
            single_node = True

        node_ids = self._map_to_local(node_ids)
        neighbors_data = super()._get_neighbors(neighbors_func, node_ids, attr_keys)

        out_data = {}
        for node_id in node_ids:
            df = neighbors_data[node_id]
            # Map DataFrame IDs from local to external using mixin method
            mapped_df = self._map_df_to_external(
                df, [DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET]
            )
            out_data[self._map_to_external(node_id)] = mapped_df

        if single_node:
            return next(iter(out_data.values()))

        return out_data

    def successors(
        self,
        node_ids: list[int] | int,
        attr_keys: Sequence[str] | str | None = None,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame:
        if self._out_of_sync:
            raise RuntimeError("Out of sync graph view cannot be used to get sucessors")
        return super().successors(node_ids, attr_keys)

    def predecessors(
        self,
        node_ids: list[int] | int,
        attr_keys: Sequence[str] | str | None = None,
    ) -> dict[int, pl.DataFrame] | pl.DataFrame:
        if self._out_of_sync:
            raise RuntimeError("Out of sync graph view cannot be used to get predecessors")
        return super().predecessors(node_ids, attr_keys)

    def _node_attrs_from_node_ids(
        self,
        *,
        node_ids: list[int] | None = None,
        attr_keys: Sequence[str] | str | None = None,
        unpack: bool = False,
    ) -> pl.DataFrame:
        node_dfs = super()._node_attrs_from_node_ids(
            node_ids=self._map_to_local(node_ids),
            attr_keys=attr_keys,
            unpack=unpack,
        )
        node_dfs = self._map_df_to_external(
            node_dfs, [DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET]
        )
        return node_dfs

    def node_attrs(
        self,
        *,
        attr_keys: Sequence[str] | str | None = None,
        unpack: bool = False,
    ) -> pl.DataFrame:
        return self._node_attrs_from_node_ids(attr_keys=attr_keys, unpack=unpack)

    def edge_attrs(
        self,
        *,
        attr_keys: Sequence[str] | str | None = None,
        unpack: bool = False,
    ) -> pl.DataFrame:
        node_ids = self._get_local_ids()
        edges_df = (
            super()
            .filter(node_ids=node_ids)
            .edge_attrs(
                attr_keys=attr_keys,
                unpack=unpack,
            )
        )
        edges_df = self._map_df_to_external(
            edges_df, [DEFAULT_ATTR_KEYS.NODE_ID, DEFAULT_ATTR_KEYS.EDGE_SOURCE, DEFAULT_ATTR_KEYS.EDGE_TARGET]
        )
        return edges_df

    def update_node_attrs(
        self,
        *,
        attrs: dict[str, Any],
        node_ids: Sequence[int] | None = None,
    ) -> None:
        if node_ids is None:
            node_ids = self.node_ids()

        self._root.update_node_attrs(
            node_ids=node_ids,
            attrs=attrs,
        )
        # because attributes are passed by reference, we need don't need if both are rustworkx graphs
        if not self._is_root_rx_graph:
            if self.sync:
                super().update_node_attrs(
                    node_ids=self._map_to_local(node_ids),
                    attrs=attrs,
                )
            else:
                self._out_of_sync = True

    def update_edge_attrs(
        self,
        *,
        attrs: dict[str, Any],
        edge_ids: Sequence[int] | None = None,
    ) -> None:
        if edge_ids is None:
            edge_ids = self.edge_ids()

        self._root.update_edge_attrs(
            edge_ids=edge_ids,
            attrs=attrs,
        )
        # because attributes are passed by reference, we need don't need if both are rustworkx graphs
        if not self._is_root_rx_graph:
            if self.sync:
                super().update_edge_attrs(
                    edge_ids=[self._edge_map_from_root[eid] for eid in edge_ids],
                    attrs=attrs,
                )
            else:
                self._out_of_sync = True

    def assign_track_ids(
        self,
        output_key: str = DEFAULT_ATTR_KEYS.TRACK_ID,
        reset: bool = True,
        track_id_offset: int = 1,
    ) -> rx.PyDiGraph:
        """
        Compute and assign track ids to nodes.

        Parameters
        ----------
        output_key : str
            The key of the output track id attribute.
        reset : bool
            Whether to reset all track ids before assigning new ones.
        track_id_offset : int
            The starting track id, useful when assigning track ids to a subgraph.

        Returns
        -------
        rx.PyDiGraph
            A compressed graph (parent -> child) with track ids lineage relationships.
        """
        try:
            node_ids, track_ids, tracks_graph = _assign_track_ids(self.rx_graph, track_id_offset)
        except RuntimeError as e:
            raise RuntimeError(
                "Are you sure this graph is a valid lineage graph?\n"
                "This function expects a solved graph.\n"
                "Often used from `graph.subgraph(edge_attr_filter={'solution': True})`"
            ) from e

        node_ids = self._map_to_external(node_ids)

        if output_key not in self.node_attr_keys:
            self.add_node_attr_key(output_key, -1)
        elif reset:
            self.update_node_attrs(attrs={output_key: -1})

        self.update_node_attrs(
            node_ids=node_ids,
            attrs={output_key: track_ids},
        )

        return tracks_graph

    def in_degree(self, node_ids: list[int] | int | None = None) -> list[int] | int:
        """
        Get the in-degree of a list of nodes.
        """
        if node_ids is None:
            node_ids = self.node_ids()
        rx_graph = self.rx_graph
        if isinstance(node_ids, int):
            return rx_graph.in_degree(self._map_to_local(node_ids))
        return [rx_graph.in_degree(self._map_to_local(node_id)) for node_id in node_ids]

    def out_degree(self, node_ids: list[int] | int | None = None) -> list[int] | int:
        """
        Get the out-degree of a list of nodes.
        """
        if node_ids is None:
            node_ids = self.node_ids()
        rx_graph = self.rx_graph
        if isinstance(node_ids, int):
            return rx_graph.out_degree(self._map_to_local(node_ids))
        return [rx_graph.out_degree(self._map_to_local(node_id)) for node_id in node_ids]

    def _replace_parent_graph_with_root(self) -> None:
        """
        Replace the parent graph with it's own parent graph (the root graph)
        This is internally called so every view of a graph maps to a single root, skipping intermediate views.
        """
        parent = self._root

        if not isinstance(parent, GraphView):
            raise ValueError(
                f"Parent graph must be a GraphView to have its parent replaced with the root graph. Got {type(parent)}."
            )

        self._root = parent._root
        self._local_to_external = bidict.bidict(
            (k, parent._local_to_external[v]) for k, v in self._local_to_external.items()
        )

        self._edge_map_to_root = bidict.bidict(
            (k, parent._edge_map_to_root[v]) for k, v in self._edge_map_to_root.items()
        )

    def contract_nodes(
        self,
        permanent_node_ids: Sequence[int],
    ) -> "GraphView":
        """
        Contract the graph to only include the given `permanent_node_ids`.
        Predecessor and sucessors of removed nodes are connected during contraction.

        Example:

        ```mermaid
        graph TD
            A((t=0)) --> B((t=1))
            A --> C((t=1))
            B --> D((t=2))
            C --> E((t=2))
            C --> F((t=2))
        ```

        After contraction only keeping nodes in `t=0` and `t=2`:
        ```mermaid
        graph TD
            A((t=0)) --> B((t=2))
            A --> C((t=2))
            A --> D((t=2))
        ```

        Parameters
        ----------
        permanent_node_ids : Sequence[int]
            The node ids to keep in the contracted graph.

        Returns
        -------
        GraphView
            A view of the contracted graph.
        """
        subgraph = super().contract_nodes(
            permanent_node_ids=self._map_to_local(permanent_node_ids),
        )
        subgraph._replace_parent_graph_with_root()

        return subgraph

    @overload
    def detach(self, reset_ids: Literal[False]) -> IndexedRXGraph: ...

    @overload
    def detach(self, reset_ids: Literal[True]) -> RustWorkXGraph: ...

    def detach(self, reset_ids: bool = False) -> IndexedRXGraph | RustWorkXGraph:
        """
        Detach the graph view from the root graph, returning a new graph with the same nodes and edges
        without the view's mapping and indenpendent ids.

        Parameters
        ----------
        reset_ids : bool
            Whether to reset the ids of the graph.

        Returns
        -------
        IndexedRXGraph | RustWorkXGraph
            The detached graph.
        """
        if reset_ids:
            return RustWorkXGraph.from_other(self)
        else:
            return IndexedRXGraph.from_other(self)

    def _rx_subgraph_with_nodemap(
        self,
        node_ids: Sequence[int] | None = None,
    ) -> tuple[rx.PyDiGraph, rx.NodeMap]:
        """
        Get a subgraph of the graph.

        Parameters
        ----------
        node_ids : Sequence[int] | None
            The node ids to include in the subgraph.

        Returns
        -------
        tuple[rx.PyDiGraph, rx.NodeMap]
            The subgraph and the node map.
        """
        node_ids = self._map_to_local(node_ids)
        rx_graph, node_map = super()._rx_subgraph_with_nodemap(node_ids)
        node_map = {k: self._map_to_external(v) for k, v in node_map.items()}
        return rx_graph, node_map

    def has_edge(self, source_id: int, target_id: int) -> bool:
        """
        Check if the graph has an edge between two nodes.
        """

        try:
            source_id = self._map_to_local(source_id)
        except KeyError:
            LOG.warning(f"`source_id` {source_id} not found in index map.")
            return False

        try:
            target_id = self._map_to_local(target_id)
        except KeyError:
            LOG.warning(f"`target_id` {target_id} not found in index map.")
            return False

        return self.rx_graph.has_edge(source_id, target_id)

    def edge_id(self, source_id: int, target_id: int) -> int:
        """
        Return the edge id between two nodes.
        """
        return self._root.edge_id(source_id, target_id)

    def copy(self, **kwargs) -> "GraphView":
        """
        Not supported for `GraphView`.

        Use `detach` to create a new reference-less graph with the same nodes and edges.

        See Also
        --------
        [detach][tracksdata.graph.GraphView.detach]
            Create a new reference-less graph with the same nodes and edges.
        """
        raise ValueError(
            "`copy` is not supported for `GraphView`.\n"
            "Use `detach` to create a new reference-less graph with the same nodes and edges."
        )
