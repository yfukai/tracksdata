from collections.abc import Sequence

import numpy as np
from scipy.spatial import KDTree

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.edges._base_edges import BaseEdgesOperator
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.utils._logging import LOG


class DistanceEdges(BaseEdgesOperator):
    """
    Operator that adds edges to a graph based on the distance between nodes.

    Parameters
    ----------
    distance_threshold : float
        The distance threshold for adding edges.
    n_neighbors : int
        The maximum number of neighbors to consider for adding edges.
        This in respect from the current to the previous frame.
        That means, a node in frame t will have edges to the clostest
        n_neighbors nodes in frame t-1.
    feature_keys : Sequence[str] | None
        The feature keys to use for the distance calculation.
        When None, "z", "y", "x" are used.
    show_progress : bool
        Whether to print progress of the edges addition.
    """

    def __init__(
        self,
        distance_threshold: float,
        n_neighbors: int,
        output_key: str = DEFAULT_ATTR_KEYS.EDGE_WEIGHT,
        feature_keys: Sequence[str] | None = None,
        show_progress: bool = True,
    ):
        super().__init__(output_key=output_key, show_progress=show_progress)
        self.distance_threshold = distance_threshold
        self.n_neighbors = n_neighbors
        self.output_key = output_key
        self.feature_keys = feature_keys

    def _add_edges_per_time(
        self,
        graph: BaseGraph,
        *,
        t: int,
    ) -> None:
        """
        Add edges to a graph based on the distance between nodes.

        Parameters
        ----------
        graph : BaseGraph
            The graph to add edges to.
        t : int
            The time point to add edges for.
        """
        if self.output_key not in graph.edge_features_keys:
            # negative value to indicate that the edge is not valid
            graph.add_edge_feature_key(self.output_key, -99999.0)

        if self.feature_keys is None:
            if "z" in graph.node_features_keys:
                feature_keys = ["z", "y", "x"]
            else:
                feature_keys = ["y", "x"]
        else:
            feature_keys = self.feature_keys

        prev_node_ids = np.asarray(graph.filter_nodes_by_attribute({DEFAULT_ATTR_KEYS.T: t - 1}))
        cur_node_ids = np.asarray(graph.filter_nodes_by_attribute({DEFAULT_ATTR_KEYS.T: t}))

        if len(prev_node_ids) == 0:
            LOG.warning(
                "No nodes found for time point %d",
                t - 1,
            )
            return

        if len(cur_node_ids) == 0:
            LOG.warning(
                "No nodes found for time point %d",
                t,
            )
            return

        prev_features = graph.node_features(node_ids=prev_node_ids, feature_keys=feature_keys)
        cur_features = graph.node_features(node_ids=cur_node_ids, feature_keys=feature_keys)

        prev_kdtree = KDTree(prev_features.to_numpy())

        distances, prev_neigh_ids = prev_kdtree.query(
            cur_features.to_numpy(),
            k=self.n_neighbors,
            distance_upper_bound=self.distance_threshold,
        )
        is_valid = ~np.isinf(distances)

        # kdtree return from 0 to n-1
        # converting back to arbitrary indexing
        prev_neigh_ids[is_valid] = prev_node_ids[prev_neigh_ids[is_valid]]

        count = 0
        for cur_id, neigh_ids, neigh_dist, neigh_valid in zip(
            cur_node_ids, prev_neigh_ids, distances, is_valid, strict=True
        ):
            for neigh_id, dist in zip(neigh_ids[neigh_valid], neigh_dist[neigh_valid], strict=True):
                graph.add_edge(
                    neigh_id,
                    cur_id,
                    attributes={self.output_key: dist},
                    validate_keys=False,
                )
                count += 1

        if count == 0:
            LOG.warning(
                "No valid edges found for the pair of time point (%d, %d)",
                t,
                t - 1,
            )
