import numba as nb
import numpy as np
from numba import typed

from tracksdata.edges._base_edges import DEFAULT_EDGE_WEIGHT_KEY
from tracksdata.graph._base_graph import BaseGraphBackend
from tracksdata.solvers._base_solver import DEFAULT_SOLUTION_KEY, BaseSolver


@nb.njit
def _constrained_nearest_neighbors(
    sorted_source: np.ndarray,
    sorted_target: np.ndarray,
    solution: np.ndarray,
    max_children: int,
) -> None:
    children_counter = typed.Dict.empty(
        key_type=np.int64,
        value_type=np.int64,
    )
    seen = set()
    size = len(sorted_source)

    for i in range(size):
        source_id = sorted_source[i]
        target_id = sorted_target[i]

        if source_id in seen:
            continue

        target_count = children_counter.get(target_id, 0)
        if target_count >= max_children:
            continue

        seen.add(source_id)
        children_counter[target_id] = target_count + 1

        solution[i] = True


class NearestNeighborsSolver(BaseSolver):
    """
    Solver tracking problem with nearest neighbor ordering of edges.
    Each node can have only one parent and up to `max_children` child.
    """

    def __init__(
        self,
        max_children: int = 2,
        edge_weight_key: str = DEFAULT_EDGE_WEIGHT_KEY,
    ):
        self.max_children = max_children
        self.edge_weight_key = edge_weight_key

    def solve(
        self,
        graph: BaseGraphBackend,
        solution_key: str = DEFAULT_SOLUTION_KEY,
    ) -> None:
        """
        Solve the tracking problem with nearest neighbor ordering of edges.
        Each node can have only one parent and up to `max_children` child.

        Parameters
        ----------
        graph : BaseGraphBackend
            The graph to solve.
        solution_key : str
            The key to store the solution in the graph.
        """
        # get edges and sort them by weight
        edges_df = graph.edge_features(feature_keys=[self.edge_weight_key])
        sorted_indices = np.argsort(edges_df[self.edge_weight_key].to_numpy())

        sorted_source = edges_df["source"].to_numpy()[sorted_indices]
        sorted_target = edges_df["target"].to_numpy()[sorted_indices]
        sorted_solution = np.zeros(len(sorted_source), dtype=bool)

        _constrained_nearest_neighbors(
            sorted_source,
            sorted_target,
            sorted_solution,
            self.max_children,
        )
        del sorted_source, sorted_target

        inverted_indices = np.empty_like(sorted_indices)
        inverted_indices[sorted_indices] = np.arange(len(sorted_indices))
        solution = sorted_solution[inverted_indices]
        del sorted_solution, inverted_indices, sorted_indices

        graph.add_edge_feature_key(solution_key, False)
        graph.update_edge_features(
            edges_df["edge_id"].to_numpy()[solution],
            {solution_key: True},
        )
