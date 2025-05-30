from typing import Any, Literal

import numpy as np

from tracksdata.graph._base_graph import BaseGraphBackend
from tracksdata.nodes._base_nodes import BaseNodesOperator
from tracksdata.utils._processing import maybe_show_progress


class RandomNodesOperator(BaseNodesOperator):
    def __init__(
        self,
        n_time_points: int,
        n_nodes: tuple[int, int],
        n_dim: Literal[2, 3] = 3,
        random_state: int = 0,
        show_progress: bool = False,
    ):
        self.n_time_points = n_time_points
        self.n_nodes = n_nodes

        if n_dim == 2:
            self.spatial_cols = ["x", "y"]
        elif n_dim == 3:
            self.spatial_cols = ["x", "y", "z"]
        else:
            raise ValueError(f"Invalid number of dimensions: {n_dim}")

        self.rng = np.random.default_rng(random_state)
        self._show_progress = show_progress

    def add_nodes(
        self,
        graph: BaseGraphBackend,
        *,
        t: int | None = None,
        **kwargs: Any,
    ) -> None:
        if t is None:
            for t in maybe_show_progress(
                range(self.n_time_points),
                desc="Processing time points",
                show_progress=self._show_progress,
            ):
                self.add_nodes(graph, t=t, **kwargs)
            return

        # Register each spatial column individually
        for col in self.spatial_cols:
            if col not in graph.node_features_keys:
                graph.add_node_feature_key(col, None)

        n_nodes_at_t = self.rng.integers(
            self.n_nodes[0],
            self.n_nodes[1],
        )

        node_ids = []
        coords = self.rng.uniform(
            low=0,
            high=1,
            size=(n_nodes_at_t, len(self.spatial_cols)),
        )

        for c in coords:
            node_id = graph.add_node(
                {"t": t, **dict(zip(self.spatial_cols, c, strict=True))},
                **kwargs,
            )
            node_ids.append(node_id)

        return graph
