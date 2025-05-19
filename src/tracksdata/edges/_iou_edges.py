import numpy as np

from tracksdata.graph._base_graph import BaseGraphBackend
from tracksdata.nodes._mask import DEFAULT_MASK_KEY
from tracksdata.utils._logging import LOG
from tracksdata.utils._processing import maybe_show_progress


class IoUEdgesOperator:
    # TODO: define API and inherit
    def __init__(self, show_progress: bool = True):
        self.show_progress = show_progress

    def add_weights(
        self,
        graph: BaseGraphBackend,
        *,
        t: int | None = None,
        mask_key: str = DEFAULT_MASK_KEY,
        weight_key: str,
    ) -> None:
        """
        Add weights to the edges of the graph based on the IoU
        of the masks of the nodes.

        Parameters
        ----------
        graph : BaseGraphBackend
            The graph to add weights to.
        t : int | None
            The time point to add weights to.
        mask_key : str
            The key to use for the masks of the nodes.
        weight_key : str
            The key to use for the computed weights.
        """

        if t is None:
            for t in maybe_show_progress(
                graph.time_points(),
                desc="Adding weights to edges",
                show_progress=self.show_progress,
            ):
                self.add_weights(graph, t=t, mask_key=mask_key, weight_key=weight_key)
            return

        source_ids = graph.filter_nodes_by_attribute(t=t)
        edges_df = graph.edge_features(node_ids=source_ids)

        if len(edges_df) == 0:
            LOG.warning(f"No edges found for time point {t} to sucessors")
            return

        source_df = graph.node_features(
            node_ids=edges_df["source"].to_numpy(), feature_keys=[DEFAULT_MASK_KEY]
        )
        target_df = graph.node_features(
            node_ids=edges_df["target"].to_numpy(), feature_keys=[DEFAULT_MASK_KEY]
        )

        weights = np.zeros(len(edges_df), dtype=np.float32)

        for i, (source_mask, target_mask) in enumerate(
            zip(source_df[DEFAULT_MASK_KEY], target_df[DEFAULT_MASK_KEY], strict=True)
        ):
            weights[i] = source_mask.iou(target_mask)

        if weight_key not in graph.edge_features_keys:
            graph.add_edge_feature_key(weight_key, -1.0)

        graph.update_edge_features(
            edges_df["edge_id"].to_numpy(), {weight_key: weights}
        )
