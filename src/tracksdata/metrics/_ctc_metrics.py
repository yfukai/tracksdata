import numpy as np

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.functional._rx import graph_track_ids
from tracksdata.graph import RustWorkXGraph
from tracksdata.graph._base_graph import BaseGraph


def compute_ctc_metrics_data(
    input_graph: BaseGraph,
    reference_graph: BaseGraph,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[list]]]:
    """
    Compute intermediate data required for CTC metrics.

    Reference:
    https://github.com/CellTrackingChallenge/py-ctcmetrics/blob/main/ctc_metrics/scripts/evaluate.py

    Parameters
    ----------
    input_graph : BaseGraph
        Input graph.
    reference_graph : BaseGraph
        Reference graph.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict[str, list], dict[str, list], list[str]]
        - input_tracks:
            (n, 4) array with reprensenting the (track_id, start_frame, end_frame, parent_track_id) of each track
        - reference_tracks:
            (n, 4) array with reprensenting the (track_id, start_frame, end_frame, parent_track_id) of each track
        - matching_data: Frame-wise matching data, defined as a dict with the following keys:
            - labels_ref: A list of lists containing the labels of the reference masks.
            - labels_comp: A list of lists containing the labels of the computed masks.
            - mapped_ref: A list of lists containing the mapped labels of the reference masks.
            - mapped_comp: A list of lists containing the mapped labels of the computed masks.
            - ious: A list of lists containing the intersection over union values
                    between mapped reference and computed masks.
    """


def _validate_graph(graph: BaseGraph, track_id_key: str) -> BaseGraph:
    """
    Validate the graph.
    """
    if track_id_key in graph.node_features_keys:
        return graph

    if not isinstance(graph, RustWorkXGraph):
        # could be replaced by having `.rx_graph` on the base class
        # rx_graph should take attributes so we avoid loading all the features
        graph = graph.subgraph(node_ids=graph.node_ids(), edge_feature_keys=[])

    # TODO:
    #  - this is pretty bad, we should not be using the rx graph here
    node_ids, track_ids, tracks_graph = graph_track_ids(graph.rx_graph)

    if hasattr(graph, "_node_map_to_root"):
        # FIXME: maybe graph_track_ids should take a `BaseGraph` as input
        node_map = graph._node_map_to_root
        node_ids = [node_map[node_id] for node_id in node_ids.tolist()]

    graph.add_node_feature_key(track_id_key, -1)
    graph.update_node_features(
        node_ids=node_ids,
        attributes={track_id_key: track_ids},
    )

    return graph


def evaluate_ctc_metrics(
    input_graph: BaseGraph,
    reference_graph: BaseGraph,
    input_track_id_key: str = DEFAULT_ATTR_KEYS.TRACK_ID,
    reference_track_id_key: str = DEFAULT_ATTR_KEYS.TRACK_ID,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """
    Evaluate CTC metrics using `py-ctcmetrics` developed by Timo Kaiser.

    If you use this function, please cite the respective papers of each metric, as described in:
    https://github.com/CellTrackingChallenge/py-ctcmetrics?tab=readme-ov-file#acknowledgement-and-citations

    Note that the `SEG` metric is not supported.

    Parameters
    ----------
    input_graph : BaseGraph
        Input graph.
    reference_graph : BaseGraph
        Reference graph.
    input_track_id_key : str, optional
        Key to obtain the track id from the input graph.
        If key does not exist, it will be created.
    reference_track_id_key : str, optional
        Key to obtain the track id from the reference graph.
        If key does not exist, it will be created.
    metrics : list[str] | None, optional
        List of metrics to evaluate. If None, all metrics are evaluated.
        Available metrics:
        "CHOTA", "BC", "CT", "CCA", "TF", "TRA", "DET", "MOTA", "HOTA",
        "IDF1", "MTML", "FAF", "LNK", "OP_CTB", "OP_CSB", "BIO", "OP_CLB"

    Returns
    -------
    dict[str, float]
        Dictionary with the results of the evaluated metrics.
    """
    try:
        from ctc_metrics.metrics import ALL_METRICS
        from ctc_metrics.scripts.evaluate import calculate_metrics
    except ImportError as e:
        raise ImportError(
            "`py-ctcmetrics` is required to evaluate CTC metrics.\nPlease install it with `pip install py-ctcmetrics`."
        ) from e

    input_graph = _validate_graph(input_graph, input_track_id_key)
    reference_graph = _validate_graph(reference_graph, reference_track_id_key)

    input_tracks, reference_tracks, matching_data = compute_ctc_metrics_data(input_graph, reference_graph)

    if metrics is None:
        metrics: list[str] = ALL_METRICS.copy()
        metrics.remove("SEG")

    if "SEG" in metrics:
        raise ValueError("SEG metric is not supported. Please remove it from the `metrics` list.")

    results = calculate_metrics(
        input_tracks,
        reference_tracks,
        matching_data,
        segm={},
        metrics=metrics,
        is_valid=True,
    )

    return results
