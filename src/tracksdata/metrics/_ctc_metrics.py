import numpy as np

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import RustWorkXGraph


def compute_ctc_metrics_data(
    input_graph: RustWorkXGraph,
    reference_graph: RustWorkXGraph,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[list]]]:
    """
    Compute intermediate data required for CTC metrics.

    Reference:
    https://github.com/CellTrackingChallenge/py-ctcmetrics/blob/main/ctc_metrics/scripts/evaluate.py

    Parameters
    ----------
    input_graph : RustWorkXGraph
        Input graph.
    reference_graph : RustWorkXGraph
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
    # TODO:
    #  - continue here
    #  - compute compressed (n, 4) representation from BaseGraph
    #  - fast matching with IoU functions


def _validate_graph(graph: RustWorkXGraph, track_id_key: str) -> None:
    if track_id_key not in graph.node_features_keys:
        graph.assign_track_ids(track_id_key)


def evaluate_ctc_metrics(
    input_graph: RustWorkXGraph,
    reference_graph: RustWorkXGraph,
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
    input_graph : RustWorkXGraph
        Input graph.
    reference_graph : RustWorkXGraph
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

    if input_track_id_key not in input_graph.node_features_keys:
        input_graph.assign_track_ids(input_track_id_key)

    if reference_track_id_key not in reference_graph.node_features_keys:
        reference_graph.assign_track_ids(reference_track_id_key)

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
