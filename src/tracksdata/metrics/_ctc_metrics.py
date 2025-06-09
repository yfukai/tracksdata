import numpy as np

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import RustWorkXGraph
from tracksdata.io._ctc import compressed_tracks_table
from tracksdata.utils._logging import LOG


def _matching_data(
    input_graph: RustWorkXGraph,
    reference_graph: RustWorkXGraph,
    reference_track_id_key: str,
    input_track_id_key: str,
) -> dict[str, list[list]]:
    result = {}

    groups_by_time = {
        "ref": {},
        "comp": {},
    }

    # computing unique labels for each graph
    for name, graph, track_id_key in [
        ("ref", reference_graph, reference_track_id_key),
        ("comp", input_graph, input_track_id_key),
    ]:
        nodes_df = graph.node_features(feature_keys=[DEFAULT_ATTR_KEYS.T, track_id_key, DEFAULT_ATTR_KEYS.MASK])
        labels = {}

        for (t,), group in nodes_df.group_by(DEFAULT_ATTR_KEYS.T):
            labels[t] = group[track_id_key].to_list()
            # storing masks of each frame
            groups_by_time[name][t] = group

        result[f"labels_{name}"] = [labels[t] for t in sorted(labels.keys())]

    mapped_ref = []
    mapped_comp = []
    ious = []

    for t in set.union(set(groups_by_time["ref"].keys()), set(groups_by_time["comp"].keys())):
        _mapped_ref = []
        mapped_ref.append(_mapped_ref)

        _mapped_comp = []
        mapped_comp.append(_mapped_comp)

        _ious = []
        ious.append(_ious)

        try:
            ref_group = groups_by_time["ref"][t]
            comp_group = groups_by_time["comp"][t]
        except KeyError:
            continue

        for ref_id, ref_mask in zip(ref_group[reference_track_id_key], ref_group[DEFAULT_ATTR_KEYS.MASK], strict=True):
            for comp_id, comp_mask in zip(
                comp_group[input_track_id_key], comp_group[DEFAULT_ATTR_KEYS.MASK], strict=True
            ):
                # intersection over reference is used to select the matches
                inter = ref_mask.intersection(comp_mask)
                ctc_score = inter / ref_mask.size()
                if ctc_score > 0.5:
                    _mapped_ref.append(ref_id)
                    _mapped_comp.append(comp_id)

                    # NOTE: there was something weird with IoU, the length when compared with `ctc_metrics` is always +1
                    iou = inter / (ref_mask.size() + comp_mask.size() - inter)
                    _ious.append(iou.item())

    result["mapped_ref"] = mapped_ref
    result["mapped_comp"] = mapped_comp
    result["ious"] = ious

    return result


def compute_ctc_metrics_data(
    input_graph: RustWorkXGraph,
    reference_graph: RustWorkXGraph,
    input_track_id_key: str,
    reference_track_id_key: str,
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
    input_track_id_key : str
        Key to obtain the track id from the input graph.
    reference_track_id_key : str
        Key to obtain the track id from the reference graph.

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
    input_tracks = compressed_tracks_table(input_graph)
    reference_tracks = compressed_tracks_table(reference_graph)

    matching_data = _matching_data(input_graph, reference_graph, input_track_id_key, reference_track_id_key)

    return input_tracks, reference_tracks, matching_data


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

    input_tracks, reference_tracks, matching_data = compute_ctc_metrics_data(
        input_graph, reference_graph, input_track_id_key, reference_track_id_key
    )

    if metrics is None:
        metrics: list[str] = ALL_METRICS.copy()

    if "SEG" in metrics:
        LOG.warning("IMPORTANT! 'SEG' metric results are based on TRA masks, not the SEG masks.")

    results = calculate_metrics(
        comp_tracks=input_tracks,
        ref_tracks=reference_tracks,
        traj=matching_data,
        segm=matching_data,
        metrics=metrics,
        is_valid=True,
    )

    results = {k: v.item() if hasattr(v, "item") else v for k, v in results.items()}

    return results
