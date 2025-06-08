import shutil
from pathlib import Path

import numpy as np
from ctc_metrics.scripts.evaluate import load_data

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph import RustWorkXGraph
from tracksdata.metrics import evaluate_ctc_metrics
from tracksdata.metrics._ctc_metrics import compute_ctc_metrics_data


def test_ctc_metrics(ctc_data_dir: Path) -> None:
    # hack required to load two ground-truths with ctc_metrics
    shutil.copy(ctc_data_dir / "man_track.txt", ctc_data_dir / "res_track.txt")

    # loading reference intermediate data
    ref_input_tracks, ref_reference_tracks, ref_matching_data, *_ = load_data(
        res=str(ctc_data_dir),
        gt=str(ctc_data_dir.parent),
        trajectory_data=True,
        segmentation_data=False,
        threads=1,
    )

    # they are the same, I know
    # testing both backends at once
    input_graph = RustWorkXGraph.from_ctc(ctc_data_dir)
    reference_graph = RustWorkXGraph.from_ctc(ctc_data_dir)

    input_tracks, reference_tracks, matching_data = compute_ctc_metrics_data(
        input_graph,
        reference_graph,
        input_track_id_key=DEFAULT_ATTR_KEYS.TRACK_ID,
        reference_track_id_key=DEFAULT_ATTR_KEYS.TRACK_ID,
    )

    for key, values in matching_data.items():
        expected_values = ref_matching_data[key]
        for t, (v, e) in enumerate(zip(values, expected_values, strict=True)):
            # I'm not really sure why only the ious shape are different
            np.testing.assert_array_equal(v, e, err_msg=f"{key=} t={t}")

    # TODO:
    # - test compressed tracks representation

    return

    metrics = evaluate_ctc_metrics(input_graph, reference_graph)

    print(metrics)
    # TODO:
    # add assertions
