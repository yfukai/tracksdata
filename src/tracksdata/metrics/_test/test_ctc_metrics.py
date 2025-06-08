from pathlib import Path

from tracksdata.graph import RustWorkXGraph, SQLGraph
from tracksdata.metrics import evaluate_ctc_metrics


def test_ctc_metrics(ctc_data_dir: Path) -> None:
    # they are the same, I know
    # testing both backends at once
    input_graph = SQLGraph.from_ctc(
        ctc_data_dir,
        drivername="sqlite",
        database=":memory:",
        overwrite=True,
    )
    reference_graph = RustWorkXGraph.from_ctc(ctc_data_dir)

    metrics = evaluate_ctc_metrics(input_graph, reference_graph)

    print(metrics)
    # TODO:
    # add assertions
