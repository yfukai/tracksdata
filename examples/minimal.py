import os
from pathlib import Path

import napari
import numpy as np
from tifffile import imread

from tracksdata.array._graph_array import GraphArrayView
from tracksdata.edges._distance_edges import DistanceEdgesOperator
from tracksdata.graph._rustworkx_graph import RustWorkXGraphBackend
from tracksdata.nodes._regionprops import RegionPropsOperator
from tracksdata.solvers._nearest_neighbors_solver import (
    DEFAULT_SOLUTION_KEY,
    NearestNeighborsSolver,
)


def main() -> None:
    # load from HeLa
    data_dir = Path(os.environ["CTC_DIR"]) / "training/Fluo-N2DL-HeLa/01_GT/TRA"
    assert data_dir.exists(), f"Data directory {data_dir} does not exist."

    labels = np.stack(
        [imread(p) for p in sorted(data_dir.glob("*.tif"))],
    )

    print("starting tracking ...")

    nodes_operator = RegionPropsOperator(show_progress=False)
    dist_operator = DistanceEdgesOperator(
        distance_threshold=15.0, n_neighbors=5, show_progress=False
    )
    solver = NearestNeighborsSolver()

    graph = RustWorkXGraphBackend()
    nodes_operator.add_nodes(graph, labels=labels)
    print(f"Number of nodes: {graph.num_nodes}")

    dist_operator.add_edges(graph)
    print(f"Number of edges: {graph.num_edges}")

    solver.solve(graph)

    array_view = GraphArrayView(graph, labels.shape, DEFAULT_SOLUTION_KEY)

    print("opening napari ...")

    napari.view_labels(array_view)
    napari.run()


if __name__ == "__main__":
    main()
