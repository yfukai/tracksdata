import os
from pathlib import Path

import numpy as np
from tifffile import imread

from tracksdata.edges._distance_edges import DistanceEdgesOperator
from tracksdata.graph._rustworkx_graph import RustWorkXGraphBackend
from tracksdata.nodes._regionprops import RegionPropsOperator
from tracksdata.solvers._nearest_neighbors_solver import NearestNeighborsSolver


def main() -> None:
    # load from HeLa
    data_dir = Path(os.environ["CTC_DIR"]) / "training/Fluo-N2DL-HeLa/01_GT/TRA"
    assert data_dir.exists(), f"Data directory {data_dir} does not exist."

    labels = np.stack(
        [imread(p) for p in sorted(data_dir.glob("*.tif"))],
    )

    nodes_operator = RegionPropsOperator(show_progress=True)
    dist_operator = DistanceEdgesOperator(distance_threshold=15.0, n_neighbors=5)
    solver = NearestNeighborsSolver()

    graph = RustWorkXGraphBackend()
    nodes_operator.add_nodes(graph, labels=labels)
    print(f"Number of nodes: {graph.num_nodes}")

    dist_operator.add_edges(graph)
    print(f"Number of edges: {graph.num_edges}")

    solver.solve(graph)


if __name__ == "__main__":
    main()
