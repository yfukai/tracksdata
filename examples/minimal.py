import os
from pathlib import Path

import napari
import numpy as np
from tifffile import imread

from tracksdata.edges._distance_edges import DistanceEdges
from tracksdata.edges._iou_edges import IoUEdgeWeights
from tracksdata.expr import AttrExpr
from tracksdata.functional._napari import to_napari_format
from tracksdata.graph._rustworkx_graph import RustWorkXGraphBackend
from tracksdata.nodes._regionprops import RegionPropsNodes
from tracksdata.solvers._nearest_neighbors_solver import NearestNeighborsSolver


def main() -> None:
    # load from HeLa
    data_dir = Path(os.environ["CTC_DIR"]) / "training/Fluo-N2DL-HeLa/01_GT/TRA"
    assert data_dir.exists(), f"Data directory {data_dir} does not exist."

    labels = np.stack(
        [imread(p) for p in sorted(data_dir.glob("*.tif"))],
    )

    print("starting tracking ...")

    nodes_operator = RegionPropsNodes(show_progress=False)
    dist_operator = DistanceEdges(distance_threshold=50.0, n_neighbors=5, show_progress=False)
    iou_operator = IoUEdgeWeights(output_key="iou", show_progress=False)

    solver = NearestNeighborsSolver(edge_weight=-AttrExpr("iou"), max_children=1)

    graph = RustWorkXGraphBackend()
    nodes_operator.add_nodes(graph, labels=labels)
    print(f"Number of nodes: {graph.num_nodes}")

    dist_operator.add_edges(graph)
    print(f"Number of edges: {graph.num_edges}")
    iou_operator.add_weights(graph)

    solver.solve(graph)

    print("Converting to napari format ...")
    labels, tracks_df, track_graph = to_napari_format(graph, labels.shape)

    print("Opening napari viewer ...")

    viewer = napari.Viewer()
    viewer.add_labels(labels)
    viewer.add_tracks(tracks_df, graph=track_graph)
    napari.run()


if __name__ == "__main__":
    main()
