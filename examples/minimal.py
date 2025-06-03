import os
from pathlib import Path

import napari
import numpy as np
from tifffile import imread

from tracksdata.array._graph_array import GraphArrayView
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.edges._distance_edges import DistanceEdges
from tracksdata.edges._iou_edges import IoUEdgeWeights
from tracksdata.expr import AttrExpr
from tracksdata.functional.graph import graph_track_ids
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
    iou_operator = IoUEdgeWeights(output_key="iou", show_progress=True)

    solver = NearestNeighborsSolver(edge_weight=-AttrExpr("iou"), max_children=2)

    graph = RustWorkXGraphBackend()
    nodes_operator.add_nodes(graph, labels=labels)
    print(f"Number of nodes: {graph.num_nodes}")

    dist_operator.add_edges(graph)
    print(f"Number of edges: {graph.num_edges}")
    iou_operator.add_weights(graph)

    solver.solve(graph)

    solution_graph = graph.subgraph(
        edge_attr_filter={DEFAULT_ATTR_KEYS.SOLUTION: True},
    )
    node_ids, track_ids, tracks_graph = graph_track_ids(solution_graph._graph)
    graph.add_node_feature_key(DEFAULT_ATTR_KEYS.TRACK_ID, -1)
    graph.update_node_features(
        node_ids=node_ids,
        attributes={DEFAULT_ATTR_KEYS.TRACK_ID: track_ids},
    )

    dict_graph = {child: parent for parent, child in tracks_graph.edge_list()}

    tracks_df = graph.subgraph(
        node_attr_filter={DEFAULT_ATTR_KEYS.SOLUTION: True},
    ).node_features(
        feature_keys=[
            DEFAULT_ATTR_KEYS.TRACK_ID,
            DEFAULT_ATTR_KEYS.T,
            "y",
            "x",
        ],
    )

    array_view = GraphArrayView(
        graph,
        labels.shape,
        feature_key=DEFAULT_ATTR_KEYS.TRACK_ID,
    )

    print("opening napari ...")

    viewer = napari.Viewer()
    viewer.add_labels(array_view)
    viewer.add_tracks(tracks_df.to_numpy(), graph=dict_graph)
    napari.run()


if __name__ == "__main__":
    main()
