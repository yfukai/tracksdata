import numpy as np

from tracksdata.array._graph_array import GraphArrayView
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.functional._rx import graph_track_ids
from tracksdata.graph._base_graph import BaseGraph


def to_napari_format(
    graph: BaseGraph,
    shape: tuple[int, ...],
    solution_key: str = DEFAULT_ATTR_KEYS.SOLUTION,
    output_track_id_key: str = DEFAULT_ATTR_KEYS.TRACK_ID,
) -> tuple[
    GraphArrayView,
    np.ndarray,
    dict[int, int],
]:
    """
    Convert the subgraph of solution nodes to a napari-ready format.

    This includes:
    - a labels layer with the solution nodes
    - a tracks layer with the solution tracks
    - a graph with the parent-child relationships for the solution tracks

    Parameters
    ----------
    graph : BaseGraph
        The graph to convert.
    shape : tuple[int, ...]
        The shape of the labels layer.
    solution_key : str, optional
        The key of the solution attribute.
    output_track_id_key : str, optional
        The key of the output track id attribute.

    Returns
    -------
    """
    solution_graph = graph.subgraph(
        edge_attr_filter={solution_key: True},
    )
    solution_graph.sync = False

    node_ids, track_ids, tracks_graph = graph_track_ids(solution_graph.rx_graph)

    node_map = solution_graph._node_map_to_root
    node_ids = [node_map[node_id] for node_id in node_ids.tolist()]

    solution_graph.add_node_feature_key(output_track_id_key, -1)
    solution_graph.update_node_features(
        node_ids=node_ids,
        attributes={output_track_id_key: track_ids},
    )

    dict_graph = {child: parent for parent, child in tracks_graph.edge_list()}

    spatial_cols = ["z", "y", "x"][-len(shape) + 1 :]

    tracks_data = solution_graph.node_features(
        feature_keys=[output_track_id_key, DEFAULT_ATTR_KEYS.T, *spatial_cols],
    )

    array_view = GraphArrayView(
        solution_graph,
        shape,
        feature_key=output_track_id_key,
    )

    return array_view, tracks_data, dict_graph
