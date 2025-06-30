import numpy as np
import polars as pl
import rustworkx as rx
from numba import njit, typed, types

from tracksdata.utils._logging import LOG

NO_PARENT = -1


@njit
def _fast_path_transverse(
    node: int,
    track_id: int,
    queue: list[tuple[int, int]],
    dag: dict[int, list[int]],
) -> list[int]:
    """
    Transverse a path in the forest directed graph and add path (track) split into queue.

    Parameters
    ----------
    node : int
        Source path node.
    track_id : int
        Reference track id for path split.
    queue : list[tuple[int, int]]
        Source nodes and path (track) id reference queue.
    dag : dict[int, list[int]]
        Directed graph (tree) of paths relationships.

    Returns
    -------
    list[int]
        Sequence of nodes in the path.
    """
    path = typed.List.empty_list(types.int64)

    while True:
        path.append(node)

        children = dag.get(node)
        if children is None:
            # end of track
            break

        elif len(children) == 1:
            node = children[0]

        else:
            for child in children:
                queue.append((child, track_id))
            break

    return path


@njit
def _fast_dag_transverse(
    roots: list[int],
    dag: dict[int, list[int]],
) -> tuple[list[list[int]], list[int], list[int], list[int]]:
    """
    Transverse the tracks DAG creating a distinct id to each path.

    Parameters
    ----------
    roots : list[int]
        Forest roots.
    dag : dict[int, list[int]]
        Directed acyclic graph.

    Returns
    -------
    tuple[list[list[int]], list[int], list[int], list[int]]
        Sequence of paths, their respective track_id, parent_track_id and length.
    """
    track_id = 1
    paths = []
    track_ids = []  # equivalent to arange
    parent_track_ids = []
    lengths = []

    for root in roots:
        queue = [(root, NO_PARENT)]

        while queue:
            node, parent_track_id = queue.pop()
            path = _fast_path_transverse(node, track_id, queue, dag)
            paths.append(path)
            track_ids.append(track_id)
            parent_track_ids.append(parent_track_id)
            lengths.append(len(path))
            track_id += 1

    return paths, track_ids, parent_track_ids, lengths


@njit
def _numba_dag(node_ids: np.ndarray, parent_ids: np.ndarray) -> dict[int, list[int]]:
    """
    Creates a dict DAG of track lineages

    Parameters
    ----------
    node_ids : np.ndarray
        Nodes indices.
    parent_ids : np.ndarray
        Parent indices.

    Returns
    -------
    dict[int, list[int]]
        DAG where parent maps to their children (parent -> children)
    """
    dag = {}
    for parent in parent_ids:
        dag[parent] = typed.List.empty_list(types.int64)

    for i in range(len(parent_ids)):
        dag[parent_ids[i]].append(node_ids[i])

    return dag


def _rx_graph_to_dict_dag(graph: rx.PyDiGraph) -> dict[int, list[int]]:
    """Creates the DAG of track lineages

    Parameters
    ----------
    graph : rx.PyDiGraph
        Directed acyclic graph of nodes.

    Returns
    -------
    dict[int, list[int]]
        DAG where parent maps to their children (parent -> children)
    """
    # target are the children
    # source are the parents
    node_indices = np.asarray(graph.node_indices(), dtype=np.int64)
    graph_df = pl.DataFrame({"target": node_indices})
    edge_list = pl.from_numpy(
        np.asarray(graph.edge_list(), dtype=np.int64),
        schema=["source", "target"],
    )
    try:
        graph_df = (
            graph_df.join(edge_list, on="target", how="left", validate="1:1")
            .with_columns(pl.col("source").fill_null(NO_PARENT))
            .select(pl.col("target"), pl.col("source"))
            .to_numpy(order="fortran")
            .T
        )
    except pl.exceptions.ComputeError as e:
        if "join keys did not fulfill 1:1" in str(e):
            raise RuntimeError("Invalid graph structure, found node with multiple parents") from e
        else:
            raise e

    # above we convert to numpy representation and then create numba dict
    # inside a njit function, otherwise it's very slow
    forest = _numba_dag(graph_df[0], graph_df[1])

    return forest


def _assign_track_ids(
    graph: rx.PyDiGraph,
) -> tuple[np.ndarray, np.ndarray, rx.PyDiGraph]:
    """
    Assigns an unique `track_id` to each simple path in the graph and
    their respective parent -> child relationships.

    Parameters
    ----------
    graph : rx.PyDiGraph
        Directed acyclic graph of tracks.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, rx.PyDiGraph]
        - node_ids: Sequence of node ids.
        - track_ids: The respective track_id for each node.
        - tracks_graph: Graph indicating the parent -> child relationships.
    """
    if graph.num_nodes() == 0:
        raise ValueError("Graph is empty")

    LOG.info(f"Graph has {graph.num_nodes()} nodes and {graph.num_edges()} edges")

    # was it better (faster) when using a numpy array for the digraph as in ultrack?
    dag = _rx_graph_to_dict_dag(graph)
    roots = dag.pop(NO_PARENT)

    paths, track_ids, parent_track_ids, lengths = _fast_dag_transverse(roots, dag)

    n_tracks = len(track_ids)

    tracks_graph = rx.PyDiGraph(node_count_hint=n_tracks, edge_count_hint=n_tracks)
    tracks_graph.add_nodes_from([None] * (n_tracks + 1))
    tracks_graph.add_edges_from_no_data(
        [(p, c) for p, c in zip(parent_track_ids, track_ids, strict=True) if p != NO_PARENT]
    )

    paths = np.concatenate(paths)
    nodes_track_ids = np.repeat(track_ids, lengths)

    return paths, nodes_track_ids, tracks_graph
