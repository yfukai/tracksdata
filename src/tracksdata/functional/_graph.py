import numpy as np
import rustworkx as rx
from numba import njit, typed, types


@njit
def _fast_path_transverse(
    node: int,
    track_id: int,
    queue: list[tuple[int, int]],
    forest: dict[int, list[int]],
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
    forest : dict[int, list[int]]
        Directed graph (tree) of paths relationships.

    Returns
    -------
    list[int]
        Sequence of nodes in the path.
    """
    path = typed.List.empty_list(types.int64)

    while True:
        path.append(node)

        children = forest.get(node)
        if children is None:
            # end of track
            break

        elif len(children) == 1:
            node = children[0]

        elif len(children) == 2:
            queue.append((children[1], track_id))
            queue.append((children[0], track_id))
            break

        else:
            raise RuntimeError("Something is wrong. Found node with more than two children when parsing tracks.")

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
        queue = [(root, None)]

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
def _create_numba_dag(node_ids: np.ndarray, parent_ids: np.ndarray) -> dict[int, list[int]]:
    """Creates the DAG of track lineages

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
    forest = {}
    for parent in parent_ids:
        forest[parent] = typed.List.empty_list(types.int64)

    for i in range(len(parent_ids)):
        forest[parent_ids[i]].append(node_ids[i])

    return forest


def graph_track_ids(
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

    node_ids = np.asarray(graph.node_indices(), dtype=int)
    no_parent = -1

    parent_ids = []
    for node in node_ids:
        parent = graph.predecessor_indices(node)
        if len(parent) == 0:
            parent_ids.append(no_parent)
        elif len(parent) == 1:
            parent_ids.append(parent[0])
        else:
            raise RuntimeError(f"Found node ({node}) with ({parent_ids}) parents. Expected at most one parent.")

    parent_ids = np.asarray(parent_ids, dtype=int)

    dag = _create_numba_dag(node_ids, parent_ids)
    roots = dag.pop(no_parent)

    paths, track_ids, parent_track_ids, lengths = _fast_dag_transverse(roots, dag)

    tracks_graph = rx.PyDiGraph(node_count_hint=len(track_ids), edge_count_hint=len(track_ids))
    tracks_graph.add_nodes_from([None] * (len(track_ids) + 1))
    tracks_graph.add_edges_from_no_data([(p, c) for p, c in zip(parent_track_ids, track_ids, strict=False)])

    paths = np.concatenate(paths)
    nodes_track_ids = np.repeat(track_ids, lengths)

    return paths, nodes_track_ids, tracks_graph
