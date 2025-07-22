import numpy as np
import polars as pl
import rustworkx as rx
from numba import njit, typed, types

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.utils._logging import LOG

NO_PARENT = -1


@njit
def _fast_path_transverse(
    node: int,
    dag: dict[int, int],
) -> list[int]:
    """
    Transverse a path in the forest directed graph and add path (track) split into queue.

    Parameters
    ----------
    node : int
        Source path node.
    dag : dict[int, int]
        Directed graph (tree) of paths relationships.

    Returns
    -------
    list[int]
        Sequence of nodes in the path.
    """
    path = typed.List.empty_list(types.int64)

    while node != NO_PARENT:
        path.append(node)
        if node in dag:
            node = dag[node]
        else:
            node = NO_PARENT

    return path


@njit
def _fast_dag_transverse(
    roots: np.ndarray,
    dag: dict[int, int],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, dict[int, int], dict[int, int]]:
    """
    Transverse the tracks DAG creating a distinct id to each path.

    Parameters
    ----------
    roots : np.ndarray
        Forest roots.
    dag : dict[int, int]
        Directed acyclic graph.

    Returns
    -------
    tuple[list[np.ndarray], np.ndarray, np.ndarray, dict[int, int], dict[int, int]]
        - sequence of paths
        - their respective track_id
        - length
        - last_to_track_id: last node -> track_id
        - first_to_track_id: first node -> track_id
    """
    paths = []
    track_ids = []
    lengths = []
    last_to_track_id = {}
    first_to_track_id = {}

    track_id = 1

    for start in roots:
        path = _fast_path_transverse(start, dag)
        paths.append(path)
        track_ids.append(track_id)
        lengths.append(len(path))
        last_to_track_id[path[-1]] = track_id
        first_to_track_id[path[0]] = track_id
        track_id += 1

    paths = [np.asarray(p) for p in paths]
    track_ids = np.asarray(track_ids)
    lengths = np.asarray(lengths)

    return paths, track_ids, lengths, last_to_track_id, first_to_track_id


@njit
def _numba_dag(node_ids: np.ndarray, parent_ids: np.ndarray) -> dict[int, int]:
    """
    Creates a dict DAG of track lineages

    Important:
    Parent-less (orphan) nodes should not be provided.

    Parameters
    ----------
    node_ids : np.ndarray
        Nodes indices.
    parent_ids : np.ndarray
        Parent indices.

    Returns
    -------
    dict[int, int]
        DAG where parent maps to their child (parent -> child)
    """
    dag = {}

    for i in range(len(parent_ids)):
        dag[parent_ids[i]] = node_ids[i]

    return dag


def _rx_graph_to_dict_dag(graph: rx.PyDiGraph) -> tuple[dict[int, int], np.ndarray, pl.DataFrame]:
    """Creates the DAG of track lineages

    Parameters
    ----------
    graph : rx.PyDiGraph
        Directed acyclic graph of nodes.

    Returns
    -------
    tuple[dict[int, int], list[int], pl.DataFrame]
        - DAG where parent maps to their child (parent -> child)
        - Roots of the DAG
        - Edges (source, target) that requires a new track id (long edges and dividing edges)
          Often referred to as "long edges"
    """
    # target are the children
    # source are the parents
    node_indices = np.asarray(graph.node_indices(), dtype=np.int64)
    times = [n[DEFAULT_ATTR_KEYS.T] for n in graph.nodes()]
    nodes_df = pl.DataFrame(
        {"node_id": node_indices, DEFAULT_ATTR_KEYS.T: times},
    )
    edges_df = pl.from_numpy(
        np.asarray(graph.edge_list(), dtype=np.int64),
        schema=["source", "target"],
    )
    try:
        edges_df = (
            edges_df.join(
                nodes_df,
                left_on="target",
                right_on="node_id",
                how="left",
                validate="1:1",
            )
            .with_columns(pl.col("t").alias("t_target"))
            .join(
                nodes_df,
                left_on="source",
                right_on="node_id",
                how="left",
                validate="m:1",
                suffix="_source",
            )
        )

    except pl.exceptions.ComputeError as e:
        if "join keys did not fulfill 1:1" in str(e):
            raise RuntimeError("Invalid graph structure, found node with multiple parents") from e
        else:
            raise e

    dividing_mask = edges_df["source"].is_duplicated()
    long_edges_mask = (edges_df["t_target"] - edges_df["t_source"]).abs() > 1
    both_mask = dividing_mask | long_edges_mask

    long_edges_df = edges_df.filter(both_mask)

    edges_df = edges_df.filter(~both_mask)
    nodes_df = (
        nodes_df.with_columns(pl.col("node_id").alias("target"))
        .join(edges_df, on="target", how="left", validate="1:1")
        .select("target", "source")
    )

    # above we convert to numpy representation and then create numba dict
    # inside a njit function, otherwise it's very slow
    has_parent = nodes_df["source"].is_not_null()
    starts = nodes_df["target"].filter(~has_parent).cast(pl.Int64).to_numpy()

    nodes_arr = nodes_df.filter(has_parent).to_numpy(order="fortran").T
    forest = _numba_dag(nodes_arr[0], nodes_arr[1])

    return forest, starts, long_edges_df


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
    dag, roots, long_edges_df = _rx_graph_to_dict_dag(graph)

    paths, track_ids, lengths, last_to_track_id, first_to_track_id = _fast_dag_transverse(roots, dag)

    n_tracks = len(track_ids)

    tracks_graph = rx.PyDiGraph(node_count_hint=n_tracks, edge_count_hint=n_tracks)
    tracks_graph.add_nodes_from([None] * (n_tracks + 1))

    if len(long_edges_df) > 0:
        # assign long edges
        for src, tgt in zip(long_edges_df["source"].to_list(), long_edges_df["target"].to_list(), strict=True):
            child_track_id = first_to_track_id[tgt]
            parent_track_id = last_to_track_id[src]
            tracks_graph.add_edge(parent_track_id, child_track_id, None)

    paths = np.concatenate(paths)
    nodes_track_ids = np.repeat(track_ids, lengths)

    return paths, nodes_track_ids, tracks_graph
