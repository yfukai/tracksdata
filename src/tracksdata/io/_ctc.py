from pathlib import Path
from typing import Any

import polars as pl
from dask.array.image import imread as dask_imread
from tifffile import imread as tiff_imread

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.nodes import RegionPropsNodes
from tracksdata.utils._logging import LOG


def _load_tracks_file(tracks_file: Path) -> dict[int, int]:
    """
    Load a CTC tracks file into a graph.

    Parameters
    ----------
    tracks_file : Path
        The path to the CTC tracks .txt file.
        The unnamed columns are:
        - track_id: the track ID
        - start: the start frame
        - end: the end frame
        - parent_track_id: the parent track ID

    Returns
    -------
    track_id_graph : dict[int, int]
        A dictionary mapping track IDs to their parent track IDs.
    """
    track_id_graph = {}

    df = pl.read_csv(
        tracks_file,
        separator=" ",
        has_header=False,
    ).rename(
        {
            "column_1": "track_id",
            "column_4": "parent_track_id",
        }
    )

    df = df.filter(pl.col("parent_track_id") > 0)

    for track_id, parent_track_id in zip(
        df["track_id"],
        df["parent_track_id"],
        strict=True,
    ):
        track_id_graph[track_id] = parent_track_id

    return track_id_graph


def load_ctc(
    data_dir: str | Path,
    graph: BaseGraph,
    region_props_kwargs: dict[str, Any] | None = None,
) -> None:
    """
    Load a CTC ground truth file into a graph.
    The resulting graph will have region properties features from a CTC ground truth file.

    Parameters
    ----------
    data_dir : str | Path
        The path to the CTC ground truth directory.
    graph : BaseGraph
        The graph to load the CTC ground truth into.
    region_props_kwargs : dict[str, Any]
        Keyword arguments to pass to RegionPropsNodes.

    Examples
    --------
    >>> from tracksdata.io import load_ctc
    >>> from tracksdata.graph import RustWorkXGraph
    >>> graph = RustWorkXGraph()
    >>> load_ctc("Fluo-N2DL-HeLa/01_GT/TRA", graph)

    See Also
    --------
    BaseGraph.from_ctc : Create a graph from a CTC data directory.
    RegionPropsNodes : Operator to create nodes from label images.

    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    if region_props_kwargs is None:
        region_props_kwargs = {}

    if "extra_properties" not in region_props_kwargs:
        region_props_kwargs["extra_properties"] = ["label"]

    elif "label" not in region_props_kwargs["extra_properties"]:
        region_props_kwargs["extra_properties"].append("label")

    tracks_file_found = False
    track_id_graph = {}

    for tracks_file in ["man_track.txt", "res_track.txt"]:
        tracks_file_path = data_dir / tracks_file
        if tracks_file_path.exists():
            tracks_file_found = True
            track_id_graph = _load_tracks_file(tracks_file_path)
            break

    if not tracks_file_found:
        LOG.warning(
            f"Tracks file {data_dir}/man_track.txt and {data_dir}/res_track.txt does not exist.\n"
            "Graph won't have edges."
        )

    labels = dask_imread(str(data_dir / "*.tif"), imread=tiff_imread)

    region_props_nodes = RegionPropsNodes(**region_props_kwargs)
    region_props_nodes.add_nodes(graph, labels=labels)

    nodes_df = graph.node_features(
        feature_keys=[
            DEFAULT_ATTR_KEYS.NODE_ID,
            DEFAULT_ATTR_KEYS.T,
            "label",
        ]
    )

    nodes_by_label = nodes_df.group_by("label")

    # reducing the nodes to the last node in each track
    # to find the divisions relationships between tracks
    if tracks_file_found:
        last_nodes = nodes_by_label.map_groups(lambda group: group.sort(DEFAULT_ATTR_KEYS.T).tail(1))
        track_id_to_last_node = dict(zip(last_nodes["label"], last_nodes[DEFAULT_ATTR_KEYS.NODE_ID], strict=True))
    else:
        track_id_to_last_node = {}  # for completeness, it won't be used if this is true

    for track_id, group in nodes_by_label:
        node_ids = group.sort(DEFAULT_ATTR_KEYS.T)[DEFAULT_ATTR_KEYS.NODE_ID].to_list()

        first_node = node_ids[0]
        if track_id in track_id_graph:
            parent_node = track_id_to_last_node[track_id_graph[track_id]]
            graph.add_edge(parent_node, first_node, {})

        for i in range(len(node_ids) - 1):
            graph.add_edge(node_ids[i], node_ids[i + 1], {})
