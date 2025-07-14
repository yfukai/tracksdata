from typing import TYPE_CHECKING, Optional

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph

if TYPE_CHECKING:
    import napari


def _points_kwargs_defaults(points_kwargs: dict | None) -> dict:
    """
    Set the default values for the points kwargs.
    """
    if points_kwargs is None:
        points_kwargs = {}
    else:
        points_kwargs = points_kwargs.copy()

    color_cycle = ["yellow", "blue"]

    default_kwargs = {
        "face_color": "transparent",
        "border_width": 0.15,
        "border_color": "yellow",
        "border_color_cycle": color_cycle,
        "size": 20,
    }
    for key, value in default_kwargs.items():
        if key not in points_kwargs:
            points_kwargs[key] = value

    return points_kwargs


def _add_matching_text_and_color(points_kwargs: dict | None) -> dict:
    """
    Add the matching text and color to the points kwargs.
    """
    if points_kwargs is None:
        points_kwargs = {}
    else:
        points_kwargs = points_kwargs.copy()

    color_cycle = ["red", "green"]

    text_params = {
        "string": "Score:{match_score:.3f} Match:{matched_node_id}",
        "size": 20,
        "color": {"feature": "matched", "colormap": color_cycle},
        "anchor": "upper_left",
    }

    default_kwargs = {
        "border_color": "matched",
        "border_color_cycle": color_cycle,
        "text": text_params,
    }
    for key, value in default_kwargs.items():
        if key not in points_kwargs:
            points_kwargs[key] = value

    points_kwargs = _points_kwargs_defaults(points_kwargs)

    return points_kwargs


def visualize_matches(
    input_graph: BaseGraph,
    ref_graph: BaseGraph,
    matched_node_id_key: str = DEFAULT_ATTR_KEYS.MATCHED_NODE_ID,
    match_score_key: str = DEFAULT_ATTR_KEYS.MATCH_SCORE,
    viewer: Optional["napari.Viewer"] = None,
    points_kwargs: dict | None = None,
) -> None:
    """
    Visualize the matches between the graph and the gt_graph.

    Parameters
    ----------
    input_graph : BaseGraph
        The predicted graph to visualize.
    ref_graph : BaseGraph
        The reference (ground truth) graph to visualize.
    matched_node_id_key : str, optional
        The key of the attribute that contains the matched node IDs.
        If not provided, the default key will be used.
    match_score_key : str, optional
        The key of the attribute that contains the match scores.
        If not provided, the default key will be used.
    viewer : napari.Viewer, optional
        The napari viewer to use. If not provided, a new viewer will be created.
    points_kwargs : dict, optional
        Additional keyword arguments to the napari.Viewer.add_points method.
    """
    try:
        import napari
    except ImportError as e:
        raise ImportError(
            "napari must be installed to use this function.\nPlease install it with `pip install napari`.",
        ) from e

    if viewer is None:
        viewer = napari.Viewer()

    if "z" in input_graph.node_attr_keys:
        pos = ["t", "z", "y", "x"]
    else:
        pos = ["t", "y", "x"]

    # TODO: select a subset of attrs
    node_attrs = input_graph.node_attrs()
    ref_node_attrs = ref_graph.node_attrs()

    matched_points_kwargs = _add_matching_text_and_color(points_kwargs)
    points_kwargs = _points_kwargs_defaults(points_kwargs)

    layer = viewer.add_points(
        node_attrs[pos],
        name="predicted",
        properties={
            "matched_node_id": node_attrs[matched_node_id_key],
            "match_score": node_attrs[match_score_key],
            "matched": node_attrs[matched_node_id_key] >= 0,
        },
        **matched_points_kwargs,
    )
    layer.text.visible = False

    matched_ref_mask = ref_node_attrs[DEFAULT_ATTR_KEYS.NODE_ID].is_in(node_attrs[matched_node_id_key])
    # matched_ref_node_attrs = ref_node_attrs.filter(matched_ref_mask)
    not_matched_ref_node_attrs = ref_node_attrs.filter(~matched_ref_mask)

    viewer.add_points(
        not_matched_ref_node_attrs[pos],
        name="Missing objects",
        **points_kwargs,
    )

    # TODO: select a subset of attrs
    # edge_attrs = input_graph.edge_attrs()
