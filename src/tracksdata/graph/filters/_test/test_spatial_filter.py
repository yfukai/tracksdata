import pytest

from tracksdata.graph import RustWorkXGraph
from tracksdata.graph.filters._spatial_filter import (
    BoundingBoxSpatialFilter,
    SpatialFilter,
)


@pytest.fixture
def sample_graph() -> RustWorkXGraph:
    """Create a sample graph with nodes for testing."""
    graph = RustWorkXGraph()
    graph.add_node_attr_key("z", 0)
    graph.add_node_attr_key("y", 0)
    graph.add_node_attr_key("x", 0)

    # Add some nodes with spatial coordinates
    nodes = [
        {"t": 0, "z": 0, "y": 10, "x": 20},
        {"t": 0, "z": 0, "y": 30, "x": 40},
        {"t": 1, "z": 1, "y": 50, "x": 60},
        {"t": 2, "z": 2, "y": 90, "x": 100},
    ]

    for node_attrs in nodes:
        graph.add_node(node_attrs)

    return graph


@pytest.fixture
def sample_bb_graph() -> RustWorkXGraph:
    """Create a sample graph with nodes for bounding box testing."""
    graph = RustWorkXGraph()
    graph.add_node_attr_key("z_min", 0)
    graph.add_node_attr_key("y_min", 0)
    graph.add_node_attr_key("x_min", 0)
    graph.add_node_attr_key("z_max", 0)
    graph.add_node_attr_key("y_max", 0)
    graph.add_node_attr_key("x_max", 0)

    # Add some nodes with bounding box coordinates
    nodes = [
        {"t": 0, "z_min": 0, "y_min": 10, "x_min": 20, "z_max": 1, "y_max": 15, "x_max": 25},
        {"t": 0, "z_min": 0, "y_min": 30, "x_min": 40, "z_max": 1, "y_max": 35, "x_max": 45},
        {"t": 1, "z_min": 1, "y_min": 50, "x_min": 60, "z_max": 2, "y_max": 55, "x_max": 65},
        {"t": 2, "z_min": 2, "y_min": 90, "x_min": 100, "z_max": 3, "y_max": 95, "x_max": 105},
    ]

    for node_attrs in nodes:
        graph.add_node(node_attrs)

    return graph


def test_spatial_filter_initialization(sample_graph: RustWorkXGraph) -> None:
    """Test SpatialFilter initialization with default and custom attributes."""
    # Test default attributes
    spatial_filter = SpatialFilter(sample_graph)
    assert spatial_filter._df_filter._attrs_keys == ["t", "z", "y", "x"]
    assert spatial_filter._df_filter._sg_graph is not None

    # Test custom attributes
    custom_attrs = ["t", "y", "x"]
    spatial_filter = SpatialFilter(sample_graph, attrs_keys=custom_attrs)
    assert spatial_filter._df_filter._attrs_keys == custom_attrs


def test_spatial_filter_querying(sample_graph: RustWorkXGraph) -> None:
    """Test spatial querying with different bounds and dimensions."""
    spatial_filter = SpatialFilter(sample_graph)

    # Test valid bounds that include nodes
    result = spatial_filter[0:2, 0:2, 0:100, 0:100]
    node_attrs = result.node_attrs()
    assert not node_attrs.is_empty()

    # Test narrow bounds that exclude most nodes
    result = spatial_filter[0:1, 0:1, 0:15, 0:25]
    node_attrs = result.node_attrs()
    assert len(node_attrs) <= 1

    # Test bounds that exclude all nodes
    result = spatial_filter[10:20, 10:20, 200:300, 200:300]
    node_attrs = result.node_attrs()
    assert node_attrs.is_empty()


def test_spatial_filter_dimensions() -> None:
    """Test SpatialFilter with different coordinate dimensions."""
    graph = RustWorkXGraph()
    graph.add_node_attr_key("z", 0)
    graph.add_node_attr_key("y", 0)
    graph.add_node_attr_key("x", 0)
    graph.add_node({"t": 0, "z": 0, "y": 10, "x": 20})

    # Test 2D coordinates
    spatial_filter_2d = SpatialFilter(graph, attrs_keys=["y", "x"])
    assert spatial_filter_2d._df_filter._attrs_keys == ["y", "x"]
    result = spatial_filter_2d[0:50, 0:50]
    assert not result.node_attrs().is_empty()

    # Test 3D coordinates
    spatial_filter_3d = SpatialFilter(graph, attrs_keys=["z", "y", "x"])
    assert spatial_filter_3d._df_filter._attrs_keys == ["z", "y", "x"]
    result = spatial_filter_3d[0:2, 0:100, 0:100]
    assert not result.node_attrs().is_empty()


def test_spatial_filter_error_handling(sample_graph: RustWorkXGraph) -> None:
    """Test error handling for invalid slice counts."""
    spatial_filter = SpatialFilter(sample_graph)

    # Test wrong number of slices
    with pytest.raises(ValueError, match="Expected 4 keys, got 3"):
        spatial_filter[0:2, 0:1, 0:50]


def test_spatial_filter_with_edges() -> None:
    """Test SpatialFilter preserves edges in subgraphs."""
    graph = RustWorkXGraph()
    graph.add_node_attr_key("y", 0)
    graph.add_node_attr_key("x", 0)
    graph.add_edge_attr_key("weight", 0.0)

    # Add nodes and edge
    node1_id = graph.add_node({"t": 0, "y": 10, "x": 20})
    node2_id = graph.add_node({"t": 1, "y": 30, "x": 40})
    graph.add_edge(node1_id, node2_id, {"weight": 1.0})

    spatial_filter = SpatialFilter(graph, attrs_keys=["y", "x"])
    result = spatial_filter[0:50, 0:50]

    # Should preserve both nodes and the edge
    assert len(result.node_attrs()) == 2
    assert len(result.edge_attrs()) == 1


def test_bb_spatial_filter_overlaps() -> None:
    """Test BoundingBoxSpatialFilter overlaps with existing nodes."""
    graph = RustWorkXGraph()
    graph.add_node_attr_key("min_y", 0)
    graph.add_node_attr_key("min_x", 0)
    graph.add_node_attr_key("max_y", 0)
    graph.add_node_attr_key("max_x", 0)

    # Add nodes with bounding boxes
    bboxes = [
        [[0, 10], [20, 30]],  # Node 1
        [[5, 15], [25, 35]],  # Node 2
        [[10, 20], [30, 40]],  # Node 3
        [[15, 25], [35, 45]],  # Node 4
    ]
    node_ids = []
    for bbox in bboxes:
        node_id = graph.add_node(
            {
                "t": 0,
                "min_y": bbox[0][0],
                "min_x": bbox[1][0],
                "max_y": bbox[0][1],
                "max_x": bbox[1][1],
            }
        )
        node_ids.append(node_id)

    spatial_filter = BoundingBoxSpatialFilter(
        graph, min_attrs_keys=["t", "min_y", "min_x"], max_attrs_keys=["t", "max_y", "max_x"]
    )
    result = spatial_filter[0:0, 15:20, 0:40]
    assert set(result.node_ids()) == {node_ids[i] for i in [1, 2, 3]}
    result = spatial_filter[0:0, 15:20, 36:40]
    assert set(result.node_ids()) == {node_ids[i] for i in [2, 3]}

    # Should return both nodes
    assert len(result.node_attrs()) == 2


def test_bb_spatial_filter_with_edges() -> None:
    """Test SpatialFilter preserves edges in subgraphs."""
    graph = RustWorkXGraph()
    graph.add_node_attr_key("min_y", 0)
    graph.add_node_attr_key("min_x", 0)
    graph.add_node_attr_key("max_y", 0)
    graph.add_node_attr_key("max_x", 0)
    graph.add_edge_attr_key("weight", 0.0)

    # Add nodes and edge
    node1_id = graph.add_node({"t": 0, "min_y": 10, "min_x": 20, "max_y": 15, "max_x": 25})
    node2_id = graph.add_node({"t": 1, "min_y": 30, "min_x": 40, "max_y": 35, "max_x": 45})
    graph.add_edge(node1_id, node2_id, {"weight": 1.0})

    spatial_filter = BoundingBoxSpatialFilter(
        graph, min_attrs_keys=["t", "min_y", "min_x"], max_attrs_keys=["t", "max_y", "max_x"]
    )
    result = spatial_filter[0:1, 0:50, 0:50]

    # Should preserve both nodes and the edge
    assert len(result.node_attrs()) == 2
    assert len(result.edge_attrs()) == 1


def test_bb_spatial_filter_initialization(sample_bb_graph: RustWorkXGraph) -> None:
    """Test BoundingBoxSpatialFilter initialization with default and custom attributes."""
    # Test default attributes - should filter to existing keys
    spatial_filter = BoundingBoxSpatialFilter(sample_bb_graph)
    # The filter should use the available min/max keys from the graph
    assert spatial_filter._node_rtree is not None

    # Test custom attributes
    custom_min_attrs = ["t", "y_min", "x_min"]
    custom_max_attrs = ["t", "y_max", "x_max"]
    spatial_filter = BoundingBoxSpatialFilter(
        sample_bb_graph, min_attrs_keys=custom_min_attrs, max_attrs_keys=custom_max_attrs
    )
    assert spatial_filter._node_rtree is not None


def test_bb_spatial_filter_querying(sample_bb_graph: RustWorkXGraph) -> None:
    """Test bounding box spatial querying with different bounds and dimensions."""
    spatial_filter = BoundingBoxSpatialFilter(
        sample_bb_graph,
        min_attrs_keys=["t", "z_min", "y_min", "x_min"],
        max_attrs_keys=["t", "z_max", "y_max", "x_max"],
    )

    # Test valid bounds that include nodes
    result = spatial_filter[0:3, 0:3, 0:100, 0:110]
    node_attrs = result.node_attrs()
    assert not node_attrs.is_empty()

    # Test narrow bounds that should include specific nodes
    result = spatial_filter[0:1, 0:2, 10:20, 20:30]
    node_attrs = result.node_attrs()
    assert len(node_attrs) >= 1

    # Test bounds that exclude all nodes
    result = spatial_filter[10:20, 10:20, 200:300, 200:300]
    node_attrs = result.node_attrs()
    assert node_attrs.is_empty()


def test_bb_spatial_filter_dimensions() -> None:
    """Test BoundingBoxSpatialFilter with different coordinate dimensions."""
    graph = RustWorkXGraph()
    graph.add_node_attr_key("z_min", 0)
    graph.add_node_attr_key("y_min", 0)
    graph.add_node_attr_key("x_min", 0)
    graph.add_node_attr_key("z_max", 0)
    graph.add_node_attr_key("y_max", 0)
    graph.add_node_attr_key("x_max", 0)
    graph.add_node({"t": 0, "z_min": 0, "y_min": 10, "x_min": 20, "z_max": 1, "y_max": 15, "x_max": 25})

    # Test 2D coordinates
    spatial_filter_2d = BoundingBoxSpatialFilter(
        graph, min_attrs_keys=["y_min", "x_min"], max_attrs_keys=["y_max", "x_max"]
    )
    result = spatial_filter_2d[0:50, 0:50]
    assert not result.node_attrs().is_empty()

    # Test 3D coordinates
    spatial_filter_3d = BoundingBoxSpatialFilter(
        graph, min_attrs_keys=["z_min", "y_min", "x_min"], max_attrs_keys=["z_max", "y_max", "x_max"]
    )
    result = spatial_filter_3d[0:2, 0:100, 0:100]
    assert not result.node_attrs().is_empty()


def test_bb_spatial_filter_error_handling() -> None:
    """Test error handling for mismatched min/max attribute lengths."""
    graph = RustWorkXGraph()
    graph.add_node_attr_key("y_min", 0)
    graph.add_node_attr_key("x_min", 0)
    graph.add_node_attr_key("y_max", 0)
    graph.add_node_attr_key("x_max", 0)
    graph.add_node({"t": 0, "y_min": 10, "x_min": 20, "y_max": 15, "x_max": 25})

    # Test mismatched min/max attributes length
    with pytest.raises(AssertionError, match="min_attrs_keys and max_attrs_keys must have the same length"):
        BoundingBoxSpatialFilter(graph, min_attrs_keys=["y_min", "x_min"], max_attrs_keys=["y_max"])  # Missing x_max
