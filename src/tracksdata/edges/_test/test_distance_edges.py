from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.edges._distance_edges import DistanceEdges
from tracksdata.graph._rustworkx_graph import RustWorkXGraph


def test_distance_edges_init_default_params() -> None:
    """Test initialization with default parameters."""
    operator = DistanceEdges(distance_threshold=10.0, n_neighbors=3)

    assert operator.output_key == DEFAULT_ATTR_KEYS.EDGE_WEIGHT
    assert operator.distance_threshold == 10.0
    assert operator.n_neighbors == 3
    assert operator.feature_keys is None
    assert operator.show_progress is True


def test_distance_edges_init_custom_params() -> None:
    """Test initialization with custom parameters."""
    operator = DistanceEdges(
        distance_threshold=5.0,
        n_neighbors=2,
        feature_keys=["x", "y"],
        show_progress=False,
        output_key="custom_distance",
    )
    assert operator.output_key == "custom_distance"
    assert operator.distance_threshold == 5.0
    assert operator.n_neighbors == 2
    assert operator.feature_keys == ["x", "y"]
    assert operator.show_progress is False


def test_distance_edges_add_edges_empty_graph() -> None:
    """Test adding edges to an empty graph."""
    graph = RustWorkXGraph()
    operator = DistanceEdges(distance_threshold=10.0, n_neighbors=3, show_progress=False)

    # Should not raise an error on empty graph
    operator.add_edges(graph)
    assert graph.num_edges == 0


def test_distance_edges_add_edges_single_timepoint_no_previous() -> None:
    """Test adding edges when there are no nodes in the previous timepoint."""
    graph = RustWorkXGraph()

    # Register feature keys
    graph.add_node_feature_key("x", 0.0)
    graph.add_node_feature_key("y", 0.0)

    # Add nodes only at t=1 (no t=0)
    graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 0.0, "y": 0.0})
    graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})

    operator = DistanceEdges(distance_threshold=10.0, n_neighbors=3, show_progress=False)

    # Should not add any edges since there are no nodes at t=0
    operator.add_edges(graph)
    assert graph.num_edges == 0


def test_distance_edges_add_edges_single_timepoint_no_current() -> None:
    """Test adding edges when there are no nodes in the current timepoint."""
    graph = RustWorkXGraph()

    # Register feature keys
    graph.add_node_feature_key("x", 0.0)
    graph.add_node_feature_key("y", 0.0)

    # Add nodes only at t=0 (no t=1)
    graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 1.0, "y": 1.0})

    operator = DistanceEdges(distance_threshold=10.0, n_neighbors=3, show_progress=False)

    # Should not add any edges since there are no nodes at t=1
    operator.add_edges(graph)
    assert graph.num_edges == 0


def test_distance_edges_add_edges_2d_coordinates() -> None:
    """Test adding edges with 2D coordinates."""
    graph = RustWorkXGraph()

    # Register feature keys
    graph.add_node_feature_key("x", 0.0)
    graph.add_node_feature_key("y", 0.0)

    # Add nodes at t=0
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 5.0, "y": 0.0})

    # Add nodes at t=1
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 6.0, "y": 1.0})

    operator = DistanceEdges(distance_threshold=3.0, n_neighbors=2, show_progress=False)

    operator.add_edges(graph)

    # Should have edges from t=0 to t=1 nodes within distance threshold
    assert graph.num_edges == 2  # other edges are outside distance threshold

    # Check that edge weights are added
    edges_df = graph.edge_features()
    assert DEFAULT_ATTR_KEYS.EDGE_WEIGHT in edges_df.columns


def test_distance_edges_add_edges_3d_coordinates() -> None:
    """Test adding edges with 3D coordinates."""
    graph = RustWorkXGraph()

    # Register feature keys
    graph.add_node_feature_key("x", 0.0)
    graph.add_node_feature_key("y", 0.0)
    graph.add_node_feature_key("z", 0.0)

    # Add nodes at t=0
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0, "z": 0.0})
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 5.0, "y": 0.0, "z": 0.0})

    # Add nodes at t=1
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0, "z": 1.0})
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 6.0, "y": 1.0, "z": 1.0})

    operator = DistanceEdges(distance_threshold=3.0, n_neighbors=2, show_progress=False)

    operator.add_edges(graph)

    # Should have edges from t=0 to t=1 nodes within distance threshold
    assert graph.num_edges == 2  # other edges are outside distance threshold


def test_distance_edges_add_edges_custom_feature_keys() -> None:
    """Test adding edges with custom feature keys."""
    graph = RustWorkXGraph()

    # Register feature keys
    graph.add_node_feature_key("pos_x", 0.0)
    graph.add_node_feature_key("pos_y", 0.0)

    # Add nodes at t=0
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "pos_x": 0.0, "pos_y": 0.0})
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "pos_x": 5.0, "pos_y": 0.0})

    # Add nodes at t=1
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "pos_x": 1.0, "pos_y": 1.0})
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "pos_x": 6.0, "pos_y": 1.0})

    operator = DistanceEdges(
        distance_threshold=3.0, n_neighbors=2, feature_keys=["pos_x", "pos_y"], show_progress=False
    )

    operator.add_edges(graph)

    # Should work with custom feature keys
    assert graph.num_edges == 2  # other edges are outside distance threshold


def test_distance_edges_add_edges_distance_threshold() -> None:
    """Test that distance threshold is respected."""
    graph = RustWorkXGraph()

    # Register feature keys
    graph.add_node_feature_key("x", 0.0)
    graph.add_node_feature_key("y", 0.0)

    # Add nodes at t=0
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})

    # Add nodes at t=1 - far away
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 100.0, "y": 100.0})

    operator = DistanceEdges(distance_threshold=1.0, n_neighbors=2, show_progress=False)  # Very small threshold

    operator.add_edges(graph)

    # Should have no edges due to distance threshold
    assert graph.num_edges == 0


def test_distance_edges_add_edges_multiple_timepoints() -> None:
    """Test adding edges for multiple timepoints."""
    graph = RustWorkXGraph()

    # Register feature keys
    graph.add_node_feature_key("x", 0.0)
    graph.add_node_feature_key("y", 0.0)

    # Add nodes at multiple timepoints
    for t in range(3):
        for i in range(2):
            graph.add_node({DEFAULT_ATTR_KEYS.T: t, "x": float(i), "y": float(t)})

    operator = DistanceEdges(distance_threshold=5.0, n_neighbors=2, show_progress=False)

    # Add edges for all timepoints
    operator.add_edges(graph)

    # Should have some edges
    assert graph.num_edges >= 0


def test_distance_edges_add_edges_custom_weight_key() -> None:
    """Test adding edges with custom weight key."""
    graph = RustWorkXGraph()

    # Register feature keys
    graph.add_node_feature_key("x", 0.0)
    graph.add_node_feature_key("y", 0.0)

    # Add nodes at t=0
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": 0.0, "y": 0.0})

    # Add nodes at t=1
    _ = graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 1.0, "y": 1.0})

    custom_weight_key = "custom_distance"
    operator = DistanceEdges(distance_threshold=5.0, n_neighbors=2, output_key=custom_weight_key, show_progress=False)

    operator.add_edges(graph)

    if graph.num_edges > 0:
        edges_df = graph.edge_features()
        assert len(edges_df) == 1
        assert custom_weight_key in edges_df.columns
        assert DEFAULT_ATTR_KEYS.EDGE_SOURCE in edges_df.columns
        assert DEFAULT_ATTR_KEYS.EDGE_TARGET in edges_df.columns
        assert DEFAULT_ATTR_KEYS.EDGE_ID in edges_df.columns


def test_distance_edges_n_neighbors_limit() -> None:
    """Test that n_neighbors limit is respected."""
    graph = RustWorkXGraph()

    # Register feature keys
    graph.add_node_feature_key("x", 0.0)
    graph.add_node_feature_key("y", 0.0)

    # Add many nodes at t=0
    for i in range(5):
        graph.add_node({DEFAULT_ATTR_KEYS.T: 0, "x": float(i), "y": 0.0})

    # Add one node at t=1
    graph.add_node({DEFAULT_ATTR_KEYS.T: 1, "x": 2.0, "y": 0.0})

    operator = DistanceEdges(
        distance_threshold=10.0,
        n_neighbors=2,
        show_progress=False,  # Limit to 2 neighbors
    )

    operator.add_edges(graph)

    # Should have at most 2 edges (limited by n_neighbors)
    assert graph.num_edges == 2
