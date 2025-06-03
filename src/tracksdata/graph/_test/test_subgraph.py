from collections.abc import Callable

import polars as pl
import pytest

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraphBackend
from tracksdata.graph._rustworkx_graph import RustWorkXGraphBackend


@pytest.fixture(params=[RustWorkXGraphBackend])
def graph_backend(request) -> BaseGraphBackend:
    """Fixture that provides all implementations of BaseGraphBackend."""
    return request.param()


def parametrize_subgraph_tests(func: Callable[..., None]) -> Callable[..., None]:
    """Decorator to parametrize tests for both original graphs and subgraphs."""
    return pytest.mark.parametrize("use_subgraph", [False, True], ids=["original", "subgraph"])(func)


def create_test_graph(graph_backend: BaseGraphBackend, use_subgraph: bool = False) -> BaseGraphBackend:
    """
    Helper function to create a test graph with multiple nodes and edges.

    Parameters
    ----------
    graph_backend : BaseGraphBackend
        The graph backend to use for creating the test graph.
    use_subgraph : bool
        If True, returns a subgraph; if False, returns the original graph.

    Returns
    -------
    BaseGraphBackend
        Either the original graph or a subgraph with test data.
    """
    # Add feature keys
    graph_backend.add_node_feature_key("x", None)
    graph_backend.add_node_feature_key("y", None)
    graph_backend.add_node_feature_key("label", None)
    graph_backend.add_edge_feature_key("weight", 0.0)
    graph_backend.add_edge_feature_key("new_feature", 0.0)

    # Add nodes with various attributes
    node0 = graph_backend.add_node({"t": 0, "x": 0.0, "y": 0.0, "label": "0"})
    node1 = graph_backend.add_node({"t": 1, "x": 1.0, "y": 2.0, "label": "A"})
    node2 = graph_backend.add_node({"t": 2, "x": 2.0, "y": 3.0, "label": "B"})
    node3 = graph_backend.add_node({"t": 2, "x": 3.0, "y": 4.0, "label": "A"})
    node4 = graph_backend.add_node({"t": 3, "x": 4.0, "y": 5.0, "label": "C"})
    node5 = graph_backend.add_node({"t": 3, "x": 5.0, "y": 6.0, "label": "A"})

    # Add edges only between adjacent time points (t -> t+1)
    edge0 = graph_backend.add_edge(node0, node1, attributes={"weight": 0.5, "new_feature": 1.0})  # t=0 -> t=1
    edge1 = graph_backend.add_edge(node1, node2, attributes={"weight": 0.5, "new_feature": 1.0})  # t=1 -> t=2
    edge2 = graph_backend.add_edge(node1, node3, attributes={"weight": 0.7, "new_feature": 2.0})  # t=1 -> t=2
    edge3 = graph_backend.add_edge(node2, node4, attributes={"weight": 0.3, "new_feature": 3.0})  # t=2 -> t=3
    edge4 = graph_backend.add_edge(node3, node5, attributes={"weight": 0.9, "new_feature": 4.0})  # t=2 -> t=3

    if use_subgraph:
        # Create subgraph with nodes 1, 2, 4 in UNSORTED order to test robustness
        subgraph_nodes = [node4, node1, node2]  # Intentionally unsorted order
        subgraph = graph_backend.subgraph(node_ids=subgraph_nodes)

        # Store the subgraph nodes for testing (keep original order for assertions)
        subgraph._test_nodes = [node1, node2, node4]  # type: ignore
        subgraph._test_edges = [edge1, edge3]  # edges within the subgraph (both go from t=0 to t=1)  # type: ignore
        subgraph._is_subgraph = True  # type: ignore
        return subgraph
    else:
        # Store original nodes and edges for reference
        graph_backend._test_nodes = [node0, node1, node2, node3, node4, node5]  # type: ignore
        graph_backend._test_edges = [edge0, edge1, edge2, edge3, edge4]  # type: ignore
        graph_backend._is_subgraph = False  # type: ignore
        return graph_backend


@parametrize_subgraph_tests
def test_node_ids_with_data(graph_backend: BaseGraphBackend, use_subgraph: bool) -> None:
    """Test retrieving node IDs on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)
    node_ids = graph_with_data.node_ids()
    expected_nodes = graph_with_data._test_nodes  # type: ignore

    assert set(node_ids) == set(expected_nodes)
    assert len(node_ids) == len(expected_nodes)


@parametrize_subgraph_tests
def test_filter_nodes_by_attribute_with_data(graph_backend: BaseGraphBackend, use_subgraph: bool) -> None:
    """Test filtering nodes by attributes on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)

    # Filter by time
    nodes = graph_with_data.filter_nodes_by_attribute({"t": 0})
    # Should find nodes with t=0 that are in this graph
    expected_t0_nodes = [
        n
        for n in graph_with_data._test_nodes  # type: ignore
        if graph_with_data.node_features(node_ids=[n])["t"].to_list()[0] == 0
    ]
    assert set(nodes) == set(expected_t0_nodes)

    # Filter by label
    nodes = graph_with_data.filter_nodes_by_attribute({"label": "A"})
    # Should find nodes with label="A" that are in this graph
    expected_label_a_nodes = [
        n
        for n in graph_with_data._test_nodes  # type: ignore
        if graph_with_data.node_features(node_ids=[n])["label"].to_list()[0] == "A"
    ]
    assert set(nodes) == set(expected_label_a_nodes)


@parametrize_subgraph_tests
def test_time_points_with_data(graph_backend: BaseGraphBackend, use_subgraph: bool) -> None:
    """Test retrieving time points on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)
    time_points = graph_with_data.time_points()

    # Get unique time points from the nodes in this graph
    expected_times = set()
    for node in graph_with_data._test_nodes:  # type: ignore
        node_data = graph_with_data.node_features(node_ids=[node])
        expected_times.add(node_data["t"].to_list()[0])

    assert set(time_points) == expected_times


@parametrize_subgraph_tests
def test_node_features_with_data(graph_backend: BaseGraphBackend, use_subgraph: bool) -> None:
    """Test retrieving node features on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)
    nodes = graph_with_data._test_nodes[:2]  # Test with first two nodes  # type: ignore

    df = graph_with_data.node_features(node_ids=nodes, feature_keys=["x", "y"])
    assert isinstance(df, pl.DataFrame)
    assert len(df) == len(nodes)
    assert "x" in df.columns
    assert "y" in df.columns

    # Test with single feature key as string
    df_single = graph_with_data.node_features(node_ids=nodes, feature_keys="x")
    assert "x" in df_single.columns
    assert len(df_single) == len(nodes)


@parametrize_subgraph_tests
def test_edge_features_with_data(graph_backend: BaseGraphBackend, use_subgraph: bool) -> None:
    """Test retrieving edge features on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)
    df = graph_with_data.edge_features(feature_keys=["weight"])
    assert isinstance(df, pl.DataFrame)

    assert len(df) == len(graph_with_data._test_edges)  # type: ignore

    assert "weight" in df.columns
    assert DEFAULT_ATTR_KEYS.EDGE_SOURCE in df.columns
    assert DEFAULT_ATTR_KEYS.EDGE_TARGET in df.columns


@parametrize_subgraph_tests
def test_add_node_feature_key_with_data(graph_backend: BaseGraphBackend, use_subgraph: bool) -> None:
    """Test adding new node feature keys on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)

    # Add a new feature key with default value
    graph_with_data.add_node_feature_key("new_node_feature", 42)

    # Check that all nodes have this feature with the default value
    nodes = graph_with_data._test_nodes  # type: ignore
    df = graph_with_data.node_features(node_ids=nodes, feature_keys=["new_node_feature"])

    assert all(val == 42 for val in df["new_node_feature"].to_list())


@parametrize_subgraph_tests
def test_add_edge_feature_key_with_data(graph_backend: BaseGraphBackend, use_subgraph: bool) -> None:
    """Test adding new edge feature keys on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)

    # Add a new edge feature key with default value
    graph_with_data.add_edge_feature_key("new_edge_feature", 99)

    # Check that all edges have this feature with the default value
    df = graph_with_data.edge_features(feature_keys=["new_edge_feature"])

    if len(df) > 0:
        assert all(val == 99 for val in df["new_edge_feature"].to_list())


@parametrize_subgraph_tests
def test_update_node_features_with_data(graph_backend: BaseGraphBackend, use_subgraph: bool) -> None:
    """Test updating node features on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)
    nodes = graph_with_data._test_nodes[:2]  # Use first two nodes  # type: ignore

    # Update with single value
    graph_with_data.update_node_features(node_ids=[nodes[0]], attributes={"x": 99.0})

    df = graph_with_data.node_features(node_ids=[nodes[0]], feature_keys=["x"])
    assert df["x"].to_list()[0] == 99.0

    # Update with list of values
    graph_with_data.update_node_features(node_ids=nodes, attributes={"x": [100.0, 101.0]})

    df = graph_with_data.node_features(node_ids=nodes, feature_keys=["x"])
    assert df["x"].to_list() == [100.0, 101.0]

    # Test error with wrong length
    with pytest.raises(ValueError):
        graph_with_data.update_node_features(node_ids=nodes, attributes={"x": [1.0]})


@parametrize_subgraph_tests
def test_update_edge_features_with_data(graph_backend: BaseGraphBackend, use_subgraph: bool) -> None:
    """Test updating edge features on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)
    df = graph_with_data.edge_features()

    if len(df) > 0:
        # Get the first edge ID
        edge_id = df[DEFAULT_ATTR_KEYS.EDGE_ID].to_list()[0]

        # Update edge feature
        graph_with_data.update_edge_features(edge_ids=[edge_id], attributes={"weight": 99.0})

        # Verify update
        df_updated = graph_with_data.edge_features(feature_keys=["weight"])
        edge_weights = df_updated["weight"].to_list()
        assert 99.0 in edge_weights

        # Test error with wrong length
        with pytest.raises(ValueError):
            graph_with_data.update_edge_features(edge_ids=[edge_id], attributes={"weight": [1.0, 2.0]})


@parametrize_subgraph_tests
def test_num_edges_with_data(graph_backend: BaseGraphBackend, use_subgraph: bool) -> None:
    """Test counting edges on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)
    num_edges = graph_with_data.num_edges
    assert num_edges == len(graph_with_data._test_edges)  # type: ignore


@parametrize_subgraph_tests
def test_num_nodes_with_data(graph_backend: BaseGraphBackend, use_subgraph: bool) -> None:
    """Test counting nodes on both original graphs and subgraphs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph)
    num_nodes = graph_with_data.num_nodes
    assert num_nodes == len(graph_with_data._test_nodes)  # type: ignore


def test_subgraph_creation(graph_backend: BaseGraphBackend) -> None:
    """Test creating subgraphs and their properties."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph=False)

    # Create a subgraph with subset of nodes in UNSORTED order
    original_nodes = graph_with_data._test_nodes  # type: ignore
    subgraph_nodes_unsorted = [original_nodes[1], original_nodes[0]]  # Reverse order

    subgraph = graph_with_data.subgraph(node_ids=subgraph_nodes_unsorted)

    # Test that subgraph has correct number of nodes
    assert subgraph.num_nodes == len(subgraph_nodes_unsorted)

    # Test that subgraph node IDs match expected (regardless of input order)
    subgraph_node_ids = subgraph.node_ids()
    assert set(subgraph_node_ids) == set(subgraph_nodes_unsorted)

    # Test that subgraph has correct node features
    for node in subgraph_nodes_unsorted:
        original_features = graph_with_data.node_features(node_ids=[node])
        subgraph_features = subgraph.node_features(node_ids=[node])

        # Compare all columns
        for col in original_features.columns:
            assert original_features[col].to_list() == subgraph_features[col].to_list()


def test_subgraph_edge_preservation(graph_backend: BaseGraphBackend) -> None:
    """Test that subgraphs preserve correct edges between included nodes."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph=False)

    # Get all nodes and create subgraph with first 2 nodes in REVERSE order
    original_nodes = graph_with_data._test_nodes  # type: ignore
    subgraph_nodes = [original_nodes[1], original_nodes[0]]  # Reverse order

    # Get edges from original graph involving only these nodes
    original_edges = graph_with_data.edge_features()
    subgraph_node_set = set(subgraph_nodes)

    expected_edges_count = 0
    for i in range(len(original_edges)):
        source = original_edges[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list()[i]
        target = original_edges[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()[i]
        if source in subgraph_node_set and target in subgraph_node_set:
            expected_edges_count += 1

    # Create subgraph
    subgraph = graph_with_data.subgraph(node_ids=subgraph_nodes)
    subgraph_edges = subgraph.edge_features()

    # Check that subgraph has the correct number of edges
    assert len(subgraph_edges) == expected_edges_count

    # Check that all edges in subgraph involve only subgraph nodes
    for i in range(len(subgraph_edges)):
        source = subgraph_edges[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list()[i]
        target = subgraph_edges[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()[i]
        assert source in subgraph_node_set
        assert target in subgraph_node_set


def test_subgraph_feature_consistency(graph_backend: BaseGraphBackend) -> None:
    """Test that subgraph node and edge features are consistent with original graph."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph=False)
    original_nodes = graph_with_data._test_nodes  # type: ignore
    # Use nodes in unsorted order to test robustness
    subgraph_nodes = [original_nodes[1], original_nodes[0]]

    # Create subgraph
    subgraph = graph_with_data.subgraph(node_ids=subgraph_nodes)

    # Test node feature consistency
    for node in subgraph_nodes:
        original_node_features = graph_with_data.node_features(node_ids=[node])
        subgraph_node_features = subgraph.node_features(node_ids=[node])

        # All columns should match
        assert set(original_node_features.columns) == set(subgraph_node_features.columns)

        # All values should match
        for col in original_node_features.columns:
            assert original_node_features[col].to_list() == subgraph_node_features[col].to_list()

    # Test edge feature consistency for edges that exist in both
    original_edges = graph_with_data.edge_features()
    subgraph_edges = subgraph.edge_features()

    # Check that subgraph edges have same feature keys as original
    assert set(original_edges.columns) == set(subgraph_edges.columns)

    # For each edge in subgraph, verify it has same weights as in original
    for i in range(len(subgraph_edges)):
        sub_source = subgraph_edges[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list()[i]
        sub_target = subgraph_edges[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()[i]
        sub_weight = subgraph_edges["weight"].to_list()[i]

        # Find corresponding edge in original graph
        found_match = False
        for j in range(len(original_edges)):
            orig_source = original_edges[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list()[j]
            orig_target = original_edges[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()[j]
            orig_weight = original_edges["weight"].to_list()[j]

            if orig_source == sub_source and orig_target == sub_target:
                assert orig_weight == sub_weight
                found_match = True
                break

        assert found_match, f"Edge ({sub_source}, {sub_target}) not found in original graph"


def test_subgraph_with_unsorted_node_ids(graph_backend: BaseGraphBackend) -> None:
    """Test that subgraph creation works correctly with unsorted node IDs."""
    graph_with_data = create_test_graph(graph_backend, use_subgraph=False)
    original_nodes = graph_with_data._test_nodes  # type: ignore

    # Test with various unsorted orders
    test_cases = [
        [original_nodes[2], original_nodes[0], original_nodes[1]],  # Mixed order
        [original_nodes[3], original_nodes[1], original_nodes[5], original_nodes[0]],  # Descending + mixed
        [original_nodes[4], original_nodes[2]],  # Just two nodes in reverse order
    ]

    for unsorted_nodes in test_cases:
        # Create subgraph with unsorted node IDs
        subgraph = graph_with_data.subgraph(node_ids=unsorted_nodes)

        # Verify the subgraph contains exactly the expected nodes
        subgraph_node_ids = set(subgraph.node_ids())
        expected_node_ids = set(unsorted_nodes)
        assert subgraph_node_ids == expected_node_ids, f"Failed for node order: {unsorted_nodes}"

        # Verify node features are preserved correctly
        for node in unsorted_nodes:
            original_features = graph_with_data.node_features(node_ids=[node])
            subgraph_features = subgraph.node_features(node_ids=[node])

            for col in original_features.columns:
                msg = f"Node {node} feature {col} mismatch in subgraph with order {unsorted_nodes}"
                assert original_features[col].to_list() == subgraph_features[col].to_list(), msg
