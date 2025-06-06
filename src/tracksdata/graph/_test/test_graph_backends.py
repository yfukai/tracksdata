import polars as pl
import pytest
import sqlalchemy as sa

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.graph._rustworkx_graph import RustWorkXGraph
from tracksdata.graph._sql_graph import SQLGraph


@pytest.fixture(params=[RustWorkXGraph, SQLGraph])
def graph_backend(request) -> BaseGraph:
    """Fixture that provides all implementations of BaseGraph."""
    graph_class: BaseGraph = request.param

    if graph_class == SQLGraph:
        db_name = ":memory:"
        engine = sa.create_engine(sa.engine.URL.create("sqlite", database=db_name))
        # clear database
        with engine.connect() as conn:
            conn.execute(sa.text("DROP TABLE IF EXISTS nodes"))
            conn.execute(sa.text("DROP TABLE IF EXISTS edges"))
            conn.commit()
        return graph_class(
            drivername="sqlite",
            database=db_name,
        )
    else:
        return graph_class()


def test_already_existing_keys(graph_backend: BaseGraph) -> None:
    """Test that adding already existing keys raises an error."""
    graph_backend.add_node_feature_key("x", None)

    with pytest.raises(ValueError):
        graph_backend.add_node_feature_key("x", None)

    graph_backend.add_edge_feature_key("w", None)

    with pytest.raises(ValueError):
        graph_backend.add_edge_feature_key("w", None)

    with pytest.raises(ValueError):
        # missing x
        graph_backend.add_node(attributes={"t": 0})


def testing_empty_graph(graph_backend: BaseGraph) -> None:
    """Test that the graph is empty."""
    assert graph_backend.num_nodes == 0
    assert graph_backend.num_edges == 0

    assert graph_backend.node_features().is_empty()
    assert graph_backend.edge_features().is_empty()


def test_node_validation(graph_backend: BaseGraph) -> None:
    """Test node validation."""
    # 't' key must exist by default
    graph_backend.add_node({"t": 1})

    with pytest.raises(ValueError):
        graph_backend.add_node({"t": 0, "x": 1.0})


def test_edge_validation(graph_backend: BaseGraph) -> None:
    """Test edge validation."""
    with pytest.raises(ValueError):
        graph_backend.add_edge(0, 1, {"weight": 0.5})


def test_add_node(graph_backend: BaseGraph) -> None:
    """Test adding nodes with various attributes."""

    for key in ["x", "y"]:
        graph_backend.add_node_feature_key(key, None)

    node_id = graph_backend.add_node({"t": 0, "x": 1.0, "y": 2.0})
    assert isinstance(node_id, int)

    # Check node features
    df = graph_backend.node_features(node_ids=[node_id])
    assert df["t"].to_list() == [0]
    assert df["x"].to_list() == [1.0]
    assert df["y"].to_list() == [2.0]


def test_add_edge(graph_backend: BaseGraph) -> None:
    """Test adding edges with attributes."""
    # Add node feature key
    graph_backend.add_node_feature_key("x", None)

    # Add two nodes first
    node1 = graph_backend.add_node({"t": 0, "x": 1.0})
    node2 = graph_backend.add_node({"t": 1, "x": 2.0})
    node3 = graph_backend.add_node({"t": 2, "x": 1.0})

    # Add edge feature key
    graph_backend.add_edge_feature_key("weight", 0.0)

    # Add edge
    edge_id = graph_backend.add_edge(node1, node2, attributes={"weight": 0.5})
    assert isinstance(edge_id, int)

    # Check edge features
    df = graph_backend.edge_features()
    assert df[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list() == [node1]
    assert df[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list() == [node2]
    assert df["weight"].to_list() == [0.5]

    # testing adding new add attribute
    graph_backend.add_edge_feature_key("new_feature", 0.0)
    edge_id = graph_backend.add_edge(node2, node3, attributes={"new_feature": 1.0, "weight": 0.1})
    assert isinstance(edge_id, int)

    # testing default value was assigned correctly
    df = graph_backend.edge_features()
    assert df["new_feature"].to_list() == [0.0, 1.0]
    assert df["weight"].to_list() == [0.5, 0.1]


def test_node_ids(graph_backend: BaseGraph) -> None:
    """Test retrieving node IDs."""
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})

    assert set(graph_backend.node_ids()) == {node1, node2}


def test_filter_nodes_by_attribute(graph_backend: BaseGraph) -> None:
    """Test filtering nodes by attributes."""
    graph_backend.add_node_feature_key("label", None)

    node1 = graph_backend.add_node({"t": 0, "label": "A"})
    node2 = graph_backend.add_node({"t": 0, "label": "B"})
    node3 = graph_backend.add_node({"t": 1, "label": "A"})

    # Filter by time
    nodes = graph_backend.filter_nodes_by_attribute({"t": 0})
    assert set(nodes) == {node1, node2}

    # Filter by label
    nodes = graph_backend.filter_nodes_by_attribute({"label": "A"})
    assert set(nodes) == {node1, node3}

    # Filter by t and label
    nodes = graph_backend.filter_nodes_by_attribute({"t": 1, "label": "A"})
    assert set(nodes) == {node3}


def test_time_points(graph_backend: BaseGraph) -> None:
    """Test retrieving time points."""
    graph_backend.add_node({"t": 0})
    graph_backend.add_node({"t": 2})
    graph_backend.add_node({"t": 1})

    assert set(graph_backend.time_points()) == {0, 1, 2}


def test_node_features(graph_backend: BaseGraph) -> None:
    """Test retrieving node features."""
    graph_backend.add_node_feature_key("x", None)

    node1 = graph_backend.add_node({"t": 0, "x": 1.0})
    node2 = graph_backend.add_node({"t": 1, "x": 2.0})

    df = graph_backend.node_features(node_ids=[node1, node2], feature_keys=["x"])
    assert isinstance(df, pl.DataFrame)
    assert df["x"].to_list() == [1.0, 2.0]


def test_edge_features(graph_backend: BaseGraph) -> None:
    """Test retrieving edge features."""
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})

    graph_backend.add_edge_feature_key("weight", 0.0)
    graph_backend.add_edge(node1, node2, attributes={"weight": 0.5})

    df = graph_backend.edge_features(feature_keys=["weight"])
    assert isinstance(df, pl.DataFrame)
    assert df["weight"].to_list() == [0.5]


def test_edge_features_subgraph_edge_ids(graph_backend: BaseGraph) -> None:
    """Test that edge_features preserves original edge IDs when using node_ids parameter."""
    # Add edge feature key
    graph_backend.add_edge_feature_key("weight", 0.0)

    # Create nodes
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})
    node3 = graph_backend.add_node({"t": 2})
    node4 = graph_backend.add_node({"t": 3})

    print(f"Created nodes: {node1=}, {node2=}, {node3=}, {node4=}")

    # Create edges
    edge1 = graph_backend.add_edge(node1, node2, attributes={"weight": 0.1})
    edge2 = graph_backend.add_edge(node2, node3, attributes={"weight": 0.2})
    edge3 = graph_backend.add_edge(node3, node4, attributes={"weight": 0.3})
    edge4 = graph_backend.add_edge(node1, node3, attributes={"weight": 0.4})

    print(f"Created edges: {edge1=}, {edge2=}, {edge3=}, {edge4=}")

    # Get all edge features (full graph)
    df_full = graph_backend.edge_features()
    print(f"Full graph edges: {df_full}")

    full_edge_ids = df_full[DEFAULT_ATTR_KEYS.EDGE_ID].to_list()
    full_sources = df_full[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list()
    full_targets = df_full[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()

    print("Full graph edge details:")
    for eid, src, tgt in zip(full_edge_ids, full_sources, full_targets, strict=False):
        print(f"  Edge {eid}: {src} -> {tgt}")

    # Get edge features for a subset of nodes [node1, node2, node3]
    # This should include:
    # - edge1: node1 -> node2
    # - edge2: node2 -> node3
    # - edge4: node1 -> node3
    # But NOT edge3: node3 -> node4 (since node4 is not in the subset)
    df_subset = graph_backend.edge_features(node_ids=[node1, node2, node3])
    print(f"Subset graph edges: {df_subset}")

    subset_edge_ids = df_subset[DEFAULT_ATTR_KEYS.EDGE_ID].to_list()
    subset_sources = df_subset[DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list()
    subset_targets = df_subset[DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list()

    print("Subset graph edge details:")
    for eid, src, tgt in zip(subset_edge_ids, subset_sources, subset_targets, strict=False):
        print(f"  Edge {eid}: {src} -> {tgt}")

    # The edge IDs should preserve the original edge IDs
    # and only include edges between the specified nodes
    expected_subset_edge_ids = {edge1, edge2, edge4}
    actual_subset_edge_ids = set(subset_edge_ids)

    # This will demonstrate the bug
    msg = f"Expected {expected_subset_edge_ids}, got {actual_subset_edge_ids}"
    assert actual_subset_edge_ids == expected_subset_edge_ids, msg


def test_add_node_feature_key(graph_backend: BaseGraph) -> None:
    """Test adding new node feature keys."""
    node = graph_backend.add_node({"t": 0})
    graph_backend.add_node_feature_key("new_feature", 42)

    df = graph_backend.node_features(node_ids=[node], feature_keys=["new_feature"])
    assert df["new_feature"].to_list() == [42]


def test_add_edge_feature_key(graph_backend: BaseGraph) -> None:
    """Test adding new edge feature keys."""
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})

    graph_backend.add_edge_feature_key("new_feature", 42)
    graph_backend.add_edge(node1, node2, attributes={"new_feature": 42})

    df = graph_backend.edge_features(feature_keys=["new_feature"])
    assert df["new_feature"].to_list() == [42]


def test_update_node_features(graph_backend: BaseGraph) -> None:
    """Test updating node features."""
    graph_backend.add_node_feature_key("x", None)

    node_1 = graph_backend.add_node({"t": 0, "x": 1.0})
    node_2 = graph_backend.add_node({"t": 0, "x": 2.0})

    graph_backend.update_node_features(node_ids=[node_1], attributes={"x": 3.0})

    df = graph_backend.node_features(node_ids=[node_1, node_2], feature_keys="x")
    assert df["x"].to_list() == [3.0, 2.0]

    # inverted access on purpose
    graph_backend.update_node_features(node_ids=[node_2, node_1], attributes={"x": [5.0, 6.0]})

    df = graph_backend.node_features(node_ids=[node_1, node_2], feature_keys="x")
    assert df["x"].to_list() == [6.0, 5.0]

    # wrong length
    with pytest.raises(ValueError):
        graph_backend.update_node_features(node_ids=[node_1, node_2], attributes={"x": [1.0]})


def test_update_edge_features(graph_backend: BaseGraph) -> None:
    """Test updating edge features."""
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})

    graph_backend.add_edge_feature_key("weight", 0.0)
    edge_id = graph_backend.add_edge(node1, node2, attributes={"weight": 0.5})

    graph_backend.update_edge_features(edge_ids=[edge_id], attributes={"weight": 1.0})
    df = graph_backend.edge_features(node_ids=[node1, node2], feature_keys=["weight"])
    assert df["weight"].to_list() == [1.0]

    # wrong length
    with pytest.raises(ValueError):
        graph_backend.update_edge_features(edge_ids=[edge_id], attributes={"weight": [1.0, 2.0]})


def test_num_edges(graph_backend: BaseGraph) -> None:
    """Test counting edges."""
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})

    graph_backend.add_edge_feature_key("weight", 0.0)
    graph_backend.add_edge(node1, node2, attributes={"weight": 0.5})

    assert graph_backend.num_edges == 1


def test_num_nodes(graph_backend: BaseGraph) -> None:
    """Test counting nodes."""
    graph_backend.add_node({"t": 0})
    graph_backend.add_node({"t": 1})

    assert graph_backend.num_nodes == 2


def test_edge_features_include_targets(graph_backend: BaseGraph) -> None:
    """Test the inclusive flag behavior in edge_features method."""
    # Add edge feature key
    graph_backend.add_edge_feature_key("weight", 0.0)

    # Create a graph with 4 nodes
    # Graph structure:
    #   node0 -> node1 -> node2 -> node3
    #        \             ^
    #         -> node3  ----+
    node0 = graph_backend.add_node({"t": 0})
    node1 = graph_backend.add_node({"t": 1})
    node2 = graph_backend.add_node({"t": 2})
    node3 = graph_backend.add_node({"t": 3})

    print(f"Created nodes: {node0=}, {node1=}, {node2=}, {node3=}")

    # Create edges with different weights for easy identification
    edge0 = graph_backend.add_edge(node0, node1, attributes={"weight": 0.1})  # node0 -> node1
    edge1 = graph_backend.add_edge(node1, node2, attributes={"weight": 0.2})  # node1 -> node2
    edge2 = graph_backend.add_edge(node2, node3, attributes={"weight": 0.3})  # node2 -> node3
    edge3 = graph_backend.add_edge(node3, node0, attributes={"weight": 0.4})  # node3 -> node0

    print(f"Created edges: {edge0=}, {edge1=}, {edge2=}, {edge3=}")

    # Get all edges for reference
    df_all = graph_backend.edge_features()
    print(f"All edges:\n{df_all}")

    # Test with include_targets=False (default)
    # When selecting [node1, node2, node3], should only include edges between these nodes:
    # - edge0: node0 -> node1 ✗ (node0 not in selection)
    # - edge1: node1 -> node2 ✓
    # - edge2: node2 -> node3 ✓
    # - edge3: node3 -> node0 ✗ (node0 not in selection)
    df_exclusive = graph_backend.edge_features(node_ids=[node1, node2, node3], include_targets=False)
    print(f"Exclusive edges (include_targets=False):\n{df_exclusive}")
    exclusive_edge_ids = set(df_exclusive[DEFAULT_ATTR_KEYS.EDGE_ID].to_list())
    expected_exclusive = {edge1, edge2}

    print(f"Expected exclusive edge IDs: {expected_exclusive}")
    print(f"Actual exclusive edge IDs: {exclusive_edge_ids}")

    msg = f"include_targets=False: Expected {expected_exclusive}, got {exclusive_edge_ids}"
    assert exclusive_edge_ids == expected_exclusive, msg

    # Verify the weights match expected edges
    exclusive_weights = df_exclusive["weight"].to_list()
    expected_weights = [0.2, 0.3]  # weights for edge1, edge2
    assert sorted(exclusive_weights) == sorted(expected_weights)

    # Test with include_targets=True
    # When selecting [node2, node3], should include edges to neighbors:
    # - edge0: node0 -> node1 ✗ (node0 not in selection)
    # - edge1: node1 -> node2 ✗ (node1 not in selection)
    # - edge2: node2 -> node3 ✓
    # - edge3: node3 -> node0 ✓
    df_inclusive = graph_backend.edge_features(node_ids=[node2, node3], include_targets=True)
    print(f"Inclusive edges (include_targets=True):\n{df_inclusive}")
    inclusive_edge_ids = set(df_inclusive[DEFAULT_ATTR_KEYS.EDGE_ID].to_list())
    expected_inclusive = {edge2, edge3}

    print(f"Expected inclusive edge IDs: {expected_inclusive}")
    print(f"Actual inclusive edge IDs: {inclusive_edge_ids}")

    msg = f"include_targets=True: Expected {expected_inclusive}, got {inclusive_edge_ids}"
    assert inclusive_edge_ids == expected_inclusive, msg

    # Verify all weights are included
    inclusive_weights = df_inclusive["weight"].to_list()
    expected_all_weights = [0.3, 0.4]  # weights for all edges
    assert sorted(inclusive_weights) == sorted(expected_all_weights)

    # Test edge case: selecting only one node with include_targets=True
    # When selecting [node1], with include_targets=True should include edges to neighbors:
    # - edge0: node0 -> node1 ✗ (node1 not in selection)
    # - edge1: node1 -> node2 ✓
    # - edge2: node2 -> node3 ✗ (node1 not in selection)
    # - edge3: node3 -> node0 ✗ (node1 not in selection)
    df_single_inclusive = graph_backend.edge_features(node_ids=[node1], include_targets=True)
    print(f"Single node inclusive edges: {df_single_inclusive}")
    single_inclusive_edge_ids = set(df_single_inclusive[DEFAULT_ATTR_KEYS.EDGE_ID].to_list())
    expected_single_inclusive = {edge1}

    msg = f"Single node include_targets=True: Expected {expected_single_inclusive}, got {single_inclusive_edge_ids}"
    assert single_inclusive_edge_ids == expected_single_inclusive, msg

    # Test edge case: selecting only one node with include_targets=False
    # When selecting [node1], with include_targets=False should include no edges
    # (since there are no edges strictly between just node1)
    df_single_exclusive = graph_backend.edge_features(node_ids=[node1], include_targets=False)
    print(f"Single node exclusive edges: {df_single_exclusive}")
    single_exclusive_edge_ids = set(df_single_exclusive[DEFAULT_ATTR_KEYS.EDGE_ID].to_list())
    expected_single_exclusive = set()  # No edges strictly within [node1]

    msg = f"Single node include_targets=False: Expected {expected_single_exclusive}, got {single_exclusive_edge_ids}"
    assert single_exclusive_edge_ids == expected_single_exclusive, msg
