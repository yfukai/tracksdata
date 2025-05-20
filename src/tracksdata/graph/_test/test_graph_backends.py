import polars as pl
import pytest

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraphBackend
from tracksdata.graph._rustworkx_graph import RustWorkXGraphBackend


@pytest.fixture(params=[RustWorkXGraphBackend])
def graph_backend(request) -> BaseGraphBackend:
    """Fixture that provides all implementations of BaseGraphBackend."""
    return request.param()


def test_node_validation(graph_backend: BaseGraphBackend) -> None:
    """Test node validation."""
    # 't' key must exist by default
    graph_backend.add_node({"t": 1})

    with pytest.raises(ValueError):
        graph_backend.add_node({"t": 0, "x": 1.0})


def test_edge_validation(graph_backend: BaseGraphBackend) -> None:
    """Test edge validation."""
    with pytest.raises(ValueError):
        graph_backend.add_edge(0, 1, {"weight": 0.5})


def test_add_node(graph_backend: BaseGraphBackend) -> None:
    """Test adding nodes with various attributes."""

    for key in ["x", "y"]:
        graph_backend.add_node_feature_key(key, None)

    node_id = graph_backend.add_node({"t": 0, "x": 1.0, "y": 2.0})
    assert isinstance(node_id, int)

    # Check node features
    print(node_id)
    df = graph_backend.node_features([node_id])
    print(df)
    assert df["t"].to_list() == [0]
    assert df["x"].to_list() == [1.0]
    assert df["y"].to_list() == [2.0]


def test_add_edge(graph_backend: BaseGraphBackend) -> None:
    """Test adding edges with attributes."""
    # Add node feature key
    graph_backend.add_node_feature_key("x", None)

    # Add two nodes first
    node1 = graph_backend.add_node({"t": 0, "x": 1.0})
    node2 = graph_backend.add_node({"t": 1, "x": 2.0})

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


def test_node_ids(graph_backend: BaseGraphBackend) -> None:
    """Test retrieving node IDs."""
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})

    assert set(graph_backend.node_ids()) == {node1, node2}


def test_filter_nodes_by_attribute(graph_backend: BaseGraphBackend) -> None:
    """Test filtering nodes by attributes."""
    node1 = graph_backend.add_node({"t": 0, "label": "A"})
    node2 = graph_backend.add_node({"t": 0, "label": "B"})
    graph_backend.add_node({"t": 1, "label": "A"})

    # Filter by time
    nodes = graph_backend.filter_nodes_by_attribute(t=0)
    assert set(nodes) == {node1, node2}

    # Filter by label
    nodes = graph_backend.filter_nodes_by_attribute(label="A")
    assert len(nodes) == 2


def test_subgraph(graph_backend: BaseGraphBackend) -> None:
    """Test creating subgraphs."""
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})
    graph_backend.add_node({"t": 2})

    subgraph = graph_backend.subgraph([node1, node2])
    assert isinstance(subgraph, BaseGraphBackend)


def test_time_points(graph_backend: BaseGraphBackend) -> None:
    """Test retrieving time points."""
    graph_backend.add_node({"t": 0})
    graph_backend.add_node({"t": 2})
    graph_backend.add_node({"t": 1})

    assert graph_backend.time_points() == [0, 1, 2]


def test_node_features(graph_backend: BaseGraphBackend) -> None:
    """Test retrieving node features."""
    node1 = graph_backend.add_node({"t": 0, "x": 1.0})
    node2 = graph_backend.add_node({"t": 1, "x": 2.0})

    df = graph_backend.node_features([node1, node2], feature_keys=["x"])
    assert isinstance(df, pl.DataFrame)
    assert df["x"].to_list() == [1.0, 2.0]


def test_edge_features(graph_backend: BaseGraphBackend) -> None:
    """Test retrieving edge features."""
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})

    graph_backend.add_edge_feature_key("weight", 0.0)
    graph_backend.add_edge(node1, node2, attributes={"weight": 0.5})

    df = graph_backend.edge_features(feature_keys=["weight"])
    assert isinstance(df, pl.DataFrame)
    assert df["weight"].to_list() == [0.5]


def test_add_node_feature_key(graph_backend: BaseGraphBackend) -> None:
    """Test adding new node feature keys."""
    node = graph_backend.add_node({"t": 0})
    graph_backend.add_node_feature_key("new_feature", 42)

    df = graph_backend.node_features([node], feature_keys=["new_feature"])
    assert df["new_feature"].to_list() == [42]


def test_add_edge_feature_key(graph_backend: BaseGraphBackend) -> None:
    """Test adding new edge feature keys."""
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})

    graph_backend.add_edge_feature_key("new_feature", 42)
    edge_id = graph_backend.add_edge(node1, node2, attributes={"new_feature": 42})

    df = graph_backend.edge_features([edge_id], feature_keys=["new_feature"])
    assert df["new_feature"].to_list() == [42]


def test_update_node_features(graph_backend: BaseGraphBackend) -> None:
    """Test updating node features."""
    node = graph_backend.add_node({"t": 0, "x": 1.0})
    graph_backend.update_node_features([node], {"x": 2.0})

    df = graph_backend.node_features([node], feature_keys=["x"])
    assert df["x"].to_list() == [2.0]


def test_update_edge_features(graph_backend: BaseGraphBackend) -> None:
    """Test updating edge features."""
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})

    graph_backend.add_edge_feature_key("weight", 0.0)
    edge_id = graph_backend.add_edge(node1, node2, attributes={"weight": 0.5})

    graph_backend.update_edge_features([edge_id], {"weight": 1.0})
    df = graph_backend.edge_features([edge_id], feature_keys=["weight"])
    assert df["weight"].to_list() == [1.0]


def test_num_edges(graph_backend: BaseGraphBackend) -> None:
    """Test counting edges."""
    node1 = graph_backend.add_node({"t": 0})
    node2 = graph_backend.add_node({"t": 1})

    graph_backend.add_edge_feature_key("weight", 0.0)
    graph_backend.add_edge(node1, node2, attributes={"weight": 0.5})

    assert graph_backend.num_edges == 1


def test_num_nodes(graph_backend: BaseGraphBackend) -> None:
    """Test counting nodes."""
    graph_backend.add_node({"t": 0})
    graph_backend.add_node({"t": 1})

    assert graph_backend.num_nodes == 2
