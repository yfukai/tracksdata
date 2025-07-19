import bidict
import pytest
import rustworkx as rx

from tracksdata.graph import IndexedRXGraph


def test_index_rx_graph_with_mapping() -> None:
    index_map = {1: 0, 5_000: 1, 3: 2}

    rx_graph = rx.PyDiGraph()
    rx_graph.add_nodes_from([{} for _ in range(len(index_map))])
    rx_graph.add_edges_from([(0, 1, {}), (1, 2, {})])

    graph = IndexedRXGraph(
        rx_graph=rx_graph,
        node_id_map=index_map,
    )

    assert graph.node_ids() == [1, 5_000, 3]


def test_duplicate_index_map() -> None:
    index_map = {0: 1, 1: 5_000, 2: 3, 3: 1}

    rx_graph = rx.PyDiGraph()
    rx_graph.add_nodes_from([{} for _ in range(len(index_map))])
    rx_graph.add_edges_from([(0, 1, {}), (1, 2, {}), (2, 3, {}), (3, 0, {})])

    with pytest.raises(bidict.ValueDuplicationError):
        IndexedRXGraph(rx_graph=rx_graph, node_id_map=index_map)

    graph = IndexedRXGraph()

    graph.add_node({"t": 0}, index=3)

    with pytest.raises(bidict.KeyDuplicationError):
        graph.add_node({"t": 5}, index=3)
