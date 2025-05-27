import numpy as np

from tracksdata.graph._rustworkx_graph import RustWorkXGraphBackend


class GraphBackendSuite:
    """
    Benchmark suite for graph backend operations.
    """

    params = (100, 1000, 10000)
    param_names = ("n_nodes",)

    def setup(self, n_nodes):
        self.graph = RustWorkXGraphBackend()

        # Add some node feature keys
        self.graph.add_node_feature_key("x", None)
        self.graph.add_node_feature_key("y", None)

        # Add nodes with random attributes
        self.nodes = []
        for i in range(n_nodes):
            attrs = {"t": i % 10, "x": np.random.random(), "y": np.random.random()}
            node_id = self.graph.add_node(attrs)
            self.nodes.append(node_id)

        # Add edge feature key
        self.graph.add_edge_feature_key("weight", 0.0)

        # Add random edges
        self.edges = []
        n_edges = n_nodes * 2
        for _ in range(n_edges):
            source = np.random.choice(self.nodes)
            target = np.random.choice(self.nodes)
            edge_id = self.graph.add_edge(source, target, {"weight": np.random.random()})
            self.edges.append(edge_id)

    def time_add_node(self, n_nodes):
        self.graph.add_node({"t": 0, "x": 1.0, "y": 2.0})

    def time_add_edge(self, n_nodes):
        source = np.random.choice(self.nodes)
        target = np.random.choice(self.nodes)
        self.graph.add_edge(source, target, {"weight": 0.5})

    def time_node_features(self, n_nodes):
        self.graph.node_features(self.nodes[:100])

    def time_edge_features(self, n_nodes):
        self.graph.edge_features(self.edges[:100])

    def time_filter_nodes_by_attribute(self, n_nodes):
        self.graph.filter_nodes_by_attribute({"t": 0})

    def time_update_node_features(self, n_nodes):
        nodes = self.nodes[:100]
        self.graph.update_node_features(nodes, {"x": np.random.random(len(nodes))})

    def time_update_edge_features(self, n_nodes):
        edges = self.edges[:100]
        self.graph.update_edge_features(edges, {"weight": np.random.random(len(edges))})
