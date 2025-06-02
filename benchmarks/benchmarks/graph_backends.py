from tracksdata.edges._distance_edges import DistanceEdgesOperator
from tracksdata.graph._base_graph import BaseGraphBackend
from tracksdata.graph._rustworkx_graph import RustWorkXGraphBackend
from tracksdata.nodes._random import RandomNodes


class GraphBackendSuite:
    """
    Benchmark suite for graph backend operations.
    """

    params = (
        (RustWorkXGraphBackend,),
        (1_000, 10_000, 100_000),
    )
    timeout = 1800  # 30 minutes
    param_names = ("backend", "n_nodes")

    def setup(
        self,
        backend: BaseGraphBackend,
        n_nodes: int,
    ) -> None:
        self.graph = backend()
        self.nodes_operator = RandomNodes(
            n_time_points=50,
            n_nodes=(int(n_nodes * 0.95), int(n_nodes * 1.05)),
            n_dim=3,
            show_progress=False,
        )
        self.edges_operator = DistanceEdgesOperator(
            distance_threshold=10,
            n_neighbors=3,
            show_progress=False,
        )

    def time_simple_workflow(self, *args, **kwargs) -> None:
        # add nodes
        self.nodes_operator.add_nodes(self.graph)
        # add edges
        self.edges_operator.add_edges(self.graph)
