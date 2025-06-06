from tracksdata.edges._distance_edges import DistanceEdges
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.graph._rustworkx_graph import RustWorkXGraph
from tracksdata.graph._sql_graph import SQLGraph as _SQLGraph
from tracksdata.nodes._random import RandomNodes


class SQLGraphWithMemory(_SQLGraph):
    def __init__(self):
        super().__init__(drivername="sqlite", database=":memory:")


class SQLGraphDisk(_SQLGraph):
    def __init__(self):
        import datetime

        path = f"/tmp/_asv_tracksdata_db_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        super().__init__(drivername="sqlite", database=path)


class GraphSuite:
    """
    Benchmark suite for graph backend operations.
    """

    params = (
        (
            RustWorkXGraph,
            SQLGraphWithMemory,
            SQLGraphDisk,
        ),
        (1_000, 10_000, 100_000),
    )
    timeout = 300  # 5 minutes
    param_names = ("backend", "n_nodes")

    def setup(
        self,
        backend: BaseGraph,
        n_nodes: int,
    ) -> None:
        self.graph = backend()
        self.nodes_operator = RandomNodes(
            n_time_points=50,
            n_nodes=(int(n_nodes * 0.95), int(n_nodes * 1.05)),
            n_dim=3,
            show_progress=False,
        )
        self.edges_operator = DistanceEdges(
            distance_threshold=10,
            n_neighbors=3,
            show_progress=False,
        )

    def time_simple_workflow(self, *args, **kwargs) -> None:
        # add nodes
        self.nodes_operator.add_nodes(self.graph)
        # add edges
        self.edges_operator.add_edges(self.graph)
