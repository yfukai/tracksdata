import time
from collections.abc import Callable

import polars as pl
from tabulate import tabulate

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.edges import DistanceEdges
from tracksdata.expr import AttrExpr
from tracksdata.graph import RustWorkXGraph, SQLGraph
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.nodes import RandomNodes
from tracksdata.solvers import NearestNeighborsSolver


class SQLGraphWithMemory(SQLGraph):
    def __init__(self):
        super().__init__(drivername="sqlite", database=":memory:")


class SQLGraphDisk(SQLGraph):
    def __init__(self):
        import datetime

        path = f"/tmp/_asv_tracksdata_db_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        super().__init__(drivername="sqlite", database=path)


# class GraphSuite:
#     """
#     Benchmark suite for graph backend operations.
#     """
#
#     params = (
#         (
#             RustWorkXGraph,
#             SQLGraphWithMemory,
#             SQLGraphDisk,
#         ),
#         (1_000, 10_000, 100_000),
#     )
#     timeout = 300  # 5 minutes
#     param_names = ("backend", "n_nodes")
#
#     def setup(
#         self,
#         backend: BaseGraph,
#         n_nodes: int,
#     ) -> None:
#         self.graph = backend()
#         n_time_points = 50
#         n_nodes /= 50
#         self.nodes_operator = RandomNodes(
#             n_time_points=n_time_points,
#             n_nodes_per_tp=(int(n_nodes * 0.95), int(n_nodes * 1.05)),
#             n_dim=3,
#             show_progress=False,
#         )
#         self.edges_operator = DistanceEdges(
#             distance_threshold=10,
#             n_neighbors=3,
#             show_progress=False,
#         )
#
#     def time_simple_workflow(self, *args, **kwargs) -> None:
#         # add nodes
#         self.nodes_operator.add_nodes(self.graph)
#         # add edges
#         self.edges_operator.add_edges(self.graph)
#


def _run_benchmark(
    backend: type[BaseGraph],
    pipeline: list[tuple[str, Callable[[BaseGraph], None]]],
) -> pl.DataFrame:
    data = []
    total_time = 0

    start = time.perf_counter()
    graph = backend()
    end = time.perf_counter()

    data.append(
        {
            "operation": "init",
            "time": end - start,
        }
    )
    total_time += end - start

    for name, func in pipeline:
        start = time.perf_counter()
        func(graph)
        end = time.perf_counter()
        data.append(
            {
                "operation": name,
                "time": end - start,
            }
        )
        total_time += end - start

    data.append(
        {
            "operation": "total",
            "time": total_time,
        }
    )

    df = pl.DataFrame(data)
    df = df.with_columns(
        pl.col("operation").cast(pl.Categorical),
        pl.col("time").cast(pl.Float64),
    )
    return df


def _assing_tracks(graph: BaseGraph) -> None:
    solution_graph = graph.subgraph(edge_attr_filter={DEFAULT_ATTR_KEYS.SOLUTION: True})
    solution_graph.assign_track_ids()


def _format_markdown_table(df: pl.DataFrame) -> str:
    # Create time string column
    df = df.with_columns(time_str=pl.format("{} ± {}", pl.col("time_avg").round(3), pl.col("time_std").round(3)))

    # Pivot the table to get n_nodes as columns
    pivoted = df.pivot(values="time_str", index=["backend", "operation"], columns="n_nodes", aggregate_function="first")

    # Format the results for markdown table
    prev_backend = pivoted["backend"][0]
    table_data = []
    for row in pivoted.iter_rows(named=True):
        if row["backend"] != prev_backend:
            table_data.append([""] * (len(df["n_nodes"].unique()) + 2))  # Empty row for separation
        table_data.append([row["backend"], row["operation"], *[row[str(n)] for n in sorted(df["n_nodes"].unique())]])
        prev_backend = row["backend"]

    # Print markdown table
    node_counts = sorted(df["n_nodes"].unique())
    headers = ["Backend", "Operation"] + [f"{n:,} nodes" for n in node_counts]
    print("\nBenchmark Results:")
    print(tabulate(table_data, headers=headers, tablefmt="pipe", numalign="right"))


def main() -> None:
    data = []
    n_repeats = 2
    n_time_points = 50

    for _ in range(n_repeats):
        for n_nodes in [1_000, 10_000]:  # , 100_000]:
            n_nodes_per_tp = int(n_nodes / n_time_points)
            pipeline = [
                (
                    "random_nodes",
                    RandomNodes(
                        n_time_points=n_time_points,
                        n_nodes_per_tp=(n_nodes_per_tp * 0.95, n_nodes_per_tp * 1.05),
                        n_dim=3,
                        show_progress=False,
                    ).add_nodes,
                ),
                ("distance_edges", DistanceEdges(distance_threshold=10, n_neighbors=5, show_progress=False).add_edges),
                (
                    "nearest_neighbors_solver",
                    NearestNeighborsSolver(edge_weight=-AttrExpr(DEFAULT_ATTR_KEYS.EDGE_WEIGHT), max_children=2).solve,
                ),
                ("assing_tracks", _assing_tracks),
            ]
            for backend in [RustWorkXGraph, SQLGraphWithMemory]:  # , SQLGraphDisk]:
                df = _run_benchmark(backend, pipeline)
                df = df.with_columns(
                    backend=pl.lit(backend.__name__),
                    n_nodes=pl.lit(n_nodes),
                )
                data.append(df)

    df = pl.concat(data)
    df = df.group_by(["backend", "n_nodes", "operation"], maintain_order=True).agg(
        pl.col("time").std().alias("time_std"),
        pl.col("time").mean().alias("time_avg"),
    )
    _format_markdown_table(df)


if __name__ == "__main__":
    main()
