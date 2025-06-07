import abc

from tracksdata.graph._base_graph import BaseGraph


class BaseSolver(abc.ABC):
    """
    Base class for a solver tracking problem solver.
    It should return a subset of nodes and edges that are part of the solution.
    This class will interact with a `BaseGraph` to do so.
    """

    @abc.abstractmethod
    def solve(self, graph: BaseGraph) -> None:
        """
        Solve the tracking problem and add the result to the graph with
        key `solution_key`.

        Parameters
        ----------
        graph : BaseGraph
            The graph to solve the tracking problem on.
        """
