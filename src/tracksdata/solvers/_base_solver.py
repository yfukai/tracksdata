import abc

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraphBackend


class BaseSolver(abc.ABC):
    """
    Base class for a solver tracking problem solver.
    It should return a subset of nodes and edges that are part of the solution.
    This class will interact with a `BaseGraphBackend` to do so.
    """

    @abc.abstractmethod
    def solve(
        self, graph: BaseGraphBackend, solution_key: str = DEFAULT_ATTR_KEYS.SOLUTION
    ) -> None:
        """
        Solve the tracking problem and add the result to the graph with
        key `solution_key`.

        Parameters
        ----------
        graph : BaseGraphBackend
            The graph to solve the tracking problem on.
        solution_key: str
            The edge and node attributes key to add the solution to.
        """
