import abc

from tracksdata._graph import BaseGraphBackend

DEFAULT_EDGE_WEIGHT_KEY = "weight"


class BaseEdgesInitializer(abc.ABC):
    """
    Base class indicating methods required to insert edges into a graph.
    It will interact with a `BaseGraphBackend` to do so.
    """

    def __init__(self, graph: BaseGraphBackend):
        self._graph = graph

    @abc.abstractmethod
    def init_edges(self, time: int, weight_key: str = DEFAULT_EDGE_WEIGHT_KEY) -> None:
        """
        Initialize the edges from nodes of given `time` to nodes in neighboring times (`time` + `\delta time`)

        Parameters
        ----------
        time: int
            The time of the nodes to initialize the edges from.
        weight_key: str
            The key to add the edge weights to.
        """
