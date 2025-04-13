import abc

from tracksdata._graph import BaseGraphBackend


class BaseNodesInitializer(abc.ABC):
    """
    Base class indicating methods required to insert nodes into a graph.
    It will interact with a `BaseGraphBackend` to do so.
    """

    def __init__(self, graph: BaseGraphBackend):
        self._graph = graph

    @abc.abstractmethod
    def __call__(self, time: int | None = None) -> None:
        """
        Initialize the nodes for a given time.
        """
