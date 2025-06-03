import abc
from typing import Any

from tracksdata.graph._base_graph import BaseGraphBackend


class BaseNodesOperator(abc.ABC):
    """
    Base class indicating methods required to insert nodes into a graph.
    It will interact with a `BaseGraphBackend` to do so.
    """

    def __init__(self, show_progress: bool = True):
        self.show_progress = show_progress

    @abc.abstractmethod
    def add_nodes(
        self,
        graph: BaseGraphBackend,
        *,
        t: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the nodes for a given time or all time points.

        Parameters
        ----------
        graph : BaseGraphBackend
            The graph to initialize the nodes in.
        t : int | None
            The time point to add nodes for. If None, add nodes for all time points.
        **kwargs : Any
            Additional keyword arguments to pass to the `_add_nodes_per_time` method.
        """
