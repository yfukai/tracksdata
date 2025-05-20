import abc
from collections.abc import Sequence
from typing import Any

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraphBackend


class BaseEdgesOperator(abc.ABC):
    """
    Base class indicating methods required to insert edges into a graph.
    It will interact with a `BaseGraphBackend` to do so.
    """

    @abc.abstractmethod
    def add_edges(
        self,
        graph: BaseGraphBackend,
        *,
        t: int | None = None,
        weight_key: Sequence[str] | str = DEFAULT_ATTR_KEYS.EDGE_WEIGHT,
        **kwargs: Any,
    ) -> None:
        r"""
        Initialize the edges from nodes of given `time` to nodes in neighboring
        times (`time` + `\delta time`)

        Parameters
        ----------
        graph : BaseGraphBackend
            The graph to initialize the edges in.
        t: int
            The time of the nodes to initialize the edges from.
        weight_key: str
            The key to add the edge weights to.
        **kwargs: Any
            Additional keyword arguments to pass to the `add_edges` method.
        """
