import abc
from collections.abc import Sequence
from typing import Any

from tqdm import tqdm

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraphBackend


class BaseEdgesOperator(abc.ABC):
    """
    Base class indicating methods required to insert edges into a graph.
    It will interact with a `BaseGraphBackend` to do so.
    """

    def __init__(self, show_progress: bool = True):
        self.show_progress = show_progress

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
        if t is None:
            for t in tqdm(graph.time_points(), disable=not self.show_progress, desc="Adding edges"):
                self._add_edges_per_time(
                    graph,
                    t=t,
                    weight_key=weight_key,
                    **kwargs,
                )
        else:
            self._add_edges_per_time(
                graph,
                t=t,
                weight_key=weight_key,
                **kwargs,
            )

    @abc.abstractmethod
    def _add_edges_per_time(
        self,
        graph: BaseGraphBackend,
        t: int,
        weight_key: Sequence[str] | str = DEFAULT_ATTR_KEYS.EDGE_WEIGHT,
        **kwargs: Any,
    ) -> None:
        """
        Add edges to a graph at a given time point.

        Parameters
        ----------
        graph : BaseGraphBackend
            The graph to add edges to.
        t : int
            The time point to add edges for.
        weight_key : Sequence[str] | str
            The key to add the edge weights to.
        **kwargs : Any
            Additional keyword arguments to pass to the `_add_edges_per_time` method.
        """
