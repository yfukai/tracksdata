import abc
from collections.abc import Sequence
from typing import Any

from tqdm import tqdm

from tracksdata.graph._base_graph import BaseGraph


class BaseNodeAttrsOperator(abc.ABC):
    """
    Base class indicating methods required to add attributes to nodes in a graph.
    It will interact with a `BaseGraph` to do so.
    """

    def __init__(
        self,
        output_key: Sequence[str] | str,
        show_progress: bool = True,
    ) -> None:
        self.output_key = output_key
        self.show_progress = show_progress

    def add_node_attrs(
        self,
        graph: BaseGraph,
        *,
        t: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Add attributes to nodes of a graph.

        Parameters
        ----------
        graph : BaseGraph
            The graph to add attributes to.
        t : int | None
            The time point to add attributes for. If None, add attributes for all time points.
        **kwargs : Any
            Additional keyword arguments to pass to the `_add_node_attrs_per_time` method.
        """
        if t is None:
            for t in tqdm(graph.time_points(), disable=not self.show_progress, desc="Adding node attributes"):
                self._add_node_attrs_per_time(
                    graph,
                    t=t,
                    **kwargs,
                )
        else:
            self._add_node_attrs_per_time(
                graph,
                t=t,
                **kwargs,
            )

    @abc.abstractmethod
    def _add_node_attrs_per_time(
        self,
        graph: BaseGraph,
        *,
        t: int,
        **kwargs: Any,
    ) -> None:
        """
        Add attributes to nodes of a graph at a given time point.

        Parameters
        ----------
        graph : BaseGraph
            The graph to add attributes to.
        t : int
            The time point to add attributes for. If None, add attributes for all time points.
        **kwargs : Any
            Additional keyword arguments to pass to the `_add_node_attrs_per_time` method.
        """
