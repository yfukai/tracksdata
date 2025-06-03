import abc
from collections.abc import Sequence
from typing import Any

from tqdm import tqdm

from tracksdata.graph._base_graph import BaseGraphBackend


class BaseWeightsOperator(abc.ABC):
    """
    Base class indicating methods required to add weights to edges in a graph.
    It will interact with a `BaseGraphBackend` to do so.
    """

    def __init__(self, output_key: Sequence[str] | str, show_progress: bool = True):
        self.output_key = output_key
        self.show_progress = show_progress

    def add_weights(
        self,
        graph: BaseGraphBackend,
        *,
        t: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Add weights to the edges of the graph for a given time or all time points.

        Parameters
        ----------
        graph : BaseGraphBackend
            The graph to add weights to.
        t : int | None
            The time point to add weights for. If None, add weights for all time points.
        **kwargs : Any
            Additional keyword arguments to pass to the `_add_weights_per_time` method.
        """
        if t is None:
            for t in tqdm(graph.time_points(), disable=not self.show_progress, desc="Adding weights"):
                self._add_weights_per_time(
                    graph,
                    t=t,
                    **kwargs,
                )
        else:
            self._add_weights_per_time(
                graph,
                t=t,
                **kwargs,
            )

    @abc.abstractmethod
    def _add_weights_per_time(
        self,
        graph: BaseGraphBackend,
        *,
        t: int,
        **kwargs: Any,
    ) -> None:
        """
        Add weights to edges of a graph at a given time point.

        Parameters
        ----------
        graph : BaseGraphBackend
            The graph to add weights to.
        t : int
            The time point to add weights for.
        **kwargs : Any
            Additional keyword arguments.
        """
