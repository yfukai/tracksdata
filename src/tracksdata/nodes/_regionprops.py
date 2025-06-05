from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skimage.measure._regionprops import RegionProperties, regionprops
from tqdm import tqdm
from typing_extensions import override

from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.nodes._base_nodes import BaseNodesOperator
from tracksdata.nodes._mask import Mask


class RegionPropsNodes(BaseNodesOperator):
    """Operator that adds nodes to a graph using scikit-image's regionprops."""

    def __init__(
        self,
        cache: bool = True,
        extra_properties: list[str | Callable[[RegionProperties], Any]] | None = None,
        spacing: tuple[float, float] | None = None,
        show_progress: bool = True,
    ):
        super().__init__(show_progress=show_progress)
        self._cache = cache
        self._extra_properties = extra_properties or []
        self._spacing = spacing

    def features_keys(self) -> list[str]:
        """
        Get the keys of the features of the nodes.
        """
        return [prop.__name__ if callable(prop) else prop for prop in self._extra_properties]

    @override
    def add_nodes(
        self,
        graph: BaseGraph,
        *,
        labels: NDArray[np.integer],
        t: int | None = None,
        intensity_image: NDArray | None = None,
    ) -> None:
        """
        Initialize the nodes in the provided graph using skimage's region properties.
        If t is None, the labels are considered to be a timelapse where axis=0 is time.

        Parameters
        ----------
        graph : BaseGraph
            The graph to initialize the nodes in.
        labels : NDArray[np.integer]
            The labels of the nodes to be initialized.
        t : int | None
            The time at which to initialize the nodes.
            If None, labels are considered to be a timelapse where axis=0 is time.
        intensity_image : NDArray | None
            The intensity image to use for the region properties.
            If None, the intensity image is not used.
        """
        if t is None:
            for t in tqdm(range(labels.shape[0]), disable=not self.show_progress, desc="Adding nodes"):
                if intensity_image is not None:
                    self._add_nodes_per_time(
                        graph=graph,
                        labels=labels[t],
                        t=t,
                        intensity_image=intensity_image[t],
                    )
                else:
                    self._add_nodes_per_time(
                        graph=graph,
                        labels=labels[t],
                        t=t,
                    )
        else:
            self._add_nodes_per_time(
                graph=graph,
                labels=labels,
                t=t,
                intensity_image=intensity_image,
            )

    def _add_nodes_per_time(
        self,
        graph: BaseGraph,
        *,
        labels: NDArray[np.integer],
        t: int,
        intensity_image: NDArray | None = None,
    ) -> None:
        """
        Add nodes for a specific time point using region properties.

        Parameters
        ----------
        graph : BaseGraph
            The graph to add nodes to.
        labels : NDArray[np.integer]
            The labels for the specific time point.
        t : int
            The time point to add nodes for.
        intensity_image : NDArray | None
            The intensity image for the specific time point.
        """
        if labels.ndim == 2:
            axis_names = ["y", "x"]
        elif labels.ndim == 3:
            axis_names = ["z", "y", "x"]
        else:
            raise ValueError(f"`labels` must be 2D or 3D, got {labels.ndim} dimensions.")

        # initialize the feature keys
        for attr_key in [DEFAULT_ATTR_KEYS.MASK, *axis_names] + [
            p.__name__ if callable(p) else p for p in self._extra_properties
        ]:
            if attr_key not in graph.node_features_keys:
                graph.add_node_feature_key(attr_key, None)

        labels = np.asarray(labels)

        for obj in regionprops(
            labels,
            intensity_image=intensity_image,
            spacing=self._spacing,
        ):
            attributes = dict(zip(axis_names, obj.centroid, strict=False))

            for prop in self._extra_properties:
                if callable(prop):
                    attributes[prop.__name__] = prop(obj)
                else:
                    attributes[prop] = getattr(obj, prop)

            attributes[DEFAULT_ATTR_KEYS.MASK] = Mask(obj.image, obj.bbox)
            attributes[DEFAULT_ATTR_KEYS.T] = t

            graph.add_node(attributes, validate_keys=False)
