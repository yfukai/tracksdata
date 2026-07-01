from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from polars.datatypes import numpy_char_code_to_dtype
from skimage.measure._regionprops import RegionProperties, regionprops
from typing_extensions import override

from tracksdata.attrs import NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.nodes._base_nodes import BaseNodesOperator
from tracksdata.nodes._mask import Mask
from tracksdata.utils._logging import LOG
from tracksdata.utils._multiprocessing import multiprocessing_apply


def _validate_properties(properties: list[str | Callable[[RegionProperties], Any]]) -> None:
    """
    Reject properties that are already added by default by `RegionPropsNodes`.

    Parameters
    ----------
    properties : list[str | Callable[[RegionProperties], Any]]
        The requested region properties.
    """
    if "centroid" in properties:
        raise ValueError(
            "`centroid` is not supported as an extra property. It's already included by default as (z), y, x."
        )
    if "bbox" in properties:
        raise ValueError("`bbox` is not supported as an extra property. It's already included by default.")


def _add_missing_node_attr_keys(graph: BaseGraph, node_attrs: dict[str, Any]) -> None:
    """
    Register node attribute keys in the graph, inferring dtypes from sample values.

    Keys already present in the graph are left untouched.

    Parameters
    ----------
    graph : BaseGraph
        The graph to register the attribute keys in.
    node_attrs : dict[str, Any]
        A sample of node attributes, mapping each key to a single value
        used to infer the dtype.
    """
    node_attr_keys = graph.node_attr_keys(return_ids=True)
    for key, value in node_attrs.items():
        if key not in node_attr_keys:
            if isinstance(value, np.ndarray):
                default_value = np.zeros_like(value)
                graph.add_node_attr_key(
                    key, pl.Array(numpy_char_code_to_dtype(value.dtype), value.shape), default_value
                )
            elif np.isscalar(value):
                dtype = numpy_char_code_to_dtype(value.dtype) if hasattr(value, "dtype") else type(value)
                graph.add_node_attr_key(key, dtype)
            elif type(value).__module__ != "builtins":
                graph.add_node_attr_key(key, pl.Object)
            else:
                graph.add_node_attr_key(key, type(value))


def _region_property_attrs(
    obj: RegionProperties,
    properties: list[str | Callable[[RegionProperties], Any]],
    channel_names: list[str] | None,
) -> dict[str, Any]:
    """
    Compute the requested region properties for a single region.

    When ``channel_names`` is provided (multichannel intensity images), any
    property whose value has a trailing axis matching the number of channels
    is split into one attribute per channel, named ``{property}_{channel_name}``
    (e.g. ``intensity_mean_DAPI``). This is the case for intensity-based
    properties computed from a multichannel intensity image, where scikit-image
    appends the channel axis as the last axis of the returned value.

    Parameters
    ----------
    obj : RegionProperties
        The scikit-image region to compute the properties for.
    properties : list[str | Callable[[RegionProperties], Any]]
        The properties to compute. Strings are looked up on ``obj``, callables
        are called with ``obj`` and named after their ``__name__``.
    channel_names : list[str] | None
        Names of the intensity image channels. If None, no per-channel splitting
        is performed.

    Returns
    -------
    dict[str, Any]
        Mapping from attribute key to value for this region.
    """
    attrs: dict[str, Any] = {}
    for prop in properties:
        if callable(prop):
            name = prop.__name__
            value = prop(obj)
        else:
            name = prop
            value = getattr(obj, prop)

        if channel_names is not None and np.ndim(value) >= 1 and np.shape(value)[-1] == len(channel_names):
            value = np.asarray(value)
            for channel_idx, channel_name in enumerate(channel_names):
                channel_value = value[..., channel_idx]
                # a scalar-per-channel property (e.g. intensity_mean) yields a 0-d
                # array here; unwrap it to a numpy scalar to keep the stored dtype
                if channel_value.ndim == 0:
                    channel_value = channel_value[()]
                attrs[f"{name}_{channel_name}"] = channel_value
        else:
            attrs[name] = value

    return attrs


class RegionPropsNodes(BaseNodesOperator):
    """
    Operator that adds nodes and (re-)computes their region properties using scikit-image's regionprops.

    Extracts region properties from labeled images to create graph nodes using
    scikit-image's regionprops function to compute geometric and intensity-based
    features. Automatically adds centroid coordinates and mask information, with
    additional properties computed based on the ``extra_properties`` parameter.

    The same operator can also (re-)compute properties for nodes that already
    exist in a graph, evaluating regionprops on each node's stored
    [Mask][tracksdata.nodes.Mask] via [add_node_attrs][tracksdata.nodes.RegionPropsNodes.add_node_attrs].
    This is useful to compute properties that were not requested when the nodes
    were created (e.g. intensity features of an additional channel) or to refresh
    properties after masks were modified, without rebuilding the graph.

    Multichannel intensity images are supported: pass ``channel_names`` and an
    intensity image whose last axis is the channel axis. Intensity-based
    properties are then split into one attribute per channel, named
    ``{property}_{channel_name}`` (e.g. ``intensity_mean_DAPI``).

    Parameters
    ----------
    extra_properties : list[str | Callable[[RegionProperties], Any]] | None, optional
        Additional properties to compute for each region. Can be:
        - String names of built-in regionprops properties (e.g. 'area', 'perimeter')
        - Callable functions that take a RegionProperties object and return a value
        If None, only centroid coordinates and masks are extracted.
    spacing : tuple[float, float] | None, optional
        Physical spacing between pixels. If provided, affects distance-based
        measurements. Should be (row_spacing, col_spacing) for 2D or
        (depth_spacing, row_spacing, col_spacing) for 3D.
    channel_names : list[str] | None, optional
        Names of the channels of a multichannel intensity image. The channel axis
        must be the last axis of the intensity image (scikit-image convention).
        When provided, intensity-based properties are split into one attribute per
        channel, suffixed with the channel name (e.g. ``intensity_mean_DAPI``).
    mask_key : str, optional
        The key of the node attribute holding the [Mask][tracksdata.nodes.Mask]
        objects, used by [add_node_attrs][tracksdata.nodes.RegionPropsNodes.add_node_attrs].

    Attributes
    ----------
    _extra_properties : list
        List of additional properties to compute.
    _spacing : tuple[float, float] | None
        Physical spacing between pixels.
    _channel_names : list[str] | None
        Names of the channels of a multichannel intensity image.

    Examples
    --------
    Create a basic RegionPropsNodes operator:

    ```python
    from tracksdata.nodes import RegionPropsNodes

    node_op = RegionPropsNodes()
    ```

    Add common geometric properties:

    ```python
    node_op = RegionPropsNodes(extra_properties=["area", "perimeter", "eccentricity"])
    ```

    Add custom properties using functions:

    ```python
    def custom_property(region):
        return region.area / region.perimeter


    node_op = RegionPropsNodes(extra_properties=["area", custom_property])
    ```

    Use with physical spacing:

    ```python
    node_op = RegionPropsNodes(
        spacing=(0.5, 0.1, 0.1),  # z, y, x spacing
        extra_properties=["area", "volume"],
    )
    ```

    Add nodes from a time series:

    ```python
    labels_series = np.random.randint(0, 10, (10, 100, 100))
    node_op.add_nodes(graph, labels=labels_series)
    ```

    Compute intensity features per channel of a multichannel image (last axis is the channel axis):

    ```python
    node_op = RegionPropsNodes(
        extra_properties=["intensity_mean"],
        channel_names=["DAPI", "GFP"],
    )
    # intensity_image[t] has shape (..., y, x, 2)
    node_op.add_nodes(graph, labels=labels, intensity_image=intensity_image)
    # creates 'intensity_mean_DAPI' and 'intensity_mean_GFP' attributes
    ```

    Recompute properties of an additional channel on an existing graph:

    ```python
    node_op = RegionPropsNodes(extra_properties=["intensity_mean", "intensity_max"])
    node_op.add_node_attrs(graph, intensity_image=second_channel)
    ```
    """

    def __init__(
        self,
        extra_properties: list[str | Callable[[RegionProperties], Any]] | None = None,
        spacing: tuple[float, float] | None = None,
        channel_names: list[str] | None = None,
        mask_key: str = DEFAULT_ATTR_KEYS.MASK,
    ):
        super().__init__()
        self._extra_properties = extra_properties or []
        _validate_properties(self._extra_properties)
        self._spacing = spacing
        self._channel_names = list(channel_names) if channel_names is not None else None
        self._mask_key = mask_key

    def _axis_names(self, labels: NDArray[np.integer]) -> list[str]:
        """
        Get the names of the axes of the labels.

        Parameters
        ----------
        labels : NDArray[np.integer]
            The (t + nD) labels to get the axis names for.

        Returns
        -------
        list[str]
            The names of the axes of the labels.
        """
        if labels.ndim == 3:
            return [DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X]
        elif labels.ndim == 4:
            return [DEFAULT_ATTR_KEYS.Z, DEFAULT_ATTR_KEYS.Y, DEFAULT_ATTR_KEYS.X]
        else:
            raise ValueError(f"`labels` must be 't + 2D' or 't + 3D', got '{labels.ndim}' dimensions.")

    def _check_channels(self, intensity_image: NDArray | None) -> None:
        """
        Validate that ``intensity_image`` is consistent with ``channel_names``.

        The channel axis is expected to be the last axis of the intensity image.

        Parameters
        ----------
        intensity_image : NDArray | None
            The (possibly multichannel) intensity image, indexed by time point.
        """
        if self._channel_names is None:
            return

        if intensity_image is None:
            raise ValueError("`channel_names` was provided but no `intensity_image` was given.")

        n_channels = intensity_image.shape[-1]
        if n_channels != len(self._channel_names):
            raise ValueError(
                f"Number of channels in `intensity_image` ({n_channels}) does not match "
                f"the number of `channel_names` ({len(self._channel_names)}). "
                "The channel axis is expected to be the last axis of the intensity image."
            )

    def _init_node_attrs(self, graph: BaseGraph, node_attrs: dict[str, Any]) -> None:
        """
        Initialize the node attributes for the graph.
        """
        _add_missing_node_attr_keys(graph, node_attrs)

    def attr_keys(self) -> list[str]:
        """
        Get the base keys of the node attributes that will be extracted.

        Returns only the keys for ``extra_properties``. The centroid coordinates
        (x, y, z) and mask are always included but not listed here. When
        ``channel_names`` is set, intensity-based properties are additionally
        suffixed with the channel name at compute time (e.g. ``intensity_mean_DAPI``),
        which is not reflected in this list.

        Returns
        -------
        list[str]
            List of attribute key names that will be added to nodes.

        Examples
        --------
        ```python
        node_op = RegionPropsNodes(extra_properties=["area", "perimeter"])
        keys = node_op.attr_keys()
        print(keys)  # ['area', 'perimeter']
        ```
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
        Add nodes to a graph using region properties from labeled images.

        Extracts region properties from labeled images and creates corresponding
        nodes in the graph. Can handle both single time point and time series data.
        When t is None, the first axis of labels represents time and processes each
        time point sequentially. Automatically initializes required attribute keys
        in the graph schema before adding nodes.

        Parameters
        ----------
        graph : BaseGraph
            The graph to add nodes to.
        labels : NDArray[np.integer]
            Labeled image(s) where each unique positive integer represents
            a different region/object. Can be:
            - 3D array (time, height, width) for 2D time series
            - 4D array (time, depth, height, width) for 3D time series
            When `t` is provided, it should be padded to include the time dimension.
        t : int | None, optional
            Time point for the nodes. If None, labels are treated as a time
            series where the first axis represents time.
        intensity_image : NDArray | None, optional
            Intensity image(s) corresponding to the labels. Used for computing
            intensity-based properties. Must have the same shape as labels
            (excluding the label values). For multichannel images an extra
            trailing channel axis is expected; see ``channel_names``.

        Examples
        --------
        Add nodes from a single 2D labeled image:

        ```python
        labels = skimage.measure.label(binary_image)
        node_op.add_nodes(graph, labels=labels, t=0)
        ```

        Add nodes from a time series:

        ```python
        labels_series = np.stack(
            [
                skimage.measure.label(binary_image_t0),
                skimage.measure.label(binary_image_t1),
            ]
        )
        node_op.add_nodes(graph, labels=labels_series)
        ```

        Add nodes with intensity information:

        ```python
        node_op.add_nodes(graph, labels=labels, t=0, intensity_image=fluorescence_image)
        ```
        """
        if "shape" not in graph.metadata:
            graph.metadata.update(shape=labels.shape)

        self._check_channels(intensity_image)

        if t is None:
            time_points = range(labels.shape[0])
        else:
            time_points = [t]

        node_ids = []
        initialized = False
        for nodes_data in multiprocessing_apply(
            func=partial(self._nodes_per_time, labels=labels, intensity_image=intensity_image),
            sequence=time_points,
            desc="Adding region properties nodes",
        ):
            if not initialized and len(nodes_data):
                self._init_node_attrs(graph, nodes_data[0])
            node_ids.extend(graph.bulk_add_nodes(nodes_data))

    def _nodes_per_time(
        self,
        t: int,
        *,
        labels: NDArray[np.integer],
        intensity_image: NDArray | None = None,
    ) -> list[dict[str, Any]]:
        """
        Add nodes for a specific time point using region properties.

        Processes a single time point, computing region properties for each labeled
        region and creating corresponding graph nodes. Determines spatial dimensions
        from label shape, ensures required attribute keys exist, computes region
        properties, extracts coordinates and extra properties, creates mask objects,
        and bulk adds all nodes.

        Parameters
        ----------
        t : int
            The time point to assign to the created nodes.
        labels : NDArray[np.integer]
            2D or 3D labeled image for a single time point.
        intensity_image : NDArray | None, optional
            Corresponding intensity image for computing intensity-based properties.

        Returns
        -------
        list[dict[str, Any]]
            The nodes to add to the graph.

        Raises
        ------
        ValueError
            If labels is not 2D or 3D.
        """
        axis_names = self._axis_names(labels)

        labels = np.asarray(labels[t])

        if intensity_image is not None:
            intensity_image = np.asarray(intensity_image[t])

        nodes_data = []

        for obj in regionprops(
            labels,
            intensity_image=intensity_image,
            spacing=self._spacing,
            cache=True,
        ):
            attrs = dict(zip(axis_names, obj.centroid, strict=False))

            attrs.update(_region_property_attrs(obj, self._extra_properties, self._channel_names))

            attrs[DEFAULT_ATTR_KEYS.MASK] = Mask(obj.image, obj.bbox)
            attrs[DEFAULT_ATTR_KEYS.BBOX] = np.asarray(obj.bbox, dtype=int)
            attrs[DEFAULT_ATTR_KEYS.T] = t

            nodes_data.append(attrs)
            obj._cache.clear()  # clearing to reduce memory footprint

        if len(nodes_data) == 0:
            LOG.warning("No valid nodes found for time point %d", t)

        return nodes_data

    def add_node_attrs(
        self,
        graph: BaseGraph,
        *,
        t: int | None = None,
        intensity_image: NDArray | None = None,
    ) -> None:
        """
        (Re-)compute region properties from the node masks and store them as node attributes.

        For each node, scikit-image's regionprops is evaluated on the node's
        [Mask][tracksdata.nodes.Mask] attribute (``mask_key``), optionally combined
        with a given intensity image cropped to the mask bounding box. Missing
        output attribute keys are registered in the graph with dtypes inferred from
        the first computed values; existing keys are overwritten.

        Parameters
        ----------
        graph : BaseGraph
            The graph to add attributes to.
        t : int | None, optional
            The time point to compute attributes for.
            If None, attributes are computed for all time points of the graph.
        intensity_image : NDArray | None, optional
            Intensity image used for computing intensity-based properties,
            indexed by time point such that `intensity_image[t]` is the frame
            matching the masks at time point `t`. For multichannel images an
            extra trailing channel axis is expected; see ``channel_names``.

        Examples
        --------
        Compute intensity features from an additional channel on an existing graph:

        ```python
        node_op = RegionPropsNodes(extra_properties=["intensity_mean", "intensity_max"])
        node_op.add_node_attrs(graph, intensity_image=second_channel)
        ```
        """
        if not self._extra_properties:
            raise ValueError("`extra_properties` must contain at least one region property to compute node attributes.")

        if self._mask_key not in graph.node_attr_keys():
            raise ValueError(f"Mask key '{self._mask_key}' not found in graph. Expected '{graph.node_attr_keys()}'")

        self._check_channels(intensity_image)

        if t is None:
            time_points = graph.time_points()
        else:
            time_points = [t]

        initialized = False
        for node_ids, node_attrs in multiprocessing_apply(
            func=partial(self._node_attrs_per_time, graph=graph, intensity_image=intensity_image),
            sequence=time_points,
            desc="Computing region properties attributes",
        ):
            if len(node_ids) == 0:
                continue
            if not initialized:
                sample_attrs = {key: values[0] for key, values in node_attrs.items()}
                _add_missing_node_attr_keys(graph, sample_attrs)
                initialized = True
            graph.update_node_attrs(node_ids=node_ids, attrs=node_attrs)

    def _node_attrs_per_time(
        self,
        t: int,
        *,
        graph: BaseGraph,
        intensity_image: NDArray | None = None,
    ) -> tuple[list[int], dict[str, list[Any]]]:
        """
        Compute region properties for the nodes of a single time point.

        Parameters
        ----------
        t : int
            The time point to compute attributes for.
        graph : BaseGraph
            The graph to add attributes to.
        intensity_image : NDArray | None, optional
            Intensity image indexed by time point, see `add_node_attrs`.

        Returns
        -------
        tuple[list[int], dict[str, list[Any]]]
            The node ids and the attributes to add to the graph.
        """
        graph_filter = graph.filter(NodeAttr(DEFAULT_ATTR_KEYS.T) == t)
        node_ids = graph_filter.node_ids()

        if len(node_ids) == 0:
            LOG.warning("No nodes found for time point %d", t)
            return [], {}

        masks = graph_filter.node_attrs(attr_keys=[self._mask_key])[self._mask_key].to_list()

        frame = np.asarray(intensity_image[t]) if intensity_image is not None else None

        results: dict[str, list[Any]] = {}
        for mask in masks:
            if not isinstance(mask, Mask):
                raise TypeError(
                    f"Expected `Mask` object in '{self._mask_key}' attribute, got '{type(mask)}'. "
                    "Use `mask_key` to select the attribute holding the masks."
                )

            regionprops_kwargs: dict[str, Any] = {"spacing": self._spacing}
            if frame is not None:
                regionprops_kwargs["intensity_image"] = mask.crop(frame)

            obj = mask.regionprops(**regionprops_kwargs)

            for key, value in _region_property_attrs(obj, self._extra_properties, self._channel_names).items():
                results.setdefault(key, []).append(value)

            obj._cache.clear()  # clearing to reduce memory footprint

        return node_ids, results
