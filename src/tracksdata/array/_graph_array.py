from collections.abc import Sequence
from copy import copy
from typing import TYPE_CHECKING, Any

import numpy as np

from tracksdata.array._base_array import ArrayIndex, BaseReadOnlyArray
from tracksdata.array._nd_chunk_cache import NDChunkCache
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.options import get_options
from tracksdata.utils._dtypes import polars_dtype_to_numpy_dtype

if TYPE_CHECKING:
    from tracksdata.nodes._mask import Mask


def _validate_shape(
    shape: tuple[int, ...] | None,
    graph: BaseGraph,
    func_name: str,
) -> tuple[int, ...]:
    """Helper function to validate the shape argument."""
    if shape is None:
        try:
            shape = graph.metadata()["shape"]
        except KeyError as e:
            raise KeyError(
                f"`shape` is required to `{func_name}`. "
                "Please provide a `shape` argument or set the `shape` in the graph metadata."
            ) from e
    return shape


def chain_indices(slicing1: ArrayIndex | None, slicing2: ArrayIndex | None) -> ArrayIndex:
    """Chain two array indexing operations into a single one.

    Parameters
    ----------
    slicing1 : ArrayIndex | None
        The first array indexing operation.
    slicing2 : ArrayIndex | None
        The second array indexing operation.

    Returns
    -------
    ArrayIndex
        The chained array index.

    Examples
    --------
    ```python
    chain_indices(slice(3, 20), slice(5, 15))
    slice(8, 18, None)
    chain_indices(slice(3, 20), slice(5, None))
    slice(8, 20, None)
    chain_indices(slice(3, 20), slice(None, 15))
    slice(3, 18, None)
    chain_indices(slice(3, 20), 4)
    7
    chain_indices(slice(3, 20), (4, 5))
    [7, 8]
    chain_indices((5, 6, 7, 8, 9, 10), (3, 5))
    [8, 10]
    ```
    """
    if slicing2 is None:
        return slicing1

    if isinstance(slicing1, slice):
        new_slicing = range(max(slicing1.start, slicing1.stop))[slicing1]
        if isinstance(slicing2, Sequence):
            return [new_slicing[i] for i in slicing2]
        else:
            new_slicing = new_slicing[slicing2]
            if isinstance(new_slicing, range):
                return slice(new_slicing.start, new_slicing.stop, new_slicing.step)
            else:
                return new_slicing

    elif isinstance(slicing1, Sequence):
        if isinstance(slicing2, Sequence):
            return [slicing1[i] for i in slicing2]
        else:
            return slicing1[slicing2]

    raise ValueError(
        f"Cannot merge indices {slicing1} and {slicing2}. slicing1 must be a slice or python check indexable."
    )


def _get_size(ind: ArrayIndex, size: int) -> int | None:
    """
    Get final size of an array after applying the indexing operation.

    Parameters
    ----------
    ind : ArrayIndex
        The indexing operation.
    size : int
        The size of the array before applying the indexing operation.

    Returns
    -------
    int | None
        The final size of the array after applying the indexing operation.
    """
    if isinstance(ind, slice):
        return len(range(ind.start or 0, ind.stop or size, ind.step or 1))
    elif isinstance(ind, Sequence):
        return len(ind)
    elif np.isscalar(ind):
        return None
    else:
        raise ValueError(f"Expected scalar, sequence or slice, got type '{type(ind)}' with value {ind}")


class GraphArrayView(BaseReadOnlyArray):
    """
    Class used to view the content of a graph as an array.

    The resulting graph behaves as a read-only numpy array,
    displaying arbitrary attributes inside their respective instance mask.

    The content is lazy loaded from the original data source as
    it's done with a [zarr.Array](https://zarr.readthedocs.io/en/stable/index.html)

    Parameters
    ----------
    graph : BaseGraph
        The graph to view as an array.
    attr_key : str
        The attribute key to view as an array.
    offset : int | np.ndarray, optional
        The offset to apply to the array.
    shape : tuple[int, ...] | None, optional
        The shape of the array. If None, the shape is inferred from the graph metadata `shape` key.
    chunk_shape : tuple[int] | None, optional
        The chunk shape for the array. If None, the default chunk size is used.
    buffer_cache_size : int, optional
        The maximum number of buffers to keep in the cache for the array.
        If None, the default buffer cache size is used.
    """

    def __init__(
        self,
        graph: BaseGraph,
        attr_key: str,
        *,
        offset: int | np.ndarray = 0,
        shape: tuple[int, ...] | None = None,
        chunk_shape: tuple[int, ...] | int | None = None,
        buffer_cache_size: int | None = None,
        dtype: np.dtype | None = None,
    ):
        if attr_key not in graph.node_attr_keys(return_ids=True):
            raise ValueError(f"Attribute key '{attr_key}' not found in graph. Expected '{graph.node_attr_keys()}'")

        self.graph = graph
        self._attr_key = attr_key
        self._offset = offset

        if dtype is None:
            # Infer the dtype from the graph's attribute
            # TODO improve performance
            df = graph.node_attrs(attr_keys=[self._attr_key])
            if df.is_empty():
                dtype = get_options().gav_default_dtype
            else:
                try:
                    dtype = polars_dtype_to_numpy_dtype(df[self._attr_key].dtype, allow_sequence=False)
                except ValueError as e:
                    raise ValueError(f"Attribute values for key '{self._attr_key}' must be scalar.") from e
                # napari support for bool is limited
                if np.issubdtype(dtype, bool):
                    dtype = np.uint8

        self._dtype = dtype
        self.original_shape = _validate_shape(shape, graph, "GraphArrayView")

        chunk_shape = chunk_shape or get_options().gav_chunk_shape
        if isinstance(chunk_shape, int):
            chunk_shape = (chunk_shape,) * (len(self.original_shape) - 1)
        elif len(chunk_shape) < len(self.original_shape) - 1:
            chunk_shape = (1,) * (len(self.original_shape) - 1 - len(chunk_shape)) + tuple(chunk_shape)

        self.chunk_shape = chunk_shape
        self.buffer_cache_size = buffer_cache_size or get_options().gav_buffer_cache_size

        self._indices = tuple(slice(0, s) for s in self.original_shape)
        self._cache = NDChunkCache(
            compute_func=self._fill_array,
            shape=self.shape[1:],
            chunk_shape=self.chunk_shape,
            buffer_cache_size=self.buffer_cache_size,
            dtype=self.dtype,
        )

        self._spatial_filter = self.graph.bbox_spatial_filter(
            frame_attr_key=DEFAULT_ATTR_KEYS.T,
            bbox_attr_key=DEFAULT_ATTR_KEYS.BBOX,
        )
        self.graph.node_added.connect(self._on_node_added)
        self.graph.node_removed.connect(self._on_node_removed)
        self.graph.node_attrs_updated.connect(self._on_node_attrs_updated)

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the array."""

        shape = [_get_size(ind, os) for ind, os in zip(self._indices, self.original_shape, strict=True)]
        return tuple(s for s in shape if s is not None)

    @property
    def size(self) -> int:
        """Returns the total number of elements in the array."""
        return int(np.prod(self.shape))

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the array."""
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype:
        """Returns the dtype of the array."""
        return np.dtype(self._dtype)

    def __getitem__(self, index: ArrayIndex) -> "GraphArrayView":
        """Return a sliced view of the GraphArrayView.

        Parameters
        ----------
        index : ArrayIndex
            The indices to slice the array.

        Returns
        -------
        GraphArrayView
            A new GraphArrayView object with updated indices.
        """
        normalized_index = []
        if not isinstance(index, tuple):
            index = (index,)
        if None in index:
            raise ValueError("None is not allowed for GraphArrayView indexing.")
        jj = 0
        for oi in self._indices:
            if np.isscalar(oi):
                normalized_index.append(None)
            else:
                if len(index) <= jj:
                    normalized_index.append(slice(None))
                else:
                    normalized_index.append(index[jj])
                jj += 1

        return self.reindex(normalized_index)

    def reindex(
        self,
        slicing: Sequence[ArrayIndex],
    ) -> "GraphArrayView":
        """
        Reindex the GraphArrayView.
        Returns a shallow copy of the GraphArrayView with the new indices.

        Parameters
        ----------
        slicing : tuple[ArrayIndex, ...]
            The new indices to apply to the GraphArrayView.

        Returns
        -------
        GraphArrayView
            A new GraphArrayView object with updated indices.
        """
        obj = copy(self)
        obj._indices = tuple(chain_indices(i1, i2) for i1, i2 in zip(self._indices, slicing, strict=False))
        return obj

    def __array__(
        self,
        dtype: np.dtype | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        """Convert the GraphArrayView to a numpy array.

        Parameters
        ----------
        dtype : np.dtype, optional
            The desired dtype of the output array. If None, the dtype of the GraphArrayView is used.
        copy : bool, optional
            This parameter is ignored, as the GraphArrayView is read-only.

        Returns
        -------
        np.ndarray
            In memory numpy array of the GraphArrayView of the current indices.
        """

        if sum(isinstance(i, Sequence) for i in self._indices) > 1:
            raise NotImplementedError("Multiple sequences in indices are not supported for __array__.")

        time = self._indices[0]
        volume_slicing = self._indices[1:]

        if np.isscalar(time):
            try:
                time = time.item()  # convert from numpy.int to int
            except AttributeError:
                pass
            result = self._cache.get(
                time=time,
                volume_slicing=volume_slicing,
            ).astype(dtype or self.dtype)
            return np.array(result) if np.isscalar(result) else result
        else:
            if isinstance(time, slice):
                time = range(self.original_shape[0])[time]

            return np.stack(
                [
                    self._cache.get(
                        time=t,
                        volume_slicing=volume_slicing,
                    )
                    for t in time
                ]
            ).astype(dtype or self.dtype)

    def _fill_array(self, time: int, volume_slicing: Sequence[slice], buffer: np.ndarray) -> np.ndarray:
        """Fill the buffer with data from the graph at a specific time.

        Parameters
        ----------
        time : int
            The time point to retrieve data for.
        volume_slicing : Sequence[slice]
            The volume slicing information (currently not fully utilized).
        buffer : np.ndarray
            The buffer to fill with data.

        Returns
        -------
        np.ndarray
            The filled buffer.
        """
        subgraph = self._spatial_filter[(slice(time, time), *volume_slicing)]
        df = subgraph.node_attrs(
            attr_keys=[self._attr_key, DEFAULT_ATTR_KEYS.MASK],
        )

        for mask, value in zip(df[DEFAULT_ATTR_KEYS.MASK], df[self._attr_key], strict=True):
            mask: Mask
            mask.paint_buffer(buffer, value, offset=self._offset)

    def _on_node_added(self, node_id: int) -> None:
        self._invalidate_node_region_by_id(node_id)

    def _on_node_removed(self, node_id: int) -> None:
        self._invalidate_node_region_by_id(node_id)

    def _on_node_attrs_updated(self, node_ids: Sequence[int], attr_keys: Sequence[str]) -> None:
        changed_keys = set(attr_keys)
        if (
            self._attr_key not in changed_keys
            and DEFAULT_ATTR_KEYS.T not in changed_keys
            and DEFAULT_ATTR_KEYS.BBOX not in changed_keys
            and DEFAULT_ATTR_KEYS.MASK not in changed_keys
        ):
            return

        if DEFAULT_ATTR_KEYS.T in changed_keys:
            self._cache.clear()
            return

        if DEFAULT_ATTR_KEYS.BBOX in changed_keys or DEFAULT_ATTR_KEYS.MASK in changed_keys:
            for node_id in node_ids:
                attrs = self._node_attrs(node_id)
                if attrs is None:
                    continue
                time = attrs.get(DEFAULT_ATTR_KEYS.T)
                if time is None:
                    self._cache.clear()
                    return
                try:
                    self._cache.invalidate(time=int(time))
                except (TypeError, ValueError):
                    self._cache.clear()
                    return
            return

        for node_id in node_ids:
            self._invalidate_node_region_by_id(node_id)

    def _invalidate_node_region_by_id(self, node_id: int) -> None:
        attrs = self._node_attrs(node_id)
        if attrs is None:
            return
        self._invalidate_node_region(attrs)

    def _node_attrs(self, node_id: int) -> dict[str, Any] | None:
        try:
            return self.graph.nodes[node_id].to_dict()
        except (IndexError, KeyError, ValueError):
            return None

    def _invalidate_node_region(self, attrs: dict[str, Any]) -> None:
        time = attrs.get(DEFAULT_ATTR_KEYS.T)
        bbox = attrs.get(DEFAULT_ATTR_KEYS.BBOX)

        if time is None or bbox is None:
            self._cache.clear()
            return

        try:
            time_int = int(time)
        except (TypeError, ValueError):
            self._cache.clear()
            return

        if time_int < 0 or time_int >= self.original_shape[0]:
            return

        spatial_ndim = len(self.original_shape) - 1
        try:
            bbox_array = np.asarray(bbox, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            self._cache.invalidate(time=time_int)
            return

        if bbox_array.size != 2 * spatial_ndim:
            self._cache.invalidate(time=time_int)
            return

        volume_slicing = []
        for axis in range(spatial_ndim):
            start = int(np.floor(bbox_array[axis]))
            stop = int(np.ceil(bbox_array[axis + spatial_ndim]))
            axis_size = self.original_shape[axis + 1]
            start = max(0, min(axis_size, start))
            stop = max(0, min(axis_size, stop))
            if stop <= start:
                return
            volume_slicing.append(slice(start, stop))

        self._cache.invalidate(time=time_int, volume_slicing=tuple(volume_slicing))
