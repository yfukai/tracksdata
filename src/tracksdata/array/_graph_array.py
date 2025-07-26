import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Sequence
from copy import copy

from tracksdata.array._base_array import ArrayIndex, BaseReadOnlyArray
from tracksdata.attrs import NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.nodes._mask import Mask
from tracksdata.utils._dtypes import polars_dtype_to_numpy_dtype
from tracksdata.array._nd_chunk_cache import NDChunkCache

import numpy as np

DEFAULT_CHUNK_SIZE = 128
DEFAULT_DTYPE = np.int32

def merge_indices(slicing1: ArrayIndex | None, slicing2: ArrayIndex | None) -> slice:
    """Merge two array indices into a single slice.
    Example:
        merge_indices(slice(3, 20), slice(5, 15)) == slice(8, 18)
        merge_indices(slice(3, 20), slice(5, None)) == slice(8, 20)
        merge_indices(slice(3, 20), slice(None, 15)) == slice(3, 18)
        merge_indices(slice(3, 20), 4) == 7
        merge_indices(slice(3, 20), (4, 5)) == (7, 8)
        merge_indices((5,6,7,8,9,10),(3,5)) == (8, 10)
    """
    if slicing2 is None:
        return slicing1
    if isinstance(slicing1, slice):
        r = range(max(slicing1.start, slicing1.stop))[slicing1]
        if isinstance(slicing2, Sequence):
            return [r[i] for i in slicing2]
        else:
            r = r[slicing2]
            if isinstance(r, range):
                return slice(r.start, r.stop, r.step)
            else:
                return r
    elif isinstance(slicing1, Sequence):
        return [slicing1[i] for i in slicing2]
    raise ValueError(f"Cannot merge indices {slicing1} and {slicing2}. slicing1 must be a slice or python check indexable.")

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
    shape : tuple[int, ...]
        The shape of the array.
    attr_key : str
        The attribute key to view as an array.
    offset : int | np.ndarray, optional
        The offset to apply to the array.
    """

    def __init__(
        self,
        graph: BaseGraph,
        shape: tuple[int, ...],
        attr_key: str,
        offset: int | np.ndarray = 0,
        chunk_shape: tuple[int] | None = None,
        max_buffers: int = 32,
    ):
        if attr_key not in graph.node_attr_keys:
            raise ValueError(f"Attribute key '{attr_key}' not found in graph. Expected '{graph.node_attr_keys}'")

        self.graph = graph
        self._attr_key = attr_key
        self._offset = offset
        # Infer the dtype from the graph's attribute
        # TODO improve performance
        df = graph.node_attrs(attr_keys=[self._attr_key])
        if df.is_empty():
            dtype = DEFAULT_DTYPE
        else:
            dtype = polars_dtype_to_numpy_dtype(df[self._attr_key].dtype)
            # napari support for bool is limited
            if np.issubdtype(dtype, bool):
                dtype = np.uint8

        self._dtype = dtype

        self.original_shape = shape
        if chunk_shape is None:
            chunk_shape = tuple([DEFAULT_CHUNK_SIZE]*(len(shape) - 1))  # Default chunk shape
        self.max_buffers = max_buffers
        self.cache = NDChunkCache(
            compute_func=self._fill_array,
            shape=shape[1:],
            chunk_shape=chunk_shape,
            max_buffers=max_buffers,
            dtype=np.dtype(self._dtype),
        )
        self._indices = tuple(slice(0, s) for s in shape)

    @property
    def shape(self) -> tuple[int, ...]:
        def _get_size(ind: ArrayIndex, size: int) -> int | None:
            if isinstance(ind, slice):
                return len(range(ind.start or 0, ind.stop or size))
            elif isinstance(ind, Sequence):
                return len(ind)
            else:
                assert np.isscalar(ind), f"Expected scalar or slice, got {type(ind)}"
                return None

        shape = [_get_size(ind, os) for ind, os in zip(self._indices, self.original_shape)]
        return tuple(s for s in shape if s is not None)

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the array."""
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __getitem__(self, index: ArrayIndex) -> "GraphArrayView":
        """Get the sliced view of the array.

        Args:
            index (ArrayIndex): _description_

        Returns:
            ArrayLike: _description_
        """
        obj = copy(self)
        normalized_index = []
        if not isinstance(index, tuple):
            index = (index,)
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

        obj._indices = tuple(merge_indices(i1, i2) for i1, i2 in zip(self._indices, normalized_index))
        return obj

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        time = self._indices[0]
        volume_slicing = self._indices[1:]
        if isinstance(time, Sequence):  # if all time points are requested
            # XXX could be dask? should be benchmarked
            return np.stack(
                [
                    self.__getitem__((t,) + volume_slicing).__array__()
                    for t in range(*time.indices(self.original_shape[0]))
                ]
            )
        else:
            try:
                time = time.item()  # convert from numpy.int to int
            except AttributeError:
                time = time
    
        return self.cache.get(
            time=time,
            volume_slicing=volume_slicing,
        )

    def _fill_array(self, time: int, volume_slicing: ArrayIndex, buffer: np.ndarray) -> np.ndarray:
        # TODO handling the slices for volume_slicing
        graph_filter = self.graph.filter(NodeAttr(DEFAULT_ATTR_KEYS.T) == time)
        df = graph_filter.node_attrs(
            attr_keys=[self._attr_key, DEFAULT_ATTR_KEYS.MASK],
        )

        for mask, value in zip(df[DEFAULT_ATTR_KEYS.MASK], df[self._attr_key], strict=False):
            mask: Mask
            mask.paint_buffer(buffer, value, offset=self._offset)
