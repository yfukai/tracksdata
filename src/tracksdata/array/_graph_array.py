import numpy as np
from numpy.typing import ArrayLike

from tracksdata.array._base_array import ArrayIndex, BaseReadOnlyArray
from tracksdata.attrs import NodeAttr
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.nodes._mask import Mask
from tracksdata.utils._dtypes import polars_dtype_to_numpy_dtype
from functools import lru_cache
from cachetools import LRUCache

import numpy as np
from typing import Tuple, Dict






# # -------------------------------------------------------------------------
# # Example usage (works for any dimensionality)
# # -------------------------------------------------------------------------
# if __name__ == "__main__":
# 
#     def expensive_compute(t: int, chunk_slc: Tuple[slice, ...]) -> np.ndarray:
#         """Dummy expensive function – replace with real work/IO."""
#         print(f"compute(t={t}, slices={chunk_slc})")
#         shape = tuple(s.stop - s.start for s in chunk_slc)
#         rng = np.random.default_rng(hash((t, chunk_slc)) & 0xFFFF_FFFF)
#         return rng.random(shape, dtype=np.float32)
# 
#     # 4‑D spatial data (e.g. Z, Y, X, C)
#     cache = NDChunkCache(
#         compute_func=expensive_compute,
#         chunk_size=(32, 32, 32, 4),
#         max_cache_size=16,
#     )
# 
#     # Request a slice spanning multiple chunks in every axis
#     data = cache.get(
#         t=0,
#         volume_slices=(
#             slice(10, 90),   # Z
#             slice(20, 100),  # Y
#             slice(5, 130),   # X
#             slice(0, 4),     # C
#         ),
#     )
#     print("result shape:", data.shape)


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
        lru_cache_size: int = 128,
        buffer_size: int = 128,
    ):
        if attr_key not in graph.node_attr_keys:
            raise ValueError(f"Attribute key '{attr_key}' not found in graph. Expected '{graph.node_attr_keys}'")

        self.graph = graph
        self._shape = shape
        self._attr_key = attr_key
        self._offset = offset
        self._dtype = np.int32
        self._cached_fill_array = lru_cache(maxsize=lru_cache_size)(
            self._cached_fill_array
        )
        self._buffer = LRUCache(maxsize=buffer_size)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __getitem__(self, index: ArrayIndex) -> ArrayLike:
        if isinstance(index, tuple):
            time, volume_slicing = index[0], index[1:]
        else:  # if only 1 (time) is provided
            time = index
            volume_slicing = tuple()

        if isinstance(time, slice):  # if all time points are requested
            # XXX could be dask? should be benchmarked
            return np.stack(
                [
                    self.__getitem__((t,) + volume_slicing).copy()
                    for t in range(*time.indices(self.shape[0]))
                ]
            )
        else:
            try:
                time = time.item()  # convert from numpy.int to int
            except AttributeError:
                time = time

        buffer = self._cached_fill_array(
            time=time,
            volume_slicing=tuple(volume_slicing),
        )

        return buffer[volume_slicing]

    def _cached_fill_array(self, time: int, volume_slicing: ArrayIndex) -> np.ndarray:
        # Executing here already means that the time and slices are 
        # not cached, so we compute the buffer

        # Reuse the buffer if it exists for that time, otherwise create a new one
        if time not in self._buffer:
            self._buffer[time] = np.zeros(self.shape[1:], dtype=self.dtype)
        buffer = self._buffer[time]

        # TODO handling the slices for volume_slicing
        graph_filter = self.graph.filter(NodeAttr(DEFAULT_ATTR_KEYS.T) == time)
        df = graph_filter.node_attrs(
            attr_keys=[self._attr_key, DEFAULT_ATTR_KEYS.MASK],
        )

        dtype = polars_dtype_to_numpy_dtype(df[self._attr_key].dtype)

        # napari support for bool is limited
        if np.issubdtype(dtype, bool):
            dtype = np.uint8

        self._dtype = dtype

        for mask, value in zip(df[DEFAULT_ATTR_KEYS.MASK], df[self._attr_key], strict=False):
            mask: Mask
            mask.paint_buffer(buffer, value, offset=self._offset)

        return buffer[volume_slicing]
