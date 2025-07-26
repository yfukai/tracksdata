import itertools
from typing import Sequence, Tuple, Dict, Callable

import numpy as np

import itertools
from collections import OrderedDict
from typing import Callable, Dict, Sequence, Tuple

import numpy as np


class NDChunkCache:
    """
    N-dimensional full-volume cache with LRU eviction on the *time* axis.

    Parameters
    ----------
    compute_func : Callable[[int, Tuple[slice, ...]], np.ndarray]
        User-supplied function returning one *chunk* given (t, chunk_slices).
    full_shape : Sequence[int]
        The absolute shape of a single time-point volume.
    chunk_size : Sequence[int]
        Edge length of a chunk in every axis.  Length defines dimensionality.
    max_buffers : int
        Maximum number of time-points kept simultaneously.
    dtype : np.dtype, optional
        Stored dtype of buffers (default float32).
    """

    # ------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------
    def __init__(
        self,
        compute_func: Callable[[int, tuple[slice, ...]], np.ndarray],
        full_shape: Sequence[int],
        chunk_size: Sequence[int],
        max_buffers: int = 4,
        dtype=np.float32,
    ):
        self.compute_func = compute_func
        self.full_shape: Tuple[int, ...] = tuple(full_shape)
        self.chunk_size: Tuple[int, ...] = tuple(chunk_size)
        self.ndim: int = len(self.full_shape)
        if len(self.full_shape) != len(self.chunk_size):
            raise ValueError("`full_shape` and `chunk_size` must have same length")

        # Grid size in chunks, e.g. (nz, ny, nx, …)
        # Use ceiling division to handle non-divisible shapes
        self.grid_shape: Tuple[int, ...] = tuple((fs + cs - 1) // cs for fs, cs in zip(self.full_shape, self.chunk_size))

        self.max_buffers = max_buffers
        self.dtype = dtype

        # (LRU) mapping   t  ->  {"buffer": ndarray, "ready": boolean ndarray}
        self._store: "OrderedDict[int, Dict[str, np.ndarray]]" = OrderedDict()

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------
    def _ensure_buffer(self, t: int) -> Dict[str, np.ndarray]:
        """
        Return the dictionary holding buffer & ready-mask for time `t`,
        allocating (and evicting) if necessary.
        """
        if t in self._store:
            # Move to MRU position
            self._store.move_to_end(t)
            return self._store[t]

        # Evict oldest if over limit
        if len(self._store) >= self.max_buffers:
            self._store.popitem(last=False)

        # Allocate new buffer and boolean mask
        buf = np.empty(self.full_shape, dtype=self.dtype)
        ready = np.zeros(self.grid_shape, dtype=bool)
        self._store[t] = {"buffer": buf, "ready": ready}
        return self._store[t]

    def _chunk_bounds(self, slices: Tuple[slice, ...]) -> Tuple[Tuple[int, int], ...]:
        """Return inclusive chunk-index bounds for every axis."""
        return tuple(
            (s.start // cs, (s.stop - 1) // cs) for s, cs in zip(slices, self.chunk_size)
        )

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def get(self, t: int, volume_slices: Tuple[slice, ...]) -> np.ndarray:
        """
        Retrieve data for time `t` and arbitrary dimensional slices.

        Returns a *view* into the cached full buffer (zero-copy).
        """
        if len(volume_slices) != self.ndim:
            raise ValueError("Number of slices must equal dimensionality")

        store_entry = self._ensure_buffer(t)
        buf = store_entry["buffer"]
        ready = store_entry["ready"]

        # Which chunks are touched by this request?
        bounds = self._chunk_bounds(volume_slices)
        chunk_ranges = [range(lo, hi + 1) for lo, hi in bounds]

        # For every intersected chunk, compute if not ready
        for chunk_idx in itertools.product(*chunk_ranges):
            if ready[chunk_idx]:
                continue  # already filled

            # Absolute slice covering this chunk
            chunk_slc = tuple(
                slice(ci * cs, min((ci + 1) * cs, fs)) 
                for ci, cs, fs in zip(chunk_idx, self.chunk_size, self.full_shape)
            )
            # Handle the case where chunk_slc exceeds volume_slices
            buf[chunk_slc] = self.compute_func(t, chunk_slc).astype(self.dtype, copy=False)
            ready[chunk_idx] = True

        # Return view on the big buffer
        return buf[volume_slices]


