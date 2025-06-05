import numpy as np
from numpy.typing import ArrayLike

from tracksdata.array._base_array import ArrayIndex, BaseReadOnlyArray
from tracksdata.constants import DEFAULT_ATTR_KEYS
from tracksdata.graph._base_graph import BaseGraph
from tracksdata.nodes._mask import Mask
from tracksdata.utils._convert import polars_dtype_to_numpy_dtype


class GraphArrayView(BaseReadOnlyArray):
    def __init__(
        self,
        graph: BaseGraph,
        shape: tuple[int, ...],
        feature_key: str,
        offset: int | np.ndarray = 0,
    ):
        if feature_key not in graph.node_features_keys:
            raise ValueError(f"Feature key '{feature_key}' not found in graph. Expected '{graph.node_features_keys}'")

        self.graph = graph
        self._shape = shape
        self._feature_key = feature_key
        self._offset = offset
        self._dtype = np.int32

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __getitem__(self, index: ArrayIndex) -> ArrayLike:
        # FIXME:
        # - just for testing, not final implementation
        # - make use of offset
        if isinstance(index, tuple):
            index = index[0]

        if isinstance(index, int):
            node_ids = self.graph.filter_nodes_by_attribute({DEFAULT_ATTR_KEYS.T: index})

            if len(node_ids) == 0:
                return np.zeros(self.shape[1:], dtype=self.dtype)

            # TODO: this should be a single `subgraph(t=index).node_features(...)` call
            df = self.graph.node_features(
                node_ids=node_ids,
                feature_keys=[self._feature_key, DEFAULT_ATTR_KEYS.MASK],
            )

            dtype = polars_dtype_to_numpy_dtype(df[self._feature_key].dtype)

            # napari support for bool is limited
            if np.issubdtype(dtype, bool):
                dtype = np.uint8

            self._dtype = dtype

            # TODO: reuse buffer
            buffer = np.zeros(self.shape[1:], dtype=self.dtype)

            for mask, value in zip(df[DEFAULT_ATTR_KEYS.MASK], df[self._feature_key], strict=False):
                mask: Mask
                mask.paint_buffer(buffer, value, offset=self._offset)

            return buffer
        else:
            raise NotImplementedError("Not implemented")
