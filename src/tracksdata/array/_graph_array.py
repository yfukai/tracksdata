import numpy as np
from numpy.typing import ArrayLike
from polars.datatypes.convert import dtype_to_ffiname

from tracksdata.array._base_array import ArrayIndex, BaseReadOnlyArray
from tracksdata.graph._base_graph import BaseGraphBackend
from tracksdata.nodes._mask import Mask


class GraphArrayView(BaseReadOnlyArray):
    def __init__(
        self,
        graph: BaseGraphBackend,
        shape: tuple[int, ...],
        feature_key: str,
        offset: int | np.ndarray = 0,
    ):
        if feature_key not in graph.node_features_keys:
            raise ValueError(
                f"Feature key '{feature_key}' not found in graph. "
                f"Expected '{graph.node_features_keys}'"
            )

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
        # FIXME: just for testing
        if isinstance(index, tuple):
            index = index[0]

        if isinstance(index, int):
            node_ids = self.graph.filter_nodes_by_attribute(t=index)

            if len(node_ids) == 0:
                return np.zeros(self.shape[1:], dtype=self.dtype)

            # TODO: this should be a single `subgraph(t=index).node_features(...)` call
            df = self.graph.node_features(
                node_ids=node_ids,
                feature_keys=[self._feature_key, "mask"],
            )

            dtype = np.dtype(dtype_to_ffiname(df[self._feature_key].dtype))
            if np.issubdtype(dtype, bool):
                # napari expects uint8 labels
                dtype = np.uint8

            self._dtype = dtype

            # TODO: reuse buffer
            buffer = np.zeros(self.shape[1:], dtype=self.dtype)

            for mask, value in zip(df["mask"], df[self._feature_key], strict=False):
                mask: Mask
                mask.paint_buffer(buffer, value)

            return buffer
        else:
            raise NotImplementedError("Not implemented")
