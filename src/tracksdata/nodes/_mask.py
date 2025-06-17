from functools import cached_property

import blosc2
import numpy as np
from numpy.typing import NDArray

from tracksdata.functional._iou import fast_intersection_with_bbox, fast_iou_with_bbox


class Mask:
    def __init__(
        self,
        mask: NDArray[np.bool_],
        bbox: np.ndarray,
    ):
        bbox = np.asarray(bbox)

        if mask.ndim != bbox.shape[0] // 2:
            raise ValueError(f"Mask dimension {mask.ndim} does not match bbox dimension {bbox.shape[0]} // 2")

        bbox_size = bbox[mask.ndim :] - bbox[: mask.ndim]

        if np.any(mask.shape != bbox_size):
            raise ValueError(f"Mask shape {mask.shape} does not match bbox size {bbox_size}")

        self._mask = mask
        self._bbox = bbox

    def __getstate__(self) -> dict:
        data_dict = self.__dict__.copy()
        data_dict["_mask"] = blosc2.pack_array(self._mask)
        return data_dict

    def __setstate__(self, state: dict) -> None:
        state["_mask"] = blosc2.unpack_array(state["_mask"])
        self.__dict__.update(state)

    def mask_indices(
        self,
        offset: NDArray[np.integer] | int = 0,
    ) -> tuple[NDArray[np.integer], ...]:
        if isinstance(offset, int):
            offset = np.full(self._mask.ndim, offset)

        indices = list(np.nonzero(self._mask))

        for i, index in enumerate(indices):
            indices[i] = index + self._bbox[i] + offset[i]

        return tuple(indices)

    def paint_buffer(
        self,
        buffer: np.ndarray,
        value: int | float,
        offset: NDArray[np.integer] | int = 0,
    ) -> None:
        # TODO: make it zarr and tensorstore compatible
        indices = self.mask_indices(offset)
        buffer[indices] = value

    def iou(self, other: "Mask") -> float:
        return fast_iou_with_bbox(self._bbox, other._bbox, self._mask, other._mask)

    def intersection(self, other: "Mask") -> float:
        return fast_intersection_with_bbox(self._bbox, other._bbox, self._mask, other._mask)

    @cached_property
    def size(self) -> int:
        return self._mask.sum()

    def __repr__(self) -> str:
        slicing_str = ", ".join(
            f"{i}:{j}"
            for i, j in zip(
                self._bbox[: self._mask.ndim],
                self._bbox[self._mask.ndim :],
                strict=True,
            )
        )
        return f"Mask(bbox=[{slicing_str}])"
