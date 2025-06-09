import numpy as np
from polars.datatypes.classes import (
    Boolean,
    DataType,
    Datetime,
    Duration,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
)

_POLARS_DTYPE_TO_NUMPY_DTYPE = {
    Datetime: np.datetime64,
    Boolean: np.bool_,
    Float32: np.float32,
    Float64: np.float64,
    Int8: np.int8,
    Int16: np.int16,
    Int32: np.int32,
    Int64: np.int64,
    Duration: np.timedelta64,
    UInt8: np.uint8,
    UInt16: np.uint16,
    UInt32: np.uint32,
    UInt64: np.uint64,
}


def polars_dtype_to_numpy_dtype(polars_dtype: DataType) -> np.dtype:
    """Convert a polars dtype to a numpy dtype.

    Parameters
    ----------
    polars_dtype : DataType
        The polars dtype to convert.

    Returns
    -------
    np.dtype
        The numpy dtype.
    """
    try:
        return _POLARS_DTYPE_TO_NUMPY_DTYPE[polars_dtype]
    except KeyError as e:
        raise ValueError(
            f"Invalid polars dtype: {polars_dtype}. Expected one of {_POLARS_DTYPE_TO_NUMPY_DTYPE.keys()}"
        ) from e
