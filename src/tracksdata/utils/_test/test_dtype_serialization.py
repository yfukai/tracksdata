import base64
import binascii

import polars as pl
import pytest

from tracksdata.utils._dtypes import deserialize_polars_dtype, serialize_polars_dtype


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Int64,
        pl.Float32,
        pl.Boolean,
        pl.String,
        pl.List(pl.Int16),
        pl.Array(pl.Float64, 4),
        pl.Array(pl.Int32, (2, 3)),
        pl.Struct({"x": pl.Int64, "y": pl.List(pl.String)}),
        pl.Datetime("us", "UTC"),
    ],
)
def test_serialize_deserialize_polars_dtype_roundtrip(dtype: pl.DataType) -> None:
    encoded = serialize_polars_dtype(dtype)

    assert isinstance(encoded, str)
    assert encoded
    assert base64.b64decode(encoded)

    restored_dtype = deserialize_polars_dtype(encoded)

    assert restored_dtype == dtype


def test_deserialize_polars_dtype_invalid_base64_raises() -> None:
    with pytest.raises(binascii.Error):
        deserialize_polars_dtype("not-base64")


def test_deserialize_polars_dtype_non_ipc_payload_raises() -> None:
    encoded = base64.b64encode(b"not-arrow-ipc").decode("utf-8")

    with pytest.raises((OSError, pl.exceptions.PolarsError)):
        deserialize_polars_dtype(encoded)
