import base64
import binascii

import numpy as np
import polars as pl
import pytest

from tracksdata.utils._dtypes import (
    AttrSchema,
    deserialize_attr_schema,
    deserialize_polars_dtype,
    serialize_attr_schema,
    serialize_polars_dtype,
)


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


@pytest.mark.parametrize(
    "schema",
    [
        AttrSchema(key="score", dtype=pl.Float64, default_value=1.25),
        AttrSchema(
            key="vector",
            dtype=pl.Array(pl.Float32, 3),
            default_value=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        ),
        AttrSchema(key="payload", dtype=pl.Object, default_value={"nested": [1, 2, 3]}),
    ],
)
def test_serialize_deserialize_attr_schema_roundtrip(schema: AttrSchema) -> None:
    encoded = serialize_attr_schema(schema)
    restored = deserialize_attr_schema(encoded, key=schema.key)
    assert restored == schema
