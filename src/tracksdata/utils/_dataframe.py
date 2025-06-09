import polars as pl


def unpack_array_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Unpack array features into a dictionary, convert array columns into multiple scalar columns.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with array features.

    Returns
    -------
    pl.DataFrame
        DataFrame with unpacked array features.
    """

    array_cols = [name for name, dtype in df.schema.items() if isinstance(dtype, pl.Array)]

    if len(array_cols) == 0:
        return df

    for col in array_cols:
        df = df.with_columns(pl.col(col).arr.to_struct(lambda x: f"{col}_{x}")).unnest(col)  # noqa: B023

    return unpack_array_features(df)
