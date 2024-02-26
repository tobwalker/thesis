import logging
import polars as pl
import pathlib

from config import columns

logger = logging.getLogger("__name__")


def retrieve_data(base_dir: str | pathlib.Path, filename_backlog: str, filename_parcel_amount: str):
    filepath_backlog = pathlib.Path(base_dir) / (filename_backlog + ".pq")
    filepath_parcel_amount = pathlib.Path(base_dir) / (filename_parcel_amount + ".pq")

    backlog = pl.read_parquet(filepath_backlog).rename({"BACKLOG": columns.BACKLOG})
    parcel_amount = pl.read_parquet(filepath_parcel_amount).rename({"PARCEL_AMOUNT": columns.PARCEL_AMOUNT})

    return (
        backlog.join(parcel_amount, on=[columns.OZ_ZSP, columns.DATE], how="outer")
        .fill_null(0)
        .with_columns(
            pl.col(columns.DATE).str.slice(0, 10).str.strptime(pl.Datetime, "%Y-%m-%d").alias(columns.DATE),
            pl.col(columns.OZ_ZSP).str.slice(0, 12).alias(columns.OZ_ZSP),
            pl.col(columns.BACKLOG).cast(pl.Int32).alias(columns.BACKLOG),
            pl.col(columns.PARCEL_AMOUNT).cast(pl.Int32).alias(columns.PARCEL_AMOUNT),
        )
        .with_columns(
            (pl.col(columns.DATE) - pl.duration(days=pl.col(columns.DATE).dt.weekday() - 1)).alias(columns.DATE)
        )
        .group_by(columns.DATE, columns.OZ_ZSP)
        .agg(pl.col(columns.BACKLOG).mean(), pl.col(columns.PARCEL_AMOUNT).mean())
        .drop_nulls()
    )
