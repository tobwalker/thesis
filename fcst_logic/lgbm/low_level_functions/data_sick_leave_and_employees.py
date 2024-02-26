import logging
import polars as pl
import pathlib
import datetime
from config import columns

logger = logging.getLogger("__name__")

# MISSING FEATURES: eduction, combiantion out of age and retiree plan, entry date


def import_sick_leave_data(
    base_dir: str,
    filename: str,
    target_column: str,
    start_date: datetime.datetime | None = None,
    end_date: datetime.datetime | None = None,
) -> pl.DataFrame:
    filepath = pathlib.Path(base_dir) / (filename + ".pq")
    sick_leave = (
        pl.read_parquet(filepath)
        .filter(pl.col(columns.OZ_ZSP).str.len_chars() == 12)
        .rename(
            {
                "DATE": columns.DATE,
                "OZ_ZSP": columns.OZ_ZSP,
            }
        )
        .sort(by=columns.DATE)
        .upsample(time_column=columns.DATE, every="1d", by=columns.OZ_ZSP)
        .with_columns(
            pl.col(columns.OZ_ZSP).fill_null(strategy="forward"),
            pl.col(target_column).fill_null(strategy="zero"),
        )
        .with_columns(
            pl.datetime(
                pl.col(columns.DATE).dt.year().cast(pl.Int64), pl.col(columns.DATE).dt.month().cast(pl.Int64), 1
            ).alias("YEAR_MONTH_1"),
        )
    )
    if start_date is not None:
        sick_leave = sick_leave.filter(pl.col(columns.DATE) >= start_date)
    if end_date is not None:
        sick_leave = sick_leave.filter(pl.col(columns.DATE) <= end_date)
    return sick_leave


def import_base_data(base_dir: str, filename: str) -> pl.DataFrame:
    filepath = pathlib.Path(base_dir) / (filename + ".pq")
    return (
        pl.read_parquet(filepath)
        .rename(
            {
                "OZ_ZSP": columns.OZ_ZSP,
            }
        )
        .filter(pl.col(columns.OZ_ZSP).str.len_chars() == 12)
        .with_columns(
            pl.datetime(pl.col("YEAR"), pl.col("MONTH"), 1).alias("YEAR_MONTH_1"),
            pl.col("ENTRY_DATE").str.strptime(pl.Datetime, format="%Y%m%d", strict=False),
        )
        .drop("YEAR", "MONTH")
    )


def preprocess_only_current_stations(sick_leave: pl.DataFrame, base_data: pl.DataFrame) -> pl.DataFrame:
    current_stations = (
        base_data.filter(pl.col("YEAR_MONTH_1") == base_data.get_column("YEAR_MONTH_1").max())
        .get_column(columns.OZ_ZSP)
        .unique()
    )
    logger.info(f"Number of current stations in the base data: {len(current_stations)}")
    sick_leave = sick_leave.filter(pl.col(columns.OZ_ZSP).is_in(current_stations))
    logger.info(f"Number of current stations in the sick leave: {sick_leave.get_column(columns.OZ_ZSP).n_unique()}")
    return sick_leave


def preprocess_clean_histories(sick_leave: pl.DataFrame, base_data: pl.DataFrame) -> pl.DataFrame:
    grouped_base_data = base_data.group_by("OZ_ZSP", "YEAR_MONTH_1").agg(pl.lit(0).alias("MISSING_VALUE"))
    grouped_base_data = (
        grouped_base_data.sort(by="YEAR_MONTH_1")
        .upsample(time_column="YEAR_MONTH_1", every="1mo", by="OZ_ZSP")
        .with_columns(
            pl.col("OZ_ZSP").fill_null(strategy="backward"),
            pl.col("MISSING_VALUE").fill_null(strategy="one"),
        )
        .filter(pl.col("MISSING_VALUE") == 1)
        .group_by("OZ_ZSP")
        .agg(pl.col("YEAR_MONTH_1").max())
    )
    sick_leave = sick_leave.join_asof(
        grouped_base_data.with_columns(pl.lit(1).alias("HISTORY_ERROR")),
        on="YEAR_MONTH_1",
        by="OZ_ZSP",
        strategy="forward",
    )
    return sick_leave.filter(pl.col("HISTORY_ERROR").is_null()).drop("HISTORY_ERROR")


def _binary_encoding(base_data: pl.DataFrame) -> pl.DataFrame:
    return base_data.with_columns(
        pl.when(pl.col("WORKING_TYPE") == "Altersteilzeit").then(1).otherwise(0).alias("ENCODING__RETIREE_PLAN"),
        pl.when(pl.col("CONTRACT_TYPE") == "unbefristet")
        .then(1)
        .otherwise(0)
        .alias("ENCODING__TIME_UNLIMITED_CONTRACT"),
        pl.when(pl.col("GENDER") == "männlich").then(1).otherwise(0).alias("ENCODING__MASCULINE"),
        pl.when(pl.col("FAMILY_STATUS").is_in(["verh", "eLebP"])).then(1).otherwise(0).alias("ENCODING__MARRIED"),
        pl.when(pl.col("WORKING_HOURS") >= 38.5).then(1).otherwise(0).alias("ENCODING__FULL_TIME"),
        pl.when(pl.col("NATIONALITY").str.to_lowercase() != "deutsch")
        .then(1)
        .otherwise(0)
        .alias("ENCODING__FOREIGNER"),
    )


def _group_over_station_and_day(base_data: pl.DataFrame) -> pl.DataFrame:
    aggregations = [
        pl.col(column).mean().alias(f"{column}_MEAN") for column in base_data.columns if column.startswith("ENCODING__")
    ] + [
        pl.col("WORKING_HOURS").mean().alias("ENCODING__WORKING_HOURS_MEAN"),
        pl.col("PERNR").count().alias("ENCODING__NUMBER_OF_EMPLOYEES"),
    ]
    return base_data.group_by(columns.OZ_ZSP, "YEAR_MONTH_1").agg(aggregations)


def pre_process_base_data(base_data: pl.DataFrame) -> pl.DataFrame:
    base_data = _binary_encoding(base_data=base_data)
    return _group_over_station_and_day(base_data=base_data)


def pre_process_sick_leave_data(sick_leave: pl.DataFrame, target_column: str) -> pl.DataFrame:
    last_entry = (
        sick_leave.select(columns.OZ_ZSP)
        .unique()
        .with_columns(
            pl.lit(sick_leave.get_column(columns.DATE).max()).dt.cast_time_unit("ns").alias(columns.DATE),
            pl.lit(0).cast(pl.Int64).alias(target_column),
        )
        .with_columns(
            pl.datetime(
                pl.col(columns.DATE).dt.year().cast(pl.Int64), pl.col(columns.DATE).dt.month().cast(pl.Int64), 1
            ).alias("YEAR_MONTH_1"),
        )
        .select(sick_leave.columns)
    )

    return (
        sick_leave.vstack(last_entry)
        .unique(subset=[columns.DATE, columns.OZ_ZSP], keep="first")
        .sort(by=columns.DATE)
        .upsample(time_column=columns.DATE, every="1d", by=columns.OZ_ZSP)
        .with_columns(
            pl.col(columns.OZ_ZSP).fill_null(strategy="forward"),
            pl.col(target_column).fill_null(strategy="zero"),
        )
        .with_columns(
            pl.datetime(
                pl.col(columns.DATE).dt.year().cast(pl.Int64), pl.col(columns.DATE).dt.month().cast(pl.Int64), 1
            ).alias("YEAR_MONTH_1"),
        )
    )


def merge_sick_leave_data_and_base_data(sick_leave: pl.DataFrame, base_data: pl.DataFrame) -> pl.DataFrame:
    data = sick_leave.join(base_data, on=[columns.OZ_ZSP, "YEAR_MONTH_1"], how="left")
    if sick_leave.height != data.height:
        raise ValueError("Multiple entries in base data for one entry in sick leave data")
    if data.select(pl.all().is_null().any()).transpose().sum().item() != 0:
        raise ValueError("There are entries in the sick leave data, for which no base data exist")
    return data


def aggregate_on_week(data: pl.DataFrame, target_column: str) -> pl.DataFrame:
    data = data.with_columns(
        (pl.col(columns.DATE) - pl.duration(days=pl.col(columns.DATE).dt.weekday() - 1)).alias(columns.DATE)
    )
    return data.group_by(columns.DATE, columns.OZ_ZSP).agg(
        [
            pl.col(column).mean().alias(column)
            for column in data.columns
            if column not in (columns.DATE, columns.OZ_ZSP, target_column, "YEAR_MONTH_1")
        ]
        + [pl.col(target_column).sum()]
    )
