import logging
import pathlib

import polars as pl

from config import columns
import fcst_logic.lgbm.data_collection as lgbm_data


logger = logging.getLogger(__name__)


def _adapt_data_to_tft(data: pl.DataFrame, target_column: str):
    data = (
        data.with_columns(pl.col(target_column).alias(f"TARGET__{target_column}"))
        .with_columns(pl.col(columns.OZ_ZSP).str.slice(0, 12).alias("SERIES__OZ_ZSP"))
        .with_columns(
            pl.col(columns.OZ_ZSP).str.slice(0, 12).cast(pl.Int64).alias("FEAT__STATIC_CATEGORICALS__ENCODING__OZ_ZSP"),
            pl.col(columns.OZ_ZSP).str.slice(0, 8).cast(pl.Int64).alias("FEAT__STATIC_CATEGORICALS__ENCODING__OZ_ZSPL"),
            pl.col(columns.OZ_ZSP).str.slice(0, 4).cast(pl.Int64).alias("FEAT__STATIC_CATEGORICALS__ENCODING__OZ_NL"),
            pl.col(columns.OZ_ZSP)
            .str.slice(4, 2)
            .cast(pl.Int64)
            .alias("FEAT__STATIC_CATEGORICALS__ENCODING__OZ_DIVISION"),
        )
        .with_columns(
            pl.col(columns.DATE).dt.month().cast(pl.Int64).alias("FEAT__TIME_VARYING_KNOWN_CATEGORICALS__MONTH"),
            pl.col(columns.DATE).dt.week().cast(pl.Int64).alias("FEAT__TIME_VARYING_KNOWN_CATEGORICALS__WEEK"),
        )
        .with_columns(
            [
                pl.col(col).alias(f"FEAT__TIME_VARYING_KNOWN_REALS__{col}")
                for col in ["COVID_CASES", "PUBLIC_HOLIDAYS", "CHRISTMAS_SEASON"]
            ]
        )
        .with_columns(
            [
                pl.col(col).alias(f"FEAT__TIME_VARYING_UNKNOWN_REALS__{col}")
                for col in [
                    "ABS_SICK_LEAVES",
                    "BACKLOG",
                    "PARCEL_AMOUNT",
                    "ABS_SICK_LEAVES_ZSPL",
                    "ABS_SICK_LEAVES_NL",
                ]
                + [col for col in data.columns if col.startswith("ENCODING__")]
            ]
        )
        .with_columns(pl.col(columns.DATE).alias(f"DATE__{columns.DATE}"), pl.lit(1).alias("FEAT__HORIZON"))
    )
    return data.select([col for col in data.columns if col.startswith(("DATE__", "SERIES__", "TARGET__", "FEAT__"))])


def _get_time_idx(data: pl.DataFrame) -> pl.DataFrame:
    time_idx = {date: i for i, date in enumerate(sorted(data.get_column(f"DATE__{columns.DATE}").unique()))}
    time_idx = pl.DataFrame(list(time_idx.items()), schema=[f"DATE__{columns.DATE}", "DATE__TIME_IDX"])
    return data.join(time_idx, on=f"DATE__{columns.DATE}", how="outer")


def collector(
    import_dir: str | pathlib.Path,
    target_column: str,
    start_date: str,
    end_date: str,
    filename_sick_leave: str,
    filename_base_data: str,
    filename_backlog: str,
    filename_parcel_amount: str,
    filename_holiday: str,
) -> pl.DataFrame:
    data = lgbm_data.collector(
        import_dir=import_dir,
        target_column=target_column,
        start_date=start_date,
        end_date=end_date,
        filename_sick_leave=filename_sick_leave,
        filename_base_data=filename_base_data,
        filename_backlog=filename_backlog,
        filename_parcel_amount=filename_parcel_amount,
        filename_holiday=filename_holiday,
    )
    data = _adapt_data_to_tft(data=data, target_column=target_column)
    return _get_time_idx(data=data)
