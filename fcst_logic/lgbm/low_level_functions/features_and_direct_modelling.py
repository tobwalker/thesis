import polars as pl
from config import columns, constants
import logging

logger = logging.getLogger("__name__")


def _lagged_feature_over_a_column(data: pl.DataFrame, column: str, lags: list[int], horizon: int) -> pl.DataFrame:
    return (
        data.sort(columns.DATE)
        .with_columns(pl.lit(horizon).alias("FEAT__HORIZON"))
        .with_columns(
            [
                pl.col({column}).shift(lag + (horizon - 1)).over(columns.OZ_ZSP).alias(f"FEAT__{column}__LAGGED_{lag}")
                for lag in lags
            ]
        )
    )


def lagged_feature_over_all_columns(
    data: pl.DataFrame,
    target_column: str,
    horizons: int,
    target_lags: list[int],
    encoding_lags: list[int],
    op_covid_lags: list[int],
) -> pl.DataFrame:
    logger.info(f"Lags on {[col for col in data.columns if target_column in col]}: {target_lags}")
    logger.info(f"Lags on {[col for col in data.columns if col.startswith('ENCODING_')]}: {encoding_lags}")
    logger.info(f"Lags on {constants.op_features}: {op_covid_lags}")
    horizoned_data = pl.DataFrame()
    for horizon in range(1, horizons + 1):
        temp_data = data.clone()
        for col in [col for col in data.columns if target_column in col]:
            temp_data = _lagged_feature_over_a_column(data=temp_data, column=col, horizon=horizon, lags=target_lags)
        for col in [col for col in data.columns if col.startswith("ENCODING_")]:
            temp_data = _lagged_feature_over_a_column(data=temp_data, column=col, horizon=horizon, lags=encoding_lags)
        for col in constants.op_features:
            temp_data = _lagged_feature_over_a_column(data=temp_data, column=col, horizon=horizon, lags=op_covid_lags)
        horizoned_data = horizoned_data.vstack(temp_data)
    return horizoned_data


def rolling_mean_on_lagged_1_feature(data: pl.DataFrame, column: str, window_size: int) -> pl.DataFrame:
    logger.info(f"Rolling mean with window size {window_size} is applied to {column}")
    return data.sort(columns.OZ_ZSP, columns.DATE).with_columns(
        pl.col(column)
        .rolling_mean(window_size=window_size, min_periods=window_size, closed=None)
        .over([columns.OZ_ZSP, "FEAT__HORIZON"])
        .alias(f"{column}__MOVING_AVG_{window_size}")
    )


def rolling_std_on_lagged_1_feature(data: pl.DataFrame, column: str, window_size: int) -> pl.DataFrame:
    logger.info(f"Rolling std with window size {window_size} is applied to {column}")
    return data.sort(columns.OZ_ZSP, columns.DATE).with_columns(
        pl.col(column)
        .rolling_std(window_size=window_size, min_periods=window_size, closed=None)
        .over([columns.OZ_ZSP, "FEAT__HORIZON"])
        .alias(f"{column}__MOVING_STD_{window_size}")
    )


# def replace_inf_with_0_in_backlog(data: pl.DataFrame) -> pl.DataFrame:
#     return data.with_columns(
#         [
#             pl.when(pl.col(column).is_infinite()).then(0).otherwise(pl.col(column)).name.keep()
#             for column in data.columns
#             if "__BACKLOG_" in column
#         ]
#     )


def static_features(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_columns(
        pl.col(columns.DATE).dt.month().alias("FEAT__MONTH"),
        pl.col(columns.DATE).dt.week().alias("FEAT__WEEK"),
        pl.col(columns.COVID_CASES).alias("FEAT__COVID_CASES"),
        pl.col(columns.OZ_ZSP).str.slice(0, 8).cast(pl.Int32).alias("FEAT__ENCODING__OZ_ZSPL"),
        pl.col(columns.OZ_ZSP).str.slice(0, 4).cast(pl.Int32).alias("FEAT__ENCODING__OZ_NL"),
        pl.col(columns.OZ_ZSP).cast(pl.Int64).alias(f"FEAT__ENCODING__{columns.OZ_ZSP}"),
        pl.col(columns.OZ_ZSP).str.slice(4, 2).map_dict({"33": 0, "36": 1}).alias("FEAT__ENCODING__DIVISION"),
        pl.col(columns.CHRISTMAS_SEASON).alias(f"FEAT__{columns.CHRISTMAS_SEASON}"),
        pl.col(columns.PUBLIC_HOLIDAYS).alias(f"FEAT__{columns.PUBLIC_HOLIDAYS}"),
    )


def define_target_column(data: pl.DataFrame, target_column: str) -> pl.DataFrame:
    return data.rename({target_column: f"TARGET__{target_column}"})


def define_date_and_series_columns(data: pl.DataFrame) -> pl.DataFrame:
    return data.rename({columns.DATE: f"DATE__{columns.DATE}", columns.OZ_ZSP: f"SERIES__{columns.OZ_ZSP}"})


def keep_relevant_columns(data: pl.DataFrame) -> pl.DataFrame:
    return data.select([col for col in data.columns if col.startswith(("FEAT__", "TARGET__", "DATE__", "SERIES__"))])


def drop_nans(data: pl.DataFrame) -> pl.DataFrame:
    return data.drop_nulls()
