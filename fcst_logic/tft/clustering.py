import logging

import polars as pl

from config import columns


logger = logging.getLogger(__name__)


def retrieve_list_of_small_stations(
    data: pl.DataFrame,
    target_column: str,
    threshold_for_small_stations: int,
    upper_quantile: int,
    lower_quantile: int,
) -> list[str]:
    mean_data = data.group_by(f"SERIES__{columns.OZ_ZSP}").agg(pl.col(f"TARGET__{target_column}").mean())
    stations_without_enough_sick_leaves = (
        mean_data.filter(pl.col(f"TARGET__{target_column}") < threshold_for_small_stations)
        .get_column(f"SERIES__{columns.OZ_ZSP}")
        .to_list()
    )
    iqr_data = (
        data.group_by(f"SERIES__{columns.OZ_ZSP}")
        .agg(
            [
                pl.col(f"TARGET__{target_column}").quantile(upper_quantile / 100).alias("upper_quantile"),
                pl.col(f"TARGET__{target_column}").quantile(lower_quantile / 100).alias("lower_quantile"),
            ]
        )
        .with_columns((pl.col("upper_quantile") - pl.col("lower_quantile")).alias("IQR"))
    )
    stations_with_iqr_of_less_than_1_over_whole_training_data = (
        iqr_data.filter(pl.col("IQR") < 1).get_column(f"SERIES__{columns.OZ_ZSP}").to_list()
    )
    small_stations = stations_without_enough_sick_leaves + stations_with_iqr_of_less_than_1_over_whole_training_data
    logger.info(f"{len(set(small_stations))} fall into the small station cluster")
    return list(set(small_stations))
