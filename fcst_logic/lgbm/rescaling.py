import logging

import polars as pl

from config import columns

logger = logging.getLogger("__name__")


def _parameter_robust(
    data: pl.DataFrame, group_columns: list[str], column: str, upper_quantile: int, lower_quantile: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    return (
        data.filter(pl.col("FEAT__HORIZON") == 1)
        .group_by(group_columns)
        .agg(
            (pl.col(column).quantile(upper_quantile / 100) - pl.col(column).quantile(lower_quantile / 100)).alias(
                "IQR"
            ),
            pl.col(column).median().alias("MEDIAN"),
        )
    )


def robust_applied_on_train_and_test(
    data: tuple[pl.DataFrame, pl.DataFrame],
    group_columns: list[str],
    columns_to_rescale: list[str],
    upper_quantile: int,
    lower_quantile: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train_data = data[0]
    test_data = data[1]
    for column in columns_to_rescale:
        rescale_parameters = _parameter_robust(
            data=train_data,
            group_columns=group_columns,
            column=column,
            upper_quantile=upper_quantile,
            lower_quantile=lower_quantile,
        )
        train_data = (
            train_data.join(rescale_parameters, on=group_columns, how="left")
            .with_columns(
                pl.when(pl.col(columns.CLUSTER) == 0).then(1).otherwise(pl.col("IQR")).alias("IQR"),
                pl.when(pl.col(columns.CLUSTER) == 0).then(0).otherwise(pl.col("MEDIAN")).alias("MEDIAN"),
            )
            .with_columns((pl.col(column) - pl.col("MEDIAN")) / pl.col("IQR").alias(column))
            .with_columns(pl.when(pl.col("IQR") == 0).then(0).otherwise(pl.col(column)).alias(column))
            .with_columns(pl.when(pl.col(column) > 10).then(10).otherwise(pl.col(column)).alias(column))
            .with_columns(pl.when(pl.col(column) < -10).then(-10).otherwise(pl.col(column)).alias(column))
            .drop("IQR", "MEDIAN")
        )
        test_data = (
            test_data.join(rescale_parameters, on=group_columns, how="left")
            .with_columns(
                pl.when(pl.col(columns.CLUSTER) == 0).then(1).otherwise(pl.col("IQR")).alias("IQR"),
                pl.when(pl.col(columns.CLUSTER) == 0).then(0).otherwise(pl.col("MEDIAN")).alias("MEDIAN"),
            )
            .with_columns((pl.col(column) - pl.col("MEDIAN")) / pl.col("IQR").alias(column))
        )
        if not column.startswith("TARGET__"):
            test_data = (
                test_data.with_columns(pl.when(pl.col("IQR") == 0).then(0).otherwise(pl.col(column)).alias(column))
                .with_columns(pl.when(pl.col(column) > 10).then(10).otherwise(pl.col(column)).alias(column))
                .with_columns(pl.when(pl.col(column) < -10).then(-10).otherwise(pl.col(column)).alias(column))
                .drop("IQR", "MEDIAN")
            )
    return rescale_parameters, train_data, test_data


def _parameter_0_1(data: pl.DataFrame, column: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    return (
        data.filter(pl.col("FEAT__HORIZON") == 1)
        .group_by(columns.CLUSTER)
        .agg(
            pl.col(column).min().alias("MIN"),
            (pl.col(column).max() - pl.col(column).min()).alias("MAX_MIN_DISTANCE"),
        )
        .with_columns(
            pl.when(pl.col("MAX_MIN_DISTANCE") == 0)
            .then(pl.col("MIN"))
            .otherwise(pl.col("MAX_MIN_DISTANCE"))
            .alias("MAX_MIN_DISTANCE")
        )
    )


def between_0_and_1_on_train_and_test(
    data: tuple[pl.DataFrame, pl.DataFrame], columns_to_rescale: list[str]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    train_data = data[0]
    test_data = data[1]
    for column in columns_to_rescale:
        rescale_parameters = _parameter_0_1(data=train_data, column=column)
        train_data = (
            train_data.join(rescale_parameters, on=[columns.CLUSTER], how="left")
            .with_columns(
                pl.when(pl.col("MAX_MIN_DISTANCE") == 0)
                .then(0)
                .otherwise((pl.col(column) - pl.col("MIN")) / (pl.col("MAX_MIN_DISTANCE")))
                .alias(column)
            )
            .drop("MIN", "MAX_MIN_DISTANCE")
        )

        test_data = (
            test_data.join(rescale_parameters, on=[columns.CLUSTER], how="left")
            .with_columns(
                pl.when(pl.col("MAX_MIN_DISTANCE") == 0)
                .then(0)
                .otherwise((pl.col(column) - pl.col("MIN")) / (pl.col("MAX_MIN_DISTANCE")))
                .alias(column)
            )
            .drop("MIN", "MAX_MIN_DISTANCE")
        )
    return train_data, test_data
