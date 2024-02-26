import logging
import pathlib
import datetime
import pickle

import polars as pl
import pandas as pd

from config import columns
from fcst_logic.tft import clustering
from fcst_logic.lgbm import rescaling
from fcst_logic.tft import data_loader
from fcst_logic.tft import hyperparameter

logger = logging.getLogger(__name__)


def connect_data_and_cv_dates(
    data: pl.DataFrame, cv_dates: dict[str, tuple[datetime.datetime, datetime.datetime]]
) -> dict[str, tuple[pl.DataFrame, pl.DataFrame]]:
    cv_dict = dict()
    for cv_id, dates in cv_dates.items():
        train_data = data.filter(pl.col(f"DATE__{columns.DATE}") < dates[0])
        stations_with_too_short_of_training_history = (
            train_data.group_by(f"SERIES__{columns.OZ_ZSP}")
            .agg(pl.col(f"DATE__{columns.DATE}").count())
            .filter(pl.col(f"DATE__{columns.DATE}") <= 10)
            .get_column(f"SERIES__{columns.OZ_ZSP}")
        )
        train_data = train_data.filter(
            ~pl.col(f"SERIES__{columns.OZ_ZSP}").is_in(stations_with_too_short_of_training_history)
        )
        test_data = data.filter(
            (pl.col(f"DATE__{columns.DATE}") >= dates[0]) & (pl.col(f"DATE__{columns.DATE}") <= dates[1])
        )

        logger.info(
            f"Stations existing in the test but not in train or have a too short history in the training data (are removed): {test_data.filter(~pl.col(f'SERIES__{columns.OZ_ZSP}').is_in(train_data.get_column(f'SERIES__{columns.OZ_ZSP}'))).get_column(f'SERIES__{columns.OZ_ZSP}').n_unique()}"
        )
        test_data = test_data.filter(
            pl.col(f"SERIES__{columns.OZ_ZSP}").is_in(train_data.get_column(f"SERIES__{columns.OZ_ZSP}"))
        )
        cv_dict[cv_id] = (train_data, test_data)
    return cv_dict


def attach_cluster_id(
    cv_dict: dict[str, tuple[pl.DataFrame, pl.DataFrame]],
    target_column: str,
    threshold_for_small_stations: int,
    upper_quantile: int,
    lower_quantile: int,
) -> pl.DataFrame:
    for cv_id, cv_data in cv_dict.items():
        small_stations = clustering.retrieve_list_of_small_stations(
            data=cv_data[0],
            target_column=target_column,
            threshold_for_small_stations=threshold_for_small_stations,
            upper_quantile=upper_quantile,
            lower_quantile=lower_quantile,
        )
        train_data = cv_data[0].with_columns(
            pl.when(pl.col(f"SERIES__{columns.OZ_ZSP}").is_in(small_stations))
            .then(0)
            .otherwise(1)
            .alias(columns.CLUSTER)
        )
        test_data = cv_data[1].with_columns(
            pl.when(pl.col(f"SERIES__{columns.OZ_ZSP}").is_in(small_stations))
            .then(0)
            .otherwise(1)
            .alias(columns.CLUSTER)
        )
        cv_dict[cv_id] = (train_data, test_data)
    return cv_dict


def robust_rescaling_of_target(
    run_dir: str | pathlib.Path,
    cv_dict: dict[str, tuple[pl.DataFrame, pl.DataFrame]],
    target_column: str,
    upper_quantile: int,
    lower_quantile: int,
) -> tuple[dict[str, tuple[pl.DataFrame, pl.DataFrame]], dict[str, pl.DataFrame]]:
    rescale_parameter_of_target_dict = dict()
    for cv_id, cv_data in cv_dict.items():
        cluster_parameter, train_data, test_data = rescaling.robust_applied_on_train_and_test(
            data=cv_data,
            group_columns=[f"SERIES__{columns.OZ_ZSP}", columns.CLUSTER],
            columns_to_rescale=[f"TARGET__{target_column}"],
            upper_quantile=upper_quantile,
            lower_quantile=lower_quantile,
        )
        cv_dict[cv_id] = (train_data, test_data)
        rescale_parameter_of_target_dict[cv_id] = cluster_parameter.with_columns(
            pl.when(pl.col(columns.CLUSTER) == 0).then(1).otherwise(pl.col("IQR")).alias("IQR"),
            pl.when(pl.col(columns.CLUSTER) == 0).then(0).otherwise(pl.col("MEDIAN")).alias("MEDIAN"),
        )
    with open(pathlib.Path(run_dir) / "rescaling_parameters_of_target.pickle", "wb") as handle:
        pickle.dump(rescale_parameter_of_target_dict, handle)
    return cv_dict, rescale_parameter_of_target_dict


def robust_rescaling_of_features(
    cv_dict: dict[str, tuple[pl.DataFrame, pl.DataFrame]],
    target_column: str,
    upper_quantile: int,
    lower_quantile: int,
) -> dict[str, tuple[pl.DataFrame, pl.DataFrame]]:
    try:
        columns_to_rescale = [
            col
            for col in cv_dict["cv_1"][0].columns
            if (
                col.startswith("FEAT__")
                & ((target_column in col) | (columns.PARCEL_AMOUNT in col) | (columns.BACKLOG in col))
            )
        ]
    except KeyError:
        columns_to_rescale = [
            col
            for col in cv_dict["test"][0].columns
            if (
                col.startswith("FEAT__")
                & ((target_column in col) | (columns.PARCEL_AMOUNT in col) | (columns.BACKLOG in col))
            )
        ]
    for cv_id, cv_data in cv_dict.items():
        _, train_data, test_data = rescaling.robust_applied_on_train_and_test(
            data=cv_data,
            group_columns=[f"SERIES__{columns.OZ_ZSP}", columns.CLUSTER],
            columns_to_rescale=columns_to_rescale,
            upper_quantile=upper_quantile,
            lower_quantile=lower_quantile,
        )
        cv_dict[cv_id] = (train_data, test_data)
    return cv_dict


def transform_polars_to_pandas(
    cv_dict: dict[str, tuple[pl.DataFrame, pl.DataFrame]], run_dir: pathlib.Path | str
) -> dict[str, dict[int, tuple[pd.DataFrame, pd.DataFrame]]]:
    for cv_id, cv_data in cv_dict.items():
        cluster_dict = dict()
        train_data = cv_data[0].to_pandas().set_index([f"DATE__{columns.DATE}", f"SERIES__{columns.OZ_ZSP}"])
        test_data = cv_data[1].to_pandas().set_index([f"DATE__{columns.DATE}", f"SERIES__{columns.OZ_ZSP}"])
        for cluster_id in train_data[columns.CLUSTER].unique():
            train = train_data[train_data[columns.CLUSTER] == cluster_id].drop(
                columns=[columns.CLUSTER, "FEAT__HORIZON"]
            )
            test = test_data[test_data[columns.CLUSTER] == cluster_id].drop(columns=[columns.CLUSTER, "FEAT__HORIZON"])
            for col in [col for col in train.columns if "_CATEGORICALS__" in col]:
                train[col] = train[col].astype("str").astype("category")
                test[col] = test[col].astype("str").astype("category")
            train["TEST"] = False
            test["TEST"] = True
            data = pd.concat([train, test], axis=0).copy()
            cluster_dict[cluster_id] = data
        cv_dict[cv_id] = cluster_dict.copy()
    with open(pathlib.Path(run_dir) / "data.pickle", "wb") as handle:
        pickle.dump(cv_dict, handle)
    return cv_dict


def hyperparameter_tuning(
    cv_dict: dict[str, tuple[pl.DataFrame, pl.DataFrame]],
    hidden_sizes: list[int],
    limit_train_batches: list[int],
    horizons: int,
    max_encoder_length: int,
    batch_size: int,
):
    result = {}
    result_cluster = {}
    for cv_id, cluster_dict in cv_dict.items():
        for cluster_id, data in cluster_dict.items():
            training, train_dataloader, val_dataloader = data_loader.data_loader(
                data=data,
                horizons=horizons,
                max_encoder_length=max_encoder_length,
                batch_size=batch_size,
            )
            models = hyperparameter(
                training=training,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                hidden_sizes=hidden_sizes,
                limit_train_batches=limit_train_batches,
            )
            result_cluster[cluster_id] = models
        result[cv_id] = result_cluster
    return result
