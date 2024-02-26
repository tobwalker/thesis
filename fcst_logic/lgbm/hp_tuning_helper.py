import logging
import pathlib
import datetime
import pickle

import polars as pl
import pandas as pd

from config import columns
from fcst_logic.lgbm import clustering
from fcst_logic.lgbm import rescaling
from fcst_logic.lgbm import hyperparameter
from fcst_logic.lgbm import pca

logger = logging.getLogger(__name__)


def connect_data_and_cv_dates(
    cv_dates: dict[str, tuple[datetime.datetime, datetime.datetime, datetime.datetime]],
    feature_data: pl.DataFrame,
) -> dict[str, tuple[pl.DataFrame, pl.DataFrame]]:
    cv_dict = dict()
    for cv_id, cv_dates in cv_dates.items():
        train_end = datetime.datetime.strptime(str(cv_dates[0].date()), "%Y-%m-%d")
        test_start = datetime.datetime.strptime(str(cv_dates[1].date()), "%Y-%m-%d")
        test_end = datetime.datetime.strptime(str(cv_dates[2].date()), "%Y-%m-%d")
        train_data = feature_data.filter(pl.col(f"DATE__{columns.DATE}") <= train_end)
        stations_with_too_short_of_training_history = (
            train_data.group_by(f"SERIES__{columns.OZ_ZSP}")
            .agg(pl.col(f"DATE__{columns.DATE}").count())
            .filter(pl.col(f"DATE__{columns.DATE}") <= 10)
            .get_column(f"SERIES__{columns.OZ_ZSP}")
        )
        train_data = train_data.filter(
            ~pl.col(f"SERIES__{columns.OZ_ZSP}").is_in(stations_with_too_short_of_training_history)
        )
        test_data = feature_data.filter(
            (pl.col(f"DATE__{columns.DATE}") >= test_start) & (pl.col(f"DATE__{columns.DATE}") <= test_end)
        )
        logger.info(
            f"Stations existing in the test but not in train or have a too short history in the training data (are removed): {test_data.filter(~pl.col(f'SERIES__{columns.OZ_ZSP}').is_in(train_data.get_column(f'SERIES__{columns.OZ_ZSP}'))).get_column(f'SERIES__{columns.OZ_ZSP}').n_unique()}"
        )
        test_data = test_data.filter(
            pl.col(f"SERIES__{columns.OZ_ZSP}").is_in(train_data.get_column(f"SERIES__{columns.OZ_ZSP}"))
        )
        cv_dict[cv_id] = (train_data, test_data)
    return cv_dict


def perform_clustering(
    perform_clustering: bool,
    cv_dict: dict[str, tuple[pl.DataFrame, pl.DataFrame]],
    relevant_weeks: int,
    threshold_for_small_stations: int,
    n_jobs: int,
    upper_quantile: int,
    lower_quantile: int,
    method: str,
    target_column: str,
    run_dir: pathlib.Path,
) -> dict[str, tuple[pl.DataFrame, pl.DataFrame]]:
    cluster_dict = {}

    for cv_id, cv_data in cv_dict.items():
        training_data = cv_data[0].rename(
            {
                f"DATE__{columns.DATE}": columns.DATE,
                f"SERIES__{columns.OZ_ZSP}": columns.OZ_ZSP,
                f"TARGET__{target_column}": target_column,
            }
        )
        if perform_clustering:
            logger.info(f"Clustering is performed - {cv_id}")
            recent_training_data = clustering.extract_recent_history(
                data=training_data, target_column=target_column, relevant_weeks=relevant_weeks
            )
            small_station_cluster = clustering.cluster_with_small_stations_and_iqr_of_zero(
                full_training_data=training_data,
                recent_training_data=recent_training_data,
                target_column=target_column,
                threshold_for_small_stations=threshold_for_small_stations,
                upper_quantile=upper_quantile,
                lower_quantile=lower_quantile,
            )
            complete_history_clusters = clustering.cluster_full_history_stations(
                data=recent_training_data,
                small_station_cluster=small_station_cluster,
                target_column=target_column,
                run_dir=run_dir,
                filename_heatmap=f"clustering_heatmap_{cv_id}",
                upper_quantile=upper_quantile,
                lower_quantile=lower_quantile,
                n_jobs=n_jobs,
                method=method,
            )
            unclusterable_cluster = clustering.retrieve_unclustered_stations(
                run_dir=run_dir,
                filename=f"unclustered_stations_{cv_id}",
                data=recent_training_data,
                complete_history_clusters=complete_history_clusters,
                small_station_cluster=small_station_cluster,
            )
            all_clusters = clustering.collect_clusters(
                data=recent_training_data,
                small_station_cluster=small_station_cluster,
                complete_history_clusters=complete_history_clusters,
                unclusterable_cluster=unclusterable_cluster,
            )

        else:
            logger.info(f"Only small stations get seperately estimated - {cv_id}")
            recent_training_data = clustering.extract_recent_history(
                data=training_data, target_column=target_column, relevant_weeks=relevant_weeks
            )
            all_clusters = clustering.just_small_station_cluster(
                full_training_data=training_data,
                recent_training_data=recent_training_data,
                target_column=target_column,
                threshold_for_small_stations=threshold_for_small_stations,
                upper_quantile=upper_quantile,
                lower_quantile=lower_quantile,
            )
        cluster_dict[cv_id] = all_clusters

    filepath = pathlib.Path(run_dir) / "cluster_dict.pickle"
    with open(filepath, "wb") as handle:
        pickle.dump(cluster_dict, handle)
    return cluster_dict


def connect_clusters_with_cv_data(
    cluster_dict: dict[str, tuple[pl.DataFrame, pl.DataFrame]],
    cv_dict: dict[str, tuple[pl.DataFrame, pl.DataFrame]],
) -> dict[str, tuple[pl.DataFrame, pl.DataFrame]]:
    for cv_id, data in cv_dict.items():
        train_data = data[0].join(
            cluster_dict[cv_id], left_on=f"SERIES__{columns.OZ_ZSP}", right_on=columns.OZ_ZSP, how="inner"
        )
        test_data = data[1].join(
            cluster_dict[cv_id], left_on=f"SERIES__{columns.OZ_ZSP}", right_on=columns.OZ_ZSP, how="inner"
        )
        cv_dict[cv_id] = (train_data, test_data)
    return cv_dict


def only_keep_unclusterable_stations_in_test_data(
    cv_dict: dict[str, tuple[pl.DataFrame, pl.DataFrame]],
) -> dict[str, tuple[pl.DataFrame, pl.DataFrame]]:
    for cv_id, cv_data in cv_dict.items():
        stations_in_other_clusters = (
            cv_data[0]
            .filter(pl.col(columns.CLUSTER) != cv_data[0].get_column(columns.CLUSTER).max())
            .get_column(f"SERIES__{columns.OZ_ZSP}")
        )
        train_data = cv_data[0]
        test_data = cv_data[1].filter(
            ~(
                (pl.col(f"SERIES__{columns.OZ_ZSP}").is_in(stations_in_other_clusters))
                & (pl.col(columns.CLUSTER) == cv_data[0].get_column(columns.CLUSTER).max())
            )
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
    columns_to_rescale = [
        col
        for col in cv_dict["cv_1"][0].columns
        if col.startswith(
            (
                f"FEAT__{target_column}",
                f"FEAT__{columns.PARCEL_AMOUNT}",
                f"FEAT__{columns.BACKLOG}",
            )
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


def rescaling_between_0_and_1_of_features(
    cv_dict: dict[str, tuple[pl.DataFrame, pl.DataFrame]],
) -> dict[str, tuple[pl.DataFrame, pl.DataFrame]]:
    columns_to_rescale = [
        col
        for col in cv_dict["cv_1"][0].columns
        if col.startswith(
            "FEAT__ENCODING",
        )
    ]
    for cv_id, cv_data in cv_dict.items():
        train_data, test_data = rescaling.between_0_and_1_on_train_and_test(
            data=cv_data, columns_to_rescale=columns_to_rescale
        )
        cv_dict[cv_id] = (train_data, test_data)
    return cv_dict


def transform_polars_to_pandas(
    cv_dict: dict[str, tuple[pl.DataFrame, pl.DataFrame]]
) -> dict[str, dict[int, tuple[pd.DataFrame, pd.DataFrame]]]:
    for cv_id, cv_data in cv_dict.items():
        cluster_dict = dict()
        train_data = cv_data[0].to_pandas().set_index([f"DATE__{columns.DATE}", f"SERIES__{columns.OZ_ZSP}"])
        test_data = cv_data[1].to_pandas().set_index([f"DATE__{columns.DATE}", f"SERIES__{columns.OZ_ZSP}"])
        for cluster_id in train_data[columns.CLUSTER].unique():
            train = train_data[train_data[columns.CLUSTER] == cluster_id].drop(columns=columns.CLUSTER)
            test = test_data[test_data[columns.CLUSTER] == cluster_id].drop(columns=columns.CLUSTER)
            cluster_dict[cluster_id] = (train, test)
        cv_dict[cv_id] = cluster_dict.copy()
    return cv_dict


def dimensionality_reduction_on_rescaled_between_0_and_1(
    cv_dict: dict[str, dict[int, tuple[pd.DataFrame, pd.DataFrame]]],
    threshold_of_variance_explanation: float,
    pca_on_encodings: bool,
) -> dict[str, dict[int, tuple[pd.DataFrame, pd.DataFrame]]]:
    if pca_on_encodings:
        logger.info("Dimensionality reduction on features that were rescaled between 0 and 1 is performed")
        columns_to_reduce = [col for col in cv_dict["cv_1"][0][0].columns if col.startswith("FEAT__ENCODING")]
        for cv_id, cluster_dict in cv_dict.items():
            for cluster_id, cluster_data in cluster_dict.items():
                train_data = cluster_data[0]
                test_data = cluster_data[1]
                train_data, test_data = pca.dimensionality_reduction(
                    data=(train_data, test_data),
                    columns_to_reduce=columns_to_reduce,
                    threshold_of_variance_explanation=threshold_of_variance_explanation,
                    cv_id=cv_id,
                    cluster_id=cluster_id,
                    name_id="ENC",
                )
                cluster_dict[cluster_id] = (train_data, test_data)
            cv_dict[cv_id] = cluster_dict
    else:
        logger.info("Dimensionality reduction on features that were rescaled between 0 and 1 is NOT performed")
    return cv_dict


def dimensionality_reduction_on_robust_rescaled(
    cv_dict: dict[str, dict[int, tuple[pd.DataFrame, pd.DataFrame]]],
    threshold_of_variance_explanation: float,
    pca_on_other_features: bool,
    target_column: str,
) -> dict[str, dict[int, tuple[pd.DataFrame, pd.DataFrame]]]:
    if pca_on_other_features:
        logger.info("Dimensionality reduction on all other features is performed")
        columns_to_reduce = [
            col
            for col in cv_dict["cv_1"][0][0].columns
            if col.startswith(
                (
                    f"FEAT__{target_column}",
                    f"FEAT__{columns.PARCEL_AMOUNT}",
                    f"FEAT__{columns.BACKLOG}",
                )
            )
        ]
        for cv_id, cluster_dict in cv_dict.items():
            for cluster_id, cluster_data in cluster_dict.items():
                train_data = cluster_data[0].copy()
                test_data = cluster_data[1].copy()
                if cluster_id != 0:
                    train_data, test_data = pca.dimensionality_reduction(
                        data=(train_data, test_data),
                        columns_to_reduce=columns_to_reduce,
                        threshold_of_variance_explanation=threshold_of_variance_explanation,
                        cv_id=cv_id,
                        cluster_id=cluster_id,
                        name_id="OTHERS",
                    )
                cluster_dict[cluster_id] = (train_data, test_data)
            cv_dict[cv_id] = cluster_dict
    else:
        logger.info("Dimensionality reduction on all other features is NOT performed")
    return cv_dict


def hyperparameter_tuning(
    cv_dict: dict[str, dict[int, tuple[pd.DataFrame, pd.DataFrame]]],
    target_column: str,
    n_estimators: list[int],
    learning_rate: list[float],
    max_depth: list[int],
    num_leaves: list[int],
    n_jobs: int,
) -> dict[str, dict[int, pd.DataFrame]]:
    for cv_id, cluster_dict in cv_dict.items():
        for cluster_id, cluster_data in cluster_dict.items():
            logger.info(f"Training of {cv_id}-{cluster_id}")
            train_data = cluster_data[0].copy()
            test_data = cluster_data[1].copy()
            predictions = hyperparameter.tuner(
                data=(train_data, test_data),
                target_column=target_column,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                num_leaves=num_leaves,
                n_jobs=n_jobs,
            )
            cluster_dict[cluster_id] = predictions.copy()
        cv_dict[cv_id] = cluster_dict
    return cv_dict


def transform_pandas_to_polars(cv_dict: dict[str, dict[int, pd.DataFrame]]) -> dict[str, pl.DataFrame]:
    for cv_id, cluster_dict in cv_dict.items():
        collector_df = pl.DataFrame()
        for cluster_id, predictions in cluster_dict.items():
            predictions_ = (
                pl.DataFrame(predictions.reset_index())
                .with_columns(
                    pl.lit(cluster_id).cast(pl.Int64).alias(columns.CLUSTER),
                )
                .clone()
            )
            collector_df = collector_df.vstack(predictions_)
        cv_dict[cv_id] = collector_df.clone()
    return cv_dict


def reverse_rescaling_of_target_and_predictions(
    cv_dict: dict[str, pl.DataFrame],
    target_rescaling_parameters: dict[str, pl.DataFrame],
    target_column: str,
) -> dict[str, pl.DataFrame]:
    for cv_id, predictions in cv_dict.items():
        predictions = (
            predictions.join(
                target_rescaling_parameters[cv_id], on=[f"SERIES__{columns.OZ_ZSP}", columns.CLUSTER], how="left"
            )
            .with_columns(
                [
                    (pl.col(column) * pl.col("IQR") + pl.col("MEDIAN")).alias(column)
                    for column in predictions.columns
                    if (column.startswith("PRED__")) | (column == f"TARGET__{target_column}")
                ]
            )
            .with_columns(
                [
                    pl.when(pl.col(column) < 0).then(0).otherwise(pl.col(column)).alias(column)
                    for column in predictions.columns
                    if column.startswith("PRED__")
                ]
            )
            .drop("IQR", "MEDIAN")
        )
        cv_dict[cv_id] = predictions.clone()
    return cv_dict


def calculate_absolute_error(
    cv_dict: dict[str, pl.DataFrame], target_column: str, run_dir: str | pathlib.Path
) -> pl.DataFrame:
    collector_df = pl.DataFrame()
    for cv_id, predictions in cv_dict.items():
        predictions = predictions.with_columns(
            [
                abs(pl.col(col) - pl.col(f"TARGET__{target_column}")).alias(f"ERROR__{col}")
                for col in predictions.columns
                if col.startswith("PRED__")
            ]
        ).with_columns(pl.lit(cv_id).alias("CV_ID"))
        collector_df = collector_df.vstack(predictions)
    filepath = pathlib.Path(run_dir) / "predictions_and_errors.pq"
    collector_df.write_parquet(filepath)
    return collector_df
