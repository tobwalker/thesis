import logging
import ast
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from fastdtw import fastdtw
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import leaves_list

from config import columns

logger = logging.getLogger(__name__)


def extract_full_history_stations(
    data: pl.DataFrame, small_station_cluster: pl.DataFrame, target_column: str
) -> list[str]:
    data = data.filter(~pl.col(columns.OZ_ZSP).is_in(small_station_cluster.get_column(columns.OZ_ZSP)))
    data_pivot = data.pivot(values=target_column, index=columns.DATE, columns=columns.OZ_ZSP).sort(columns.DATE)
    list_of_stations_with_full_history = (
        (data_pivot.select(pl.exclude(columns.DATE)).null_count() / len(data_pivot))
        .transpose(include_header=True)
        .filter(pl.col("column_0") == 0)
        .get_column("column")
        .to_list()
    )
    logger.info(
        f"{len(list_of_stations_with_full_history)} of {len(data_pivot.select(pl.exclude(columns.DATE)).columns)} stations have a full history and therefore will be clustered"
    )
    return list_of_stations_with_full_history


def reshaping_and_rescaling_data_of_stations_with_full_history(
    data: pl.DataFrame,
    list_of_stations_with_full_history: list,
    target_column: str,
    upper_quantile: int = 75,
    lower_quantile: int = 25,
) -> list[tuple[str, list[int]]]:
    data = data.filter(pl.col(columns.OZ_ZSP).is_in(list_of_stations_with_full_history))
    data_rescaling_parameters = (
        data.group_by(columns.OZ_ZSP)
        .agg(
            pl.col(target_column).quantile(upper_quantile / 100).alias("upper_quantile"),
            pl.col(target_column).quantile(lower_quantile / 100).alias("lower_quantile"),
            pl.col(target_column).median().alias("MEDIAN"),
        )
        .with_columns(
            (pl.col("upper_quantile") - pl.col("lower_quantile")).alias("IQR"),
        )
    ).select(columns.OZ_ZSP, "IQR", "MEDIAN")
    data_pivot = (
        data.join(data_rescaling_parameters, on=columns.OZ_ZSP, how="left")
        .with_columns(((pl.col(target_column) - pl.col("MEDIAN")) / pl.col("IQR")).alias(target_column))
        .pivot(values=target_column, index=columns.DATE, columns=columns.OZ_ZSP)
        .sort(columns.DATE)
        .select(pl.exclude(columns.DATE))
    )
    return [
        (station_id, [float(value) for value in time_series])
        for station_id, time_series in zip(data_pivot.columns, data_pivot.transpose().to_numpy())
    ]


def _compute_dtw_distance_between_two_series(
    args: tuple[int, int],
    time_series: list[tuple[str, list[int]]],
) -> tuple[int, int, float]:
    i, j = args
    distance, _ = fastdtw(time_series[i][1], time_series[j][1])
    return i, j, distance


def compute_dtw_distance_matrix(
    reshaped_time_series_data: list[tuple[str, list[int]]], n_jobs: float = 4
) -> np.ndarray:
    number_of_time_series = len(reshaped_time_series_data)
    distance_matrix = np.zeros((number_of_time_series, number_of_time_series))
    args_list = [(i, j) for i in range(number_of_time_series) for j in range(i, number_of_time_series)]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_dtw_distance_between_two_series)(args, reshaped_time_series_data) for args in args_list
    )
    for i, j, distance in results:
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance
    logger.debug(f"Shape of dtw matrix: {distance_matrix.shape}")
    return distance_matrix


def compute_similarity_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    median_dist = np.median(distance_matrix)
    return np.exp(-(distance_matrix**2) / (2.0 * median_dist**2))


def compute_linkage_matrix(similarity_matrix: np.ndarray, method: str = "average") -> np.ndarray:
    return sch.linkage(similarity_matrix, method=method)


def _sorted_heatmap(
    linkage_matrix: np.ndarray,
    similarity_matrix: np.ndarray,
    clusters: np.ndarray,
    base_dir: pathlib.Path | str,
    filename: str,
    show: bool = False,
) -> None:
    order = leaves_list(linkage_matrix)
    sorted_matrix = similarity_matrix[order][:, order]
    plt.figure(figsize=(10, 10))
    sns.heatmap(sorted_matrix, cmap="viridis")
    for cluster in set(clusters):
        positions = np.where(np.diff(clusters[order] == cluster))[0]
        for position in positions:
            plt.axhline(position + 1, color="white", lw=2)
            plt.axvline(position + 1, color="white", lw=2)
    plt.title("Heatmap on the similarity matrix, sorted based on the linkage")
    plt.savefig(pathlib.Path(base_dir) / (filename + ".png"))
    if show:
        plt.show()
    else:
        plt.close()


def retrieve_clusters_based_on_heatmap(
    linkage_matrix: np.ndarray, similarity_matrix: np.ndarray, filename_heatmap: str, run_dir: str | pathlib.Path
) -> np.ndarray:
    while True:
        threshold = input("Enter a float to set cut threshold (or 'move on' to move on): ")
        if threshold == "move on":
            break
        else:
            try:
                clusters = fcluster(linkage_matrix, threshold, criterion="distance")
                _sorted_heatmap(
                    linkage_matrix=linkage_matrix,
                    similarity_matrix=similarity_matrix,
                    clusters=clusters,
                    base_dir=run_dir,
                    filename=filename_heatmap,
                )
            except ValueError:
                print("Please enter a valid input.")

    combined_cluster = np.array([])
    while True:
        user_input = input(
            f"Enter a list of clusters to combine {set(sorted(combined_cluster))} ('reverse' to reverse last combination, 'move on' to finish merging): "
        )
        if user_input == "move on":
            combined_cluster = combined_cluster if len(combined_cluster) != 0 else clusters
            _sorted_heatmap(
                linkage_matrix=linkage_matrix,
                similarity_matrix=similarity_matrix,
                clusters=combined_cluster,
                base_dir=run_dir,
                filename=filename_heatmap,
                show=True,
            )
            return combined_cluster
        if user_input == "reverse":
            if combined_cluster.size == 0:
                print("Reverse only possible after a combine list was passed")
            else:
                combined_cluster = clusters
                _sorted_heatmap(
                    linkage_matrix=linkage_matrix,
                    similarity_matrix=similarity_matrix,
                    clusters=combined_cluster,
                    base_dir=run_dir,
                    filename=filename_heatmap,
                )
        else:
            try:
                if combined_cluster.size != 0:
                    clusters = combined_cluster
                combine_cluster = ast.literal_eval(user_input)
                combined_cluster = np.where(np.isin(clusters, combine_cluster), min(combine_cluster), clusters)
                _sorted_heatmap(
                    linkage_matrix=linkage_matrix,
                    similarity_matrix=similarity_matrix,
                    clusters=combined_cluster,
                    base_dir=run_dir,
                    filename=filename_heatmap,
                )
            except ValueError:
                print("Please enter a valid input.")


def connect_station_with_cluster(
    reshaped_time_series: list[tuple[str, list[int]]], clusters: np.ndarray
) -> pl.DataFrame:
    complete_history_clusters = pl.DataFrame(
        zip([station_id for station_id, _ in reshaped_time_series], clusters)
    ).rename({"column_0": columns.OZ_ZSP, "column_1": columns.CLUSTER})
    logger.info(
        f"Distribution of clusters:\n{complete_history_clusters.get_column(columns.CLUSTER).value_counts().sort(columns.CLUSTER)}"
    )
    return complete_history_clusters


def remove_clusters(
    complete_history_clusters: pl.DataFrame,
) -> pl.DataFrame:
    remove_cluster = input(
        f"Input clusters to remove {complete_history_clusters.get_column(columns.CLUSTER).unique()}: list[int]"
    )
    remove_cluster = ast.literal_eval(remove_cluster)
    logger.info(f"Remove clusters: {remove_cluster}")
    if remove_cluster is not None:
        complete_history_clusters = complete_history_clusters.filter(~pl.col(columns.CLUSTER).is_in(remove_cluster))
    return complete_history_clusters
