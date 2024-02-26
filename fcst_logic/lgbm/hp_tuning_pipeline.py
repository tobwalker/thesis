import logging
import pathlib
import datetime

from config import columns
from fcst_logic.lgbm import data_collection, features, cross_validation, hp_tuning_helper

logger = logging.getLogger(__name__)


def perform_cross_validation(config: dict):
    # Setup directory structure
    run_dir = (
        pathlib.Path(config["base_dir"])
        / "cross_validation"
        / f"cross_validation_pipeline_{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"
    )
    run_dir.mkdir(parents=False, exist_ok=False)
    import_dir_ = pathlib.Path(config["base_dir"]) / "imports"
    # Import and preprocess data
    data = data_collection.collector(
        import_dir=import_dir_,
        target_column=config["target_column"],
        start_date=config["start_date"],
        end_date=config["end_date"],
        filename_base_data=config["filename_base_data"],
        filename_sick_leave=config["filename_sick_leave"],
        filename_backlog=config["filename_backlog"],
        filename_parcel_amount=config["filename_parcel_amount"],
        filename_holiday=config["filename_holidays_36"],
    )
    data.write_parquet(pathlib.Path(run_dir) / "collected_data.pq")
    # Compute cross validation dates
    cv_dates = cross_validation.dates(
        horizons=config["horizons"],
        n_cv=config["number_cv_splits"],
        end_date=data.get_column(columns.DATE).max(),
        test_set=True,
    )
    # Feature engineering
    data = features.feature_engineering(
        data=data,
        target_column=config["target_column"],
        horizons=config["horizons"],
        target_lags=config["target_lags"],
        encoding_lags=config["encoding_lags"],
        other_features_lags=config["op_lags"],
    )
    # Combine CV and Feature
    cv_dict = hp_tuning_helper.connect_data_and_cv_dates(cv_dates=cv_dates, feature_data=data)
    # Perform clustering
    cluster_dict = hp_tuning_helper.perform_clustering(
        perform_clustering=config["perform_clustering"],
        cv_dict=cv_dict,
        target_column=config["target_column"],
        threshold_for_small_stations=config["threshold_for_small_stations"],
        relevant_weeks=config["relevant_weeks"],
        n_jobs=config["n_jobs"],
        upper_quantile=config["rescaling_quantiles"][1],
        lower_quantile=config["rescaling_quantiles"][0],
        method="average",
        run_dir=run_dir,
    )
    cv_dict = hp_tuning_helper.connect_clusters_with_cv_data(
        cluster_dict=cluster_dict,
        cv_dict=cv_dict,
    )
    if config["perform_clustering"]:
        cv_dict = hp_tuning_helper.only_keep_unclusterable_stations_in_test_data(cv_dict=cv_dict)
    # Rescaling
    cv_dict, rescale_parameter_of_target_dict = hp_tuning_helper.robust_rescaling_of_target(
        run_dir=run_dir,
        cv_dict=cv_dict,
        target_column=config["target_column"],
        upper_quantile=config["rescaling_quantiles"][1],
        lower_quantile=config["rescaling_quantiles"][0],
    )
    cv_dict = hp_tuning_helper.robust_rescaling_of_features(
        cv_dict=cv_dict,
        target_column=config["target_column"],
        upper_quantile=config["rescaling_quantiles"][1],
        lower_quantile=config["rescaling_quantiles"][0],
    )
    cv_dict = hp_tuning_helper.rescaling_between_0_and_1_of_features(cv_dict=cv_dict)
    # Transform from Polars to pandas
    cv_dict = hp_tuning_helper.transform_polars_to_pandas(cv_dict=cv_dict)
    # Dimensionality reduction
    cv_dict = hp_tuning_helper.dimensionality_reduction_on_rescaled_between_0_and_1(
        cv_dict=cv_dict,
        pca_on_encodings=config["peform_dimensionality_reduction"],
        threshold_of_variance_explanation=config["threshold_encodings"],
    )
    cv_dict = hp_tuning_helper.dimensionality_reduction_on_robust_rescaled(
        cv_dict=cv_dict,
        pca_on_other_features=config["peform_dimensionality_reduction"],
        threshold_of_variance_explanation=config["threshold_robust"],
        target_column=config["target_column"],
    )
    # Hyperparameter_tuning
    cv_dict = hp_tuning_helper.hyperparameter_tuning(
        cv_dict=cv_dict,
        target_column=config["target_column"],
        n_estimators=config["n_estimators"],
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        num_leaves=config["num_leaves"],
        n_jobs=config["n_jobs"],
    )
    cv_dict = hp_tuning_helper.transform_pandas_to_polars(cv_dict=cv_dict)
    # Reverse rescaling
    cv_dict = hp_tuning_helper.reverse_rescaling_of_target_and_predictions(
        cv_dict=cv_dict,
        target_rescaling_parameters=rescale_parameter_of_target_dict,
        target_column=config["target_column"],
    )
    # Calculate absolute error and save predictions
    cv_dict = hp_tuning_helper.calculate_absolute_error(
        cv_dict=cv_dict, target_column=config["target_column"], run_dir=run_dir
    )
    return cv_dict
