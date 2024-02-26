import logging
import datetime
import pathlib

from config import columns
from fcst_logic.tft import data_collection, cross_validation, hp_tuning_helper

logger = logging.getLogger(__name__)


def perform_cross_validation(config: dict[str, int | bool | str | None]):
    # Setup directory structure
    run_dir = (
        pathlib.Path(config["base_dir"])
        / "hp_tuning_tft"
        / f"run_{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"
    )
    run_dir.mkdir(parents=False, exist_ok=False)
    import_dir = pathlib.Path(config["base_dir"]) / "imports"

    # Collect data
    data = data_collection.collector(
        import_dir=import_dir,
        target_column=config["target_column"],
        start_date=config["start_date"],
        end_date=config["end_date"],
        filename_backlog=config["filename_backlog"],
        filename_base_data=config["filename_base_data"],
        filename_sick_leave=config["filename_sick_leave"],
        filename_holiday=config["filename_holidays_36"],
        filename_parcel_amount=config["filename_parcel_amount"],
    )
    # Get cross-validation dates
    cv_dates = cross_validation.dates(
        horizons=config["horizons"],
        n_cv=config["number_cv_splits"],
        end_date=data.get_column(f"DATE__{columns.DATE}").max(),
        test_set=True,
    )
    # Connect data and cv_dates
    cv_dict = hp_tuning_helper.connect_data_and_cv_dates(data=data, cv_dates=cv_dates)
    # Attach cluster id
    cv_dict = hp_tuning_helper.attach_cluster_id(
        cv_dict=cv_dict,
        target_column=config["target_column"],
        threshold_for_small_stations=config["threshold_for_small_stations"],
        upper_quantile=config["rescaling_quantiles"][1],
        lower_quantile=config["rescaling_quantiles"][0],
    )
    # Rescaling
    cv_dict = hp_tuning_helper.robust_rescaling_of_features(
        cv_dict=cv_dict,
        target_column=config["target_column"],
        upper_quantile=config["rescaling_quantiles"][1],
        lower_quantile=config["rescaling_quantiles"][0],
    )
    cv_dict = hp_tuning_helper.transform_polars_to_pandas(cv_dict=cv_dict, run_dir=run_dir)
    return hp_tuning_helper.hyperparameter_tuning(
        cv_dict=cv_dict,
        hidden_sizes=config["hidden_sizes"],
        limit_train_batches=config["limit_train_batches"],
        horizons=config["horizons"],
        max_encoder_length=config["max_encoder_length"],
        batch_size=config["batch_size"],
    )
