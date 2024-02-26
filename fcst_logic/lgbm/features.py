import logging

import polars as pl

from fcst_logic.lgbm.low_level_functions import features_and_direct_modelling

logger = logging.getLogger("__name__")


def feature_engineering(
    data: pl.DataFrame,
    target_column: str,
    horizons: int,
    target_lags: list[int],
    encoding_lags: list[int],
    other_features_lags: list[int],
) -> pl.DataFrame:
    data = features_and_direct_modelling.lagged_feature_over_all_columns(
        data=data,
        target_column=target_column,
        horizons=horizons,
        target_lags=target_lags,
        encoding_lags=encoding_lags,
        op_covid_lags=other_features_lags,
    )
    for col in [col for col in data.columns if col[-10:] == "__LAGGED_1"]:
        data = features_and_direct_modelling.rolling_mean_on_lagged_1_feature(data=data, column=col, window_size=10)
        data = features_and_direct_modelling.rolling_std_on_lagged_1_feature(data=data, column=col, window_size=10)

    data = features_and_direct_modelling.static_features(data=data)
    data = features_and_direct_modelling.define_target_column(data=data, target_column=target_column)
    data = features_and_direct_modelling.define_date_and_series_columns(data=data)
    data = features_and_direct_modelling.keep_relevant_columns(data=data)
    return features_and_direct_modelling.drop_nans(data=data)
