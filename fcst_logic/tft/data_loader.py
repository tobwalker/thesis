import logging

import polars as pl
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

logger = logging.getLogger(__name__)


def data_loader(data: pl.DataFrame, horizons: int, max_encoder_length: int, batch_size: int):
    training = TimeSeriesDataSet(
        data=data[data.TEST is False],
        time_idx="DATE__TIME_IDX",
        target="TARGET__ABS_SICK_LEAVES",
        group_ids=["FEAT__STATIC_CATEGORICALS__ENCODING__OZ_ZSP"],
        min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=horizons,
        static_categoricals=[col for col in data.columns if col.startswith("FEAT__STATIC_CATEGORICALS__")],
        static_reals=[],
        time_varying_known_categoricals=[
            col for col in data.columns if col.startswith("FEAT__TIME_VARYING_KNOWN_CATEGORICALS__")
        ],
        variable_groups={},
        time_varying_known_reals=[col for col in data.columns if col.startswith("FEAT__TIME_VARYING_KNOWN_REALS")],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[col for col in data.columns if col.startswith("'FEAT__TIME_VARYING_UNKNOWN_REALS")],
        target_normalizer=GroupNormalizer(groups=["FEAT__STATIC_CATEGORICALS__ENCODING__OZ_ZSP"], method="robust"),
        add_relative_time_idx=True,
        add_encoder_length=True,
        add_target_scales=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 100, num_workers=0)
    return training, train_dataloader, val_dataloader
