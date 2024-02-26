import logging
import polars as pl
import datetime
from config import columns

logger = logging.getLogger("__name__")


def _black_friday(year: int) -> datetime.datetime:
    thanksgiving = datetime.datetime(year, 11, 22 + (3 - datetime.datetime(year, 11, 1).weekday() + 7) % 7)
    return thanksgiving + datetime.timedelta(days=1)


def _christmas(year: int) -> datetime.datetime:
    return datetime.datetime(year, 12, 25)


def annotate_weeks_black_friday_to_christmas(data: pl.DataFrame) -> pl.DataFrame:
    data = data.with_columns(pl.col(columns.DATE).dt.year().alias("YEAR"))
    return data.with_columns(
        pl.when(
            (pl.col(columns.DATE) >= pl.col("YEAR").apply(lambda year: _black_friday(year)))
            & (pl.col(columns.DATE) <= pl.col("YEAR").apply(lambda year: _christmas(year)))
        )
        .then(pl.col(columns.DATE).dt.week() - pl.col("YEAR").apply(lambda year: _black_friday(year).isocalendar()[1]))
        .otherwise(None)
        .alias(columns.CHRISTMAS_SEASON)
    )
