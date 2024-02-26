import logging
import requests

import polars as pl

from config import columns
from config import paths

logger = logging.getLogger(__name__)


def retrieve_data() -> pl.DataFrame:
    response = requests.get(paths.covid_api_adress)
    if response.status_code == 200:
        return (
            pl.DataFrame(response.json()["data"])
            .rename({"date": columns.DATE, "cases": columns.COVID_CASES})
            .with_columns(pl.col(columns.DATE).str.slice(0, 10).str.to_datetime())
            .with_columns(
                (pl.col(columns.DATE) - pl.duration(days=pl.col(columns.DATE).dt.weekday() - 1)).alias(columns.DATE)
            )
            .group_by(columns.DATE)
            .agg(pl.col(columns.COVID_CASES).mean())
        )
    else:
        raise ValueError(f"COVID API access failed with: {response.status_code} - {response.text}")
