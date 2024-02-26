import logging
import glob
import holidays
import pathlib

import polars as pl

from config import columns
from config import paths

logger = logging.getLogger(__name__)


def _import_holidays_to_36_station(base_dir: str | pathlib.Path, filename: str) -> pl.DataFrame:
    filepath = pathlib.Path(base_dir) / (filename + ".csv")
    data = pl.read_csv(filepath, separator=";")
    return (
        data.melt(id_vars="OrgOZ", value_vars=[col for col in data.columns if col != "OrgOZ"], value_name=columns.DATE)
        .drop_nulls()
        .rename({"OrgOZ": columns.OZ_ZSP})
        .drop("variable")
        .with_columns(
            pl.lit(1).alias("HOLIDAY"),
            pl.col(columns.DATE).str.strptime(pl.Date, "%d.%m.%Y").alias(columns.DATE),
            pl.col(columns.OZ_ZSP).cast(pl.Utf8).str.slice(0, 12).alias(columns.OZ_ZSP),
        )
    )


def _import_holidays_to_33_station(year: int) -> pl.DataFrame:
    # ToDo: Misses ZB holidays
    holiday_path = sorted(glob.glob(paths.holidays))[-1]
    logger.info(f"Holiday file (33): {holiday_path}")
    return (
        pl.read_csv(holiday_path, separator=";")
        .rename({"ZSPOZ": columns.OZ_ZSP, "FEIERTAG": columns.DATE})
        .select(columns.OZ_ZSP, columns.DATE)
        .with_columns(
            pl.lit(1).alias("HOLIDAY"),
            pl.col(columns.DATE).str.strptime(pl.Date, "%d.%m.%y").alias(columns.DATE),
            pl.col(columns.OZ_ZSP).cast(pl.Utf8).str.slice(0, 12).alias(columns.OZ_ZSP),
        )
    ).filter(pl.col(columns.DATE).dt.year() == year)


def _get_holiday_desc(year: int) -> pl.DataFrame:
    holidays_data = pl.DataFrame()
    for state in [
        "BB",
        "BE",
        "BW",
        "BY",
        "BYP",
        "HB",
        "HE",
        "HH",
        "MV",
        "NI",
        "NW",
        "RP",
        "SH",
        "SL",
        "SN",
        "ST",
        "TH",
    ]:
        temp = (
            pl.DataFrame({value: key for key, value in holidays.Germany(years=year, subdiv=state).items()})
            .transpose(include_header=True)
            .rename({"column": "HOLIDAY_DESC", "column_0": columns.DATE})
        )
        holidays_data = holidays_data.vstack(temp)
    return holidays_data.unique()


def _get_holidays_over_last_years() -> pl.DataFrame:
    combined_dict = {}
    for state in [
        "BB",
        "BE",
        "BW",
        "BY",
        "BYP",
        "HB",
        "HE",
        "HH",
        "MV",
        "NI",
        "NW",
        "RP",
        "SH",
        "SL",
        "SN",
        "ST",
        "TH",
    ]:
        german_holidays = holidays.Germany(years=range(2019, 2028), subdiv=state)
        reversed_german_holidays = {}
        for key, value in german_holidays.items():
            reversed_german_holidays.setdefault(value, []).append(key)
        for holiday in set(combined_dict.keys()).union(reversed_german_holidays.keys()):
            combined_dict[holiday] = list(
                set(combined_dict.get(holiday, [])).union(reversed_german_holidays.get(holiday, []))
            )
    data = []
    for holiday, dates in combined_dict.items():
        for date in dates:
            data.append((holiday, date))
    return (
        pl.DataFrame(data)
        .rename({"column_0": "HOLIDAY_DESC"})
        .with_columns(pl.col("column_1").dt.year().alias("year"))
        .pivot(index="HOLIDAY_DESC", columns="year", values="column_1")
    )


def combine_holidays_with_station_data(year: int, base_dir: str | pathlib.Path, filename: str) -> pl.DataFrame:
    station_33_to_holiday = _import_holidays_to_33_station(year=year)
    station_36_to_holiday = _import_holidays_to_36_station(base_dir=base_dir, filename=filename)
    station_to_holiday = station_33_to_holiday.vstack(station_36_to_holiday)
    holiday_desc = _get_holiday_desc(year=year)
    holiday_dates = _get_holidays_over_last_years()
    holiday_data = station_to_holiday.join(holiday_desc, on=columns.DATE, how="left").join(
        holiday_dates, on="HOLIDAY_DESC", how="left"
    )
    if holiday_data.null_count().transpose().sum().to_numpy() != 0:
        logger.warning(
            f"There are unassociated holidays: {holiday_data.filter(pl.col('HOLIDAY_DESC').is_null()).get_column(columns.DATE).value_counts().to_dicts()}"
        )
    return holiday_data.melt(
        id_vars=columns.OZ_ZSP,
        value_vars=[
            col for col in holiday_data.columns if col not in [columns.OZ_ZSP, columns.DATE, "HOLIDAY", "HOLIDAY_DESC"]
        ],
        value_name=columns.DATE,
        variable_name="YEAR",
    ).drop("YEAR")


def process_holiday_data(holiday_data: pl.DataFrame) -> pl.DataFrame:
    holiday_data = holiday_data.filter(pl.col(columns.DATE).dt.weekday() != 6)
    holiday_data = holiday_data.with_columns(
        pl.lit(1).alias(columns.PUBLIC_HOLIDAYS),
        (pl.col(columns.DATE) - pl.duration(days=pl.col(columns.DATE).dt.weekday() - 1))
        .cast(pl.Datetime)
        .alias(columns.DATE),
    )
    return holiday_data.group_by(columns.OZ_ZSP, columns.DATE).sum()
