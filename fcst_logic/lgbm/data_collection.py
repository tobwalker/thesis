import logging
import pathlib
import datetime
import polars as pl

from config import columns
from fcst_logic.lgbm.low_level_functions import data_sick_leave_and_employees
from fcst_logic.lgbm.low_level_functions import data_public_holidays
from fcst_logic.lgbm.low_level_functions import data_christmas_season
from fcst_logic.lgbm.low_level_functions import data_covid_api
from fcst_logic.lgbm.low_level_functions import data_operational


logger = logging.getLogger(__name__)


def _import_and_process_sick_leave_and_employee_data(
    target_column: str,
    import_dir: str | pathlib.Path,
    filename_sick_leave: str,
    filename_base_data: str,
    start_date: datetime.datetime | None = None,
    end_date: datetime.datetime | None = None,
) -> pl.DataFrame:
    sick_leave_data = data_sick_leave_and_employees.import_sick_leave_data(
        base_dir=import_dir,
        filename=filename_sick_leave,
        target_column=target_column,
        start_date=start_date,
        end_date=end_date,
    )
    employee_data = data_sick_leave_and_employees.import_base_data(
        base_dir=import_dir,
        filename=filename_base_data,
    )
    sick_leave_data = data_sick_leave_and_employees.preprocess_only_current_stations(
        sick_leave=sick_leave_data,
        base_data=employee_data,
    )
    sick_leave_data = data_sick_leave_and_employees.preprocess_clean_histories(
        sick_leave=sick_leave_data, base_data=employee_data
    )
    sick_leave_data = data_sick_leave_and_employees.pre_process_sick_leave_data(
        sick_leave=sick_leave_data, target_column=target_column
    )
    employee_data = data_sick_leave_and_employees.pre_process_base_data(base_data=employee_data)
    sick_leave_and_employee_data = data_sick_leave_and_employees.merge_sick_leave_data_and_base_data(
        sick_leave=sick_leave_data, base_data=employee_data
    )
    return data_sick_leave_and_employees.aggregate_on_week(
        data=sick_leave_and_employee_data, target_column=target_column
    )


def _import_and_process_public_holiday_data(year: int, base_dir: str | pathlib.Path, filename: str) -> pl.DataFrame:
    holiday_data = data_public_holidays.combine_holidays_with_station_data(
        year=year, base_dir=base_dir, filename=filename
    )
    return data_public_holidays.process_holiday_data(holiday_data=holiday_data)


def _import_and_process_covid_data() -> pl.DataFrame:
    return data_covid_api.retrieve_data()


def _import_and_process_operational_data(
    base_dir: str | pathlib.Path, filename_backlog: str, filename_parcel_amount: str
) -> pl.DataFrame:
    return data_operational.retrieve_data(
        base_dir=base_dir, filename_backlog=filename_backlog, filename_parcel_amount=filename_parcel_amount
    )


def _aggregated_values_of_target(data: pl.DataFrame, target_column: str) -> pl.DataFrame:
    return (
        data.with_columns(
            pl.col(columns.OZ_ZSP).str.slice(0, 8).alias("OZ_ZSPL"),
            pl.col(columns.OZ_ZSP).str.slice(0, 4).alias("OZ_NL"),
        )
        .with_columns(
            pl.col(target_column).mean().over("OZ_ZSPL", columns.DATE).alias(f"{target_column}_ZSPL"),
            pl.col(target_column).mean().over("OZ_NL", columns.DATE).alias(f"{target_column}_NL"),
        )
        .drop("OZ_ZSPL", "OZ_NL")
    )


def _merge_data(
    target_column: str,
    sick_leave_and_employee: pl.DataFrame,
    holiday_data: pl.DataFrame,
    covid_data: pl.DataFrame,
    operational_data: pl.DataFrame,
) -> pl.DataFrame:
    data = sick_leave_and_employee.join(holiday_data, on=[columns.DATE, columns.OZ_ZSP], how="left").with_columns(
        pl.col(columns.PUBLIC_HOLIDAYS).fill_null(0)
    )
    data = data_christmas_season.annotate_weeks_black_friday_to_christmas(data=data).with_columns(
        pl.col(columns.CHRISTMAS_SEASON).fill_null(0)
    )
    data = data.join(covid_data, on=columns.DATE, how="left").with_columns(pl.col(columns.COVID_CASES))
    data = data.join(operational_data, on=[columns.DATE, columns.OZ_ZSP], how="left").filter(
        pl.col(columns.DATE) >= operational_data.get_column(columns.DATE).min()
    )
    logger.info(
        f"Missing entries in operative data: {data.select(columns.BACKLOG, columns.PARCEL_AMOUNT).null_count().to_dicts()}"
    )
    data = data.with_columns(
        pl.col(columns.BACKLOG).fill_null(0),
        pl.col(columns.PARCEL_AMOUNT).fill_null(0),
    )
    if any(value > 0 for value in data.null_count().to_dicts()[0].values()):
        raise ValueError(f"Data contains null values: {data.null_count().to_dicts()}")
    data = _aggregated_values_of_target(data=data, target_column=target_column)
    return data.drop(["YEAR"])


def collector(
    import_dir: str | pathlib.Path,
    target_column: str,
    start_date: str,
    end_date: str,
    filename_sick_leave: str,
    filename_base_data: str,
    filename_backlog: str,
    filename_parcel_amount: str,
    filename_holiday: str,
) -> pl.DataFrame:
    start_date_ = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_date_ = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    year = datetime.datetime.now().year

    sick_leave_and_employee_data = _import_and_process_sick_leave_and_employee_data(
        target_column=target_column,
        start_date=start_date_,
        end_date=end_date_,
        import_dir=import_dir,
        filename_sick_leave=filename_sick_leave,
        filename_base_data=filename_base_data,
    )
    holiday_data = _import_and_process_public_holiday_data(year=year, base_dir=import_dir, filename=filename_holiday)
    covid_data = _import_and_process_covid_data()
    operational_data = _import_and_process_operational_data(
        base_dir=import_dir, filename_backlog=filename_backlog, filename_parcel_amount=filename_parcel_amount
    )
    return _merge_data(
        target_column=target_column,
        sick_leave_and_employee=sick_leave_and_employee_data,
        holiday_data=holiday_data,
        covid_data=covid_data,
        operational_data=operational_data,
    )
