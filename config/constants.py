import datetime
from config import columns

BLACK_FRIDAY = [
    datetime.datetime(2020, 11, 27),
    datetime.datetime(2021, 11, 26),
    datetime.datetime(2022, 11, 25),
    datetime.datetime(2023, 11, 24),
    datetime.datetime(2024, 11, 29),
    datetime.datetime(2025, 11, 28),
    datetime.datetime(2026, 11, 27),
    datetime.datetime(2027, 11, 26),
    datetime.datetime(2028, 11, 24),
]

op_features = [
    columns.BACKLOG,
    columns.PARCEL_AMOUNT,
]
