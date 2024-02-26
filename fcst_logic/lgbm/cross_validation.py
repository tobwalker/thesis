import logging
import datetime

logger = logging.getLogger(__name__)


def dates(
    horizons: int,
    n_cv: int,
    end_date: datetime.datetime,
    test_set: bool = True,
) -> dict[str, tuple[datetime.datetime, datetime.datetime, datetime.datetime]]:
    """
    First date marks end of training (<=). Second date marks start and third date marks end of validation set (>= 2nd date and <= 3rd date).
    """
    end_date = end_date - datetime.timedelta(days=end_date.weekday() + 7)
    cv_dict = {}
    for i in range(n_cv):
        id_name = "test" if i == 0 and test_set else f"cv_{i}"
        cv_dict[id_name] = (
            end_date - datetime.timedelta(days=7 * horizons * 2),
            end_date - datetime.timedelta(days=7 * (horizons - 1)),
            end_date,
        )
        end_date = end_date - datetime.timedelta(days=7 * (horizons))
    logger.info(f"CV dates: {cv_dict}")
    return cv_dict
