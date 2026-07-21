import holidays
import numpy as np
import pandas as pd
from config.feature_config import DATE_COL


def create_holiday_features(df, date_col=DATE_COL):
    """
    Create holiday-based features.

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()
    dates = pd.to_datetime(df[date_col])
    years = dates.dt.year.unique().tolist()
    years = sorted(set(years + [min(years) - 1, max(years) + 1]))

    uk_holidays = holidays.UK(years=years)
    date_only = dates.dt.date

    df["is_holiday"] = date_only.isin(uk_holidays).astype(int)
    df["holiday_name"] = [uk_holidays.get(day, "") for day in date_only]
    # Convert the holiday string column into a pandas Category
    df["holiday_name"] = df["holiday_name"].astype("category")

    holiday_dates = np.array(sorted(uk_holidays.keys()), dtype="datetime64[D]")
    date_values = dates.dt.normalize().values.astype("datetime64[D]")
    idx = np.searchsorted(holiday_dates, date_values)

    next_holidays = [
        holiday_dates[i] if i < len(holiday_dates) else np.datetime64("NaT")
        for i in idx
    ]
    prev_holidays = [
        holiday_dates[i - 1] if i > 0 else np.datetime64("NaT")
        for i in idx
    ]

    next_holidays = pd.to_datetime(next_holidays)
    prev_holidays = pd.to_datetime(prev_holidays)

    df["days_to_next_holiday"] = (next_holidays - dates).dt.days
    df["days_since_previous_holiday"] = (dates - prev_holidays).dt.days

    return df
