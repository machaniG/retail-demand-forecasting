import datetime as dt
import pandas as pd

from config.feature_config import DATE_COL


def _calculate_easter(year):
    """Return Easter Sunday for the given year using the Anonymous Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = (19 * a + b - b // 4 - ((b - (b + 8) // 25 + 1) // 3) + 15) % 30
    e = (32 + 2 * (b % 4) + 2 * (c // 4) - d - (c % 4)) % 7
    f = (a + 11 * d + 22 * e) // 451
    month = (d + e - 7 * f + 114) // 31
    day = ((d + e - 7 * f + 114) % 31) + 1
    return dt.date(year, month, day)


def create_christmas_features(df, date_col=DATE_COL):
    """Create retail-focused holiday shopping period features."""

    df = df.copy()
    dates = pd.to_datetime(df[date_col])

    christmas_dates = pd.to_datetime(
        dates.dt.year.astype(str) + "-12-25"
    )
    df["days_to_christmas"] = (christmas_dates - dates).dt.days
    df["christmas_window"] = (
        ((df["days_to_christmas"] >= 0) & (df["days_to_christmas"] <= 40))
        | ((df["days_to_christmas"] < 0) & (df["days_to_christmas"] >= -7))
    ).astype(int)
    df["christmas_countdown"] = df["days_to_christmas"].clip(lower=0)

    easter_dates = pd.to_datetime(
        dates.dt.year.apply(lambda year: _calculate_easter(int(year)))
    )
    df["days_to_easter"] = (easter_dates - dates).dt.days
    df["easter_window"] = (
        (df["days_to_easter"] >= -21)
        & (df["days_to_easter"] <= 7)
    ).astype(int)

    return df