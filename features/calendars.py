import pandas as pd

from config.feature_config import DATE_COL

def create_calendar_features(df, date_col=DATE_COL):
    """
    Create calendar-based features.

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()

    df[date_col] = pd.to_datetime(df[date_col])

    df["day_of_week"] = df[date_col].dt.dayofweek
    df["day_of_month"] = df[date_col].dt.day
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df