import pandas as pd

from config.feature_config import PRODUCT_COL, TARGET_COL

def create_lag_features(
    df,
    group_col=PRODUCT_COL,
    target_col=TARGET_COL,
    lags=(1, 7, 14, 28)
):
    """
    Create lag demand features.
    """

    df = df.copy()

    for lag in lags:
        df[f"lag_{lag}"] = (
            df.groupby(group_col)[target_col]
              .shift(lag)
        )

    return df