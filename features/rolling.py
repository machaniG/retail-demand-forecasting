import pandas as pd
from config.feature_config import PRICE_COL, PRODUCT_COL, TARGET_COL
import numpy as np

def create_rolling_features(
    df,
    group_col=PRODUCT_COL,
    target_col=TARGET_COL,
    price_col=PRICE_COL,
    windows=(7, 14, 28)
):
    """
    Rolling demand statistics.
    """

    df = df.copy()

    for window in windows:

        df[f"rolling_mean_{window}"] = (
            df.groupby(group_col)[target_col]
              .transform(
                  lambda x:
                  x.shift(1)
                   .rolling(window)
                   .mean()
              )
        )

        df[f"rolling_std_{window}"] = (
            df.groupby(group_col)[target_col]
              .transform(
                  lambda x:
                  x.shift(1)
                   .rolling(window)
                   .std()
              )
        )

    return df