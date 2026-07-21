import pandas as pd
from config.feature_config import TARGET_COL
import numpy as np

def create_target(
    df,
    target_col=TARGET_COL
):
    """
    Log-transformed demand target.
    """

    df = df.copy()

    df["target"] = np.log1p(df[target_col])

    return df