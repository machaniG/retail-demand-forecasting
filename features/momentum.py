import pandas as pd

def create_momentum_features(df):
    """
    Demand acceleration features.
    """

    df = df.copy()

    df["momentum_7_28"] = (
        df["rolling_mean_7"]
        / (df["rolling_mean_28"] + 1e-6)
    )

    df["momentum_14_28"] = (
        df["rolling_mean_14"]
        / (df["rolling_mean_28"] + 1e-6)
    )

    return df