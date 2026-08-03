import pandas as pd
from config.feature_config import PRICE_COL, PRODUCT_COL, TARGET_COL, DATE_COL
import numpy as np

def create_sku_features(
    df,
    sku_col=PRODUCT_COL,
    qty_col=TARGET_COL,
    price_col=PRICE_COL,
    date_col = DATE_COL
):
    df = df.copy()
    df = df.sort_values([sku_col, date_col])

    grouped_qty = df.groupby(sku_col)[qty_col]

    df["sku_mean_qty"] = (
        grouped_qty
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    df["sku_median_qty"] = (
        grouped_qty
        .transform(lambda x: x.shift(1).expanding().median())
    )

    df["sku_std_qty"] = (
        grouped_qty
        .transform(lambda x: x.shift(1).expanding().std())
    )

    df["sku_cv"] = (
        df["sku_std_qty"]
        / (df["sku_mean_qty"] + 1e-6)
    )

    df["sku_avg_price"] = (
        df.groupby(sku_col)[price_col]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    return df