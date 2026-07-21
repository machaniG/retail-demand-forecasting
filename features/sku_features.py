import pandas as pd
from config.feature_config import PRICE_COL, PRODUCT_COL, TARGET_COL
import numpy as np

def create_sku_features(
    df,
    sku_col=PRODUCT_COL,
    qty_col=TARGET_COL,
    price_col=PRICE_COL
):
    """
    Product-level attributes.
    """

    sku_stats = (
        df.groupby(sku_col)
          .agg(
              sku_mean_qty=(qty_col, "mean"),
              sku_median_qty=(qty_col, "median"),
              sku_std_qty=(qty_col, "std"),
              sku_avg_price=(price_col, "mean")
          )
          .reset_index()
    )

    sku_stats["sku_cv"] = (
        sku_stats["sku_std_qty"]
        / (sku_stats["sku_mean_qty"] + 1e-6)
    )

    return sku_stats