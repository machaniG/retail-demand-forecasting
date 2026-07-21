import pandas as pd
import datetime as dt

from config.feature_config import DATE_COL, PRODUCT_COL

def create_product_lifecycle_features(
    df,
    date_col=DATE_COL,
    sku_col=PRODUCT_COL
):
    """Create product lifecycle features based on days since first sale."""

    first_sale = (
        df.groupby(sku_col)[date_col]
          .min()
          .rename("first_sale_date")
    )

    df = df.merge(
        first_sale,
        on=sku_col,
        how="left"
    )

    df["days_since_first_sale"] = (
        df[date_col] -
        df["first_sale_date"]
    ).dt.days

    return df