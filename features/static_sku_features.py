"""
SKU-level static features.

Stats are computed on WEEKLY-aggregated demand rather than raw daily
rows, since inventory/reorder decisions run on a roughly weekly cadence
-- "sold 3/week steadily" and "sold 2 Monday, 1 Tuesday, 0 the rest of
the week" should look like the same SKU profile, which a daily mean
doesn't capture but a weekly one does.

These are fit on TRAINING data only and then applied (transformed)
onto train, test, and any future/inference data. This prevents:

  1. Target leakage: a row's own target value can never contribute to
     its own SKU-level aggregate feature, because the aggregate is
     frozen at fit() time and simply looked up at transform() time.
  2. Train/test leakage: test-period target values can never bleed
     into train features (and vice versa), since the encoder never
     sees test data when it's fit.

Note: fewer independent observations per SKU at weekly granularity
(~13 weeks vs. ~90 days over a typical quarter of history), so
sku_std_weekly_qty / sku_cv will be noisier for short-lived SKUs than
the daily version was.

Usage:
    encoder = SKUFeatureEncoder()
    encoder.fit(train_df)
    train_df = encoder.transform(train_df)
    test_df  = encoder.transform(test_df)

    # persist for production
    import joblib
    joblib.dump(encoder, "sku_feature_encoder.pkl")
"""
import pandas as pd
import numpy as np

from config.feature_config import PRICE_COL, PRODUCT_COL, TARGET_COL, DATE_COL


class SKUFeatureEncoder:
    def __init__(
        self,
        sku_col=PRODUCT_COL,
        qty_col=TARGET_COL,
        price_col=PRICE_COL,
        date_col=DATE_COL,
        freq="W",
    ):
        self.sku_col = sku_col
        self.qty_col = qty_col
        self.price_col = price_col
        self.date_col = date_col
        self.freq = freq
        self.stats_ = None
        self.global_stats_ = None
        self.is_fitted_ = False

    def _weekly_agg(self, df):
        # Sum within each (SKU, week) -- missing/no-sale days contribute
        # 0 either way, so this works whether df is a zero-filled daily
        # calendar or sparse (only rows where a sale occurred).
        return (
            df.groupby([self.sku_col, pd.Grouper(key=self.date_col, freq=self.freq)])
              .agg(
                  weekly_qty=(self.qty_col, "sum"),
                  weekly_price=(self.price_col, "mean"),
              )
              .reset_index()
        )

    def fit(self, df):
        weekly = self._weekly_agg(df)

        stats = (
            weekly.groupby(self.sku_col)
                  .agg(
                      sku_mean_weekly_qty=("weekly_qty", "mean"),
                      sku_median_weekly_qty=("weekly_qty", "median"),
                      sku_std_weekly_qty=("weekly_qty", "std"),
                      sku_avg_price=("weekly_price", "mean"),
                  )
                  .reset_index()
        )
        stats["sku_cv"] = (
            stats["sku_std_weekly_qty"] / (stats["sku_mean_weekly_qty"] + 1e-6)
        )
        self.stats_ = stats

        # Fallback for SKUs never seen at fit time (new products at inference)
        global_mean = weekly["weekly_qty"].mean()
        global_std = weekly["weekly_qty"].std()
        self.global_stats_ = {
            "sku_mean_weekly_qty": global_mean,
            "sku_median_weekly_qty": weekly["weekly_qty"].median(),
            "sku_std_weekly_qty": global_std,
            "sku_avg_price": weekly["weekly_price"].mean(),
            "sku_cv": global_std / (global_mean + 1e-6),
        }
        self.is_fitted_ = True
        return self

    def transform(self, df):
        if not self.is_fitted_:
            raise RuntimeError("SKUFeatureEncoder must be fit() before transform().")

        out = df.merge(self.stats_, on=self.sku_col, how="left")
        if out["sku_mean_weekly_qty"].isna().any():
            for col, fallback in self.global_stats_.items():
                out[col] = out[col].fillna(fallback)
        return out

    def fit_transform(self, df):
        return self.fit(df).transform(df)
