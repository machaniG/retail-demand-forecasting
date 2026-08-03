"""
SKU order-frequency diagnostic.

Purpose: identify SKUs with very few distinct selling occasions across
the training window, regardless of how many units those orders were
for. A SKU that sold twice in 12.5 months has almost no learnable
signal for a per-SKU model, independent of whether those two orders
were for 1 unit or 500 units -- that's a volume question (handled by
the slow-moving/velocity check) and frequency is a separate axis.

As with everything else in this pipeline: fit the frequency stats and
the exclusion decision on TRAIN data only, then apply that same
SKU list to test/inference -- deciding "which SKUs are excluded" using
information from the test period would be leakage, same as any other
stat in this pipeline.
"""
import pandas as pd
import numpy as np


def compute_sku_order_frequency(
    df,
    sku_col="StockCode",
    date_col="InvoiceDate",
    invoice_col="InvoiceNo",
):
    """
    One row per SKU with:
      - n_orders: distinct invoices the SKU appeared on
                  (frequency of being ordered, regardless of quantity)
      - n_active_days: distinct calendar days the SKU had any sale
      - first_sale / last_sale: date range the SKU was actually sold in
      - active_span_days: (last_sale - first_sale) in days -- lets you
        tell "sold twice, 2 days apart" apart from "sold twice, 300
        days apart" (very different situations even with the same
        order count)
      - orders_per_active_month: normalizes n_orders by how long the
        SKU has actually existed in the data, rather than assuming
        every SKU had the full 12.5-month window to sell in
    """
    g = df.groupby(sku_col)

    freq = g.agg(
        n_orders=(invoice_col, "nunique"),
        n_active_days=(date_col, lambda x: x.dt.normalize().nunique()),
        first_sale=(date_col, "min"),
        last_sale=(date_col, "max"),
    ).reset_index()

    freq["active_span_days"] = (freq["last_sale"] - freq["first_sale"]).dt.days
    freq["active_span_months"] = (freq["active_span_days"] / 30.44).clip(lower=1)
    freq["orders_per_active_month"] = freq["n_orders"] / freq["active_span_months"]

    return freq


def flag_low_frequency_skus(
    freq_df,
    min_orders=5,
    min_active_days=3,
):
    """
    Simple, explainable threshold: exclude SKUs with fewer than
    `min_orders` distinct orders OR fewer than `min_active_days`
    distinct selling days across the whole training window.

    Two conditions rather than one because a SKU could show up on
    5 different invoices all placed the same day (bulk single-event
    purchase) -- that's still low information for learning a demand
    pattern over time, even though n_orders alone looks fine.
    """
    freq_df = freq_df.copy()
    freq_df["low_frequency"] = (
        (freq_df["n_orders"] < min_orders)
        | (freq_df["n_active_days"] < min_active_days)
    )
    return freq_df


def summarize_frequency_distribution(freq_df):
    """
    Quick diagnostic to help pick a threshold rather than guessing --
    print the distribution of order counts so the cutoff is chosen
    from evidence.
    """
    print("Distribution of n_orders per SKU:")
    print(freq_df["n_orders"].describe(percentiles=[0.05, 0.10, 0.25, 0.5, 0.75, 0.90]))
    print()
    print("Distribution of n_active_days per SKU:")
    print(freq_df["n_active_days"].describe(percentiles=[0.05, 0.10, 0.25, 0.5, 0.75, 0.90]))


def report_exclusion_impact(df, freq_df, sku_col="StockCode", qty_col="Quantity"):
    """
    Same discipline as the other filtering steps: report what fraction
    of SKUs, rows, and TOTAL VOLUME the low-frequency exclusion drops --
    volume share is usually the number that matters most for deciding
    whether the cutoff is reasonable.
    """
    low_freq_skus = set(freq_df.loc[freq_df["low_frequency"], sku_col])

    n_skus_total = freq_df[sku_col].nunique()
    n_skus_excluded = len(low_freq_skus)

    mask = df[sku_col].isin(low_freq_skus)
    rows_excluded = mask.sum()
    volume_excluded = df.loc[mask, qty_col].sum()

    print(f"SKUs excluded: {n_skus_excluded} / {n_skus_total} "
          f"({n_skus_excluded / n_skus_total:.1%})")
    print(f"Rows excluded: {rows_excluded} / {len(df)} "
          f"({rows_excluded / len(df):.1%})")
    print(f"Volume excluded: {volume_excluded} / {df[qty_col].sum()} "
          f"({volume_excluded / df[qty_col].sum():.1%})")

    return low_freq_skus


# --------------------------------------------------------------------
# Example usage (fit on train_raw only, apply the same SKU list to test)
# --------------------------------------------------------------------
# freq_df = compute_sku_order_frequency(train_raw)
# summarize_frequency_distribution(freq_df)
#
# freq_df = flag_low_frequency_skus(freq_df, min_orders=5, min_active_days=3)
# excluded_skus = report_exclusion_impact(train_raw, freq_df)
#
# train_df = train_raw[~train_raw["StockCode"].isin(excluded_skus)]
# test_df  = test_raw[~test_raw["StockCode"].isin(excluded_skus)]


def classify_forecast_tier(
    df,
    sku_col="StockCode",
    date_col="InvoiceDate",
    invoice_col="InvoiceNo",
    weekly_min_active_months_frac=0.8,   # active in ~80%+ of the observed span
    monthly_min_active_months=3,         # at least a handful of active months
):
    """
    Matches forecast granularity to how often a SKU actually generates
    demand, rather than forcing every SKU into the same weekly cadence.

    Tiers:
      - "weekly": sells consistently across most months in the training
        window -> enough signal to support week-level forecasting.
      - "monthly": sells regularly but not consistently enough for
        week-level patterns to be meaningful -> forecast/review at
        monthly granularity instead (still ML-forecastable, just
        coarser -- this is the "financial planning" cadence you
        mentioned, and it's a legitimate tier, not a downgrade).
      - "manual": too sparse for either -- handled by rules/safety
        stock, not a trained model (same bucket as your existing
        slow-moving / spike manual-review logic).

    IMPORTANT: distinct ACTIVE MONTHS, not order count -- 12 orders
    all placed in one month is not "sells every month," it's a
    single active month with high order count. Using n_active_months
    catches that; a raw order-count threshold wouldn't.
    """
    df = df.copy()
    df["_year_month"] = df[date_col].dt.to_period("M")

    span = df.groupby(sku_col)["_year_month"].agg(
        n_active_months="nunique",
        first_month="min",
        last_month="max",
    ).reset_index()

    # total months the SKU COULD have sold in, given its own first/last
    # active month -- not the full dataset span, so a SKU introduced
    # partway through isn't unfairly penalized
    span["observed_span_months"] = (
        (span["last_month"] - span["first_month"]).apply(lambda x: x.n) + 1
    )
    span["active_month_frac"] = span["n_active_months"] / span["observed_span_months"]

    def tier(row):
        if row["active_month_frac"] >= weekly_min_active_months_frac and row["n_active_months"] >= 6:
            return "weekly"
        if row["n_active_months"] >= monthly_min_active_months:
            return "monthly"
        return "manual"

    span["forecast_tier"] = span.apply(tier, axis=1)
    return span.drop(columns=["first_month", "last_month"])


"""
tiers = classify_forecast_tier(train_raw)
print(tiers["forecast_tier"].value_counts())

merged = train_raw.merge(tiers[["StockCode", "forecast_tier"]], on="StockCode")
print(merged.groupby("forecast_tier")["Quantity"].sum() / merged["Quantity"].sum())"""