import pandas as pd
from features.holidays import create_holiday_features
import numpy as np
from .calendars import create_calendar_features
from .easter_christmas import create_christmas_features
from .lag_features import create_lag_features
from .momentum import create_momentum_features
from .rolling import create_rolling_features
from .sku import create_sku_features
from .create_target import create_target
from .product_lifecycle import create_product_lifecycle_features
from .price_ratio import create_price_ratio
from .static_sku_features import SKUFeatureEncoder

from config.feature_config import DATE_COL 

def build_demand_features(df):

    # 1. price ratio
    df = create_price_ratio(df, )

    # 2. Calendar
    df = create_calendar_features(df)

    # 3. Christmas
    df = create_christmas_features(df)

    # 4. Holidays
    df = create_holiday_features(df)

    # 5. Product lifecycle
    df = create_product_lifecycle_features(df)

    # 6. Demand history
    df = create_lag_features(df)

    # 7. Demand history aggregates
    df = create_rolling_features(df)

    # 8. Derived demand dynamics
    df = create_momentum_features(df)

    # 9. SKU attributes with shift(1)
    df = create_sku_features(df)

    # 10. Static SKU attributes (fit on train, transform everywhere)
     
    #df["sku_expanding_mean_qty"] = df["sku_expanding_mean_qty"].fillna(df["sku_mean_weekly_qty"])

    # 11. Target
    df = create_target(df)

    return df