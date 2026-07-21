import pandas as pd
from config.feature_config import PRICE_COL, PRODUCT_COL
import numpy as np

def create_price_ratio(df, 
                       sku_col=PRODUCT_COL, 
                       price_col=PRICE_COL
):
   """Is the current price cheaper or more expensive than usual for this item?"""
   df["sku_avg_price"] = df.groupby(sku_col)[price_col].transform("mean")
   df["price_ratio"] = df[price_col] / df["sku_avg_price"]
   df = df.drop(["sku_avg_price"], axis =1)

   return df