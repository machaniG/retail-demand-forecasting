import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(    
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/etl.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


UNIT_PRICE_COL = "UnitPrice"

# Upstream helper functions to prepare transaction data for feature building. 
# add_transaction_type adds a 'transaction_type' column based on the sign of 'Quantity'.
def add_transaction_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['transaction_type'] = np.select(
        [
            df['Quantity'] > 0,
            df['Quantity'] < 0
        ],
        [
            'purchase',
            'return'
        ],
        default='other'
    )

    return df


# add_total_price ensures a 'TotalPrice' column exists, calculated as Quantity * UnitPrice (or a fallback price column).
def add_total_price(df: pd.DataFrame, price_col: str = UNIT_PRICE_COL) -> pd.DataFrame:
    """Ensure a TotalPrice column exists (Quantity * UnitPrice).

    This is an upstream helper so downstream feature builders can rely on
    `TotalPrice` being present.
    """
    df = df.copy()

    # if TotalPrice exists already, leave it
    if 'TotalPrice' in df.columns:
        return df

    # prefer the provided price_col; fall back to common alternatives
    if price_col in df.columns:
        df['TotalPrice'] = df['Quantity'] * df[price_col]
    elif 'Price' in df.columns:
        df['TotalPrice'] = df['Quantity'] * df['Price']
    elif UNIT_PRICE_COL in df.columns:
        # use the canonical UNIT_PRICE_COL constant 
        # string so callers who change the constant are respected here
        df['TotalPrice'] = df['Quantity'] * df[UNIT_PRICE_COL]
    else:
        # No suitable price column found — warn and fall back to zeros.
        # the warning is logged so operators can detect and fix upstream data issues.
        logger.warning(
            "No price column '%s' found and no fallback 'Price'/'UnitPrice' columns present. "
            "Falling back to TotalPrice=0 for all rows. Please pass `price_col` to build_customer_features or add a price column to the data.",
            price_col,
        )
        df['TotalPrice'] = 0

    return df
