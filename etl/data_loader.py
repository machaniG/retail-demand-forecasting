import pandas as pd
from config import PROCESSED_PATH


def load_processed_data() -> pd.DataFrame:
    """
    Load processed transaction-level dataset.
    """
    return pd.read_csv(PROCESSED_PATH, parse_dates=['InvoiceDate'])
