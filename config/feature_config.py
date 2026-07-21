# feature_config.py

DATE_COL = "InvoiceDate"
TARGET_COL = "Quantity"
PRODUCT_COL = "StockCode"
PRICE_COL = "Price"

FEATURE_CONFIG = {

    "drop_cols": [

        "Date",
        "Quantity",
        "target",
        ""

    ],

    "required_cols": [

        "lag_28",
        "rolling_mean_28"

    ]

}

DROP_COLS = ["first_sale_date", "InvoiceDate", "Quantity", "StockCode", "target"]