# prepare.py
"""feature selection module"""
import pandas as pd

def get_feature_columns(
    df,
    drop_cols
):

    return [
        c
        for c in df.columns
        if c not in drop_cols
    ]