"""
Data cleaning utilities for the Jena Climate dataset.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and select the feature columns used from the Jena Climate dataset.

    Steps:
    - Normalize column names by replacing spaces with underscores.
    - Select a subset of relevant columns.
    - Fix physically impossible values (e.g., negative wind speed is set to 0).
    - Warn if NaN values are present.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the raw Jena Climate data.

    Returns
    -------
    pd.DataFrame
        A cleaned dataframe containing only the selected columns.
    """
    logger.info("Cleaning dataframe...")

    # Normalize column names.
    df = df.copy()
    df.columns = [col.replace(" ", "_") for col in df.columns]

    # Select columns used for training/evaluation.
    selected_columns = [
        "p_(mbar)",
        "T_(degC)",
        "rh_(%)",
        "sh_(g/kg)",
        "wv_(m/s)",
        "wd_(deg)",
    ]
    df = df[selected_columns].copy()

    # Fix invalid values: wind speed cannot be negative.
    invalid_mask = df["wv_(m/s)"] < 0
    invalid_count = int(invalid_mask.sum())
    if invalid_count > 0:
        logger.info("Fixing %d invalid wind-speed values (<0).", invalid_count)
        df.loc[invalid_mask, "wv_(m/s)"] = 0

    # Check for missing values.
    if df.isna().any().any():
        logger.warning("NaN values detected in the dataset.")

    return df
