"""
Train/validation/test splitting utilities for time series data.
"""

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def create_splits_df(
    df: pd.DataFrame, test_ratio: float = 0.15, val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time series dataframe into train, validation, and test sets.

    The split is performed sequentially (no shuffling), preserving
    temporal order — which is essential for time series forecasting.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe sorted by time.
    test_ratio : float
        Proportion of samples assigned to the test split (0 < test_ratio < 1).
    val_ratio : float
        Proportion of samples assigned to the validation split (0 < val_ratio < 1).

    Returns
    -------
    train_df : pd.DataFrame
        Training split.
    val_df : pd.DataFrame
        Validation split.
    test_df : pd.DataFrame
        Test split.

    Raises
    ------
    ValueError
        If ratios are invalid or their sum is >= 1.
    """
    logger.info(
        "Creating sequential splits (val_ratio=%.2f, test_ratio=%.2f)",
        val_ratio,
        test_ratio,
    )

    # Validate ratios
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")

    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    if test_ratio + val_ratio >= 1:
        raise ValueError("The sum of test_ratio and val_ratio must be < 1.")

    df_size = len(df)

    test_size = int(df_size * test_ratio)
    val_size = int(df_size * val_ratio)
    train_size = df_size - test_size - val_size

    if train_size <= 0:
        raise ValueError("Train split would be empty. Adjust ratios.")

    logger.info(
        "Split sizes -> train: %d | val: %d | test: %d",
        train_size,
        val_size,
        test_size,
    )

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size : train_size + val_size]
    test_df = df.iloc[train_size + val_size :]

    return train_df, val_df, test_df
