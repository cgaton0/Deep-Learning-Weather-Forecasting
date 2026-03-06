"""
Windowing utilities for time series forecasting.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_windows_df(
    df: pd.DataFrame, window_size: int, target_size: int, feature_name: str = "T_(degC)"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows (x) and multistep targets (y) from a dataframe.

    The output is intended for supervised learning in time series forecasting:
    - x contains sequences of length `window_size` using all features.
    - y contains the future values of a single target feature for `target_size` steps.

    The function expects the dataframe to be sorted in chronological order.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features (columns) and indexed by time.
    window_size : int
        Number of past timesteps used as input.
    target_size : int
        Forecast horizon (number of future timesteps to predict).
    feature_name : str
        Column name to use as the prediction target.

    Returns
    -------
    x : np.ndarray
        Input windows of shape (n_samples, window_size, n_features), dtype float32.
    y : np.ndarray
        Targets of shape (n_samples, target_size), dtype float32.

    Raises
    ------
    ValueError
        If window_size/target_size are invalid or the dataframe is too short.
    KeyError
        If feature_name is not present in df.columns.
    """
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    if target_size <= 0:
        raise ValueError("target_size must be a positive integer.")
    if feature_name not in df.columns:
        raise KeyError(f"feature_name='{feature_name}' not found in dataframe columns.")

    n_rows = len(df)
    n_samples = n_rows - window_size - target_size + 1
    if n_samples <= 0:
        raise ValueError(
            "Dataframe is too short to create windows with the given window_size "
            f"({window_size}) and target_size ({target_size})."
        )

    n_features = df.shape[1]
    target_idx = int(df.columns.get_loc(feature_name))

    logger.debug(
        "Creating windows: n_rows=%d, window_size=%d, target_size=%d -> n_samples=%d",
        n_rows,
        window_size,
        target_size,
        n_samples,
    )

    # Convert to numpy once for speed.
    values = df.to_numpy()

    x = np.empty((n_samples, window_size, n_features), dtype=np.float32)
    y = np.empty((n_samples, target_size), dtype=np.float32)

    for i in range(n_samples):
        x[i] = values[i : i + window_size]
        y[i] = values[i + window_size : i + window_size + target_size, target_idx]

    return x, y
