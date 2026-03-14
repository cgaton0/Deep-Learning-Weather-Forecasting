"""
Resampling utilities for time series data.
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def downsample_df(
    df: pd.DataFrame,
    downsample_time: Optional[str],
    aggregation_method: str = "mean",
    missing_method: str = "interpolate",
) -> pd.DataFrame:
    """
    Downsample a time-indexed DataFrame to a specified frequency.

    The function resamples the data using the selected aggregation method
    and applies the selected strategy to handle missing values created
    during resampling.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a DatetimeIndex.
    downsample_time : str or None
        Resampling frequency string (e.g., "1H", "6H", "1D").
        If None, the original dataframe is returned unchanged.
    aggregation_method : str, default="mean"
        Aggregation method used during resampling.
        Supported values are: "mean", "median".
    missing_method : str, default="interpolate"
        Method used to handle missing values after resampling.
        Supported values are: "bfill", "ffill", "interpolate".

    Returns
    -------
    pd.DataFrame
        Resampled dataframe with missing values handled according
        to the selected method.

    Raises
    ------
    TypeError
        If the dataframe index is not a pandas.DatetimeIndex.
    ValueError
        If aggregation or missing_method is not supported.
    """
    if downsample_time is None:
        logger.info("No resampling frequency provided. Returning original dataframe.")
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "DataFrame index must be a pandas.DatetimeIndex for resampling."
        )

    valid_aggregations = {"mean", "median"}
    valid_missing_methods = {"bfill", "ffill", "interpolate"}

    if aggregation_method not in valid_aggregations:
        raise ValueError(
            f"Unsupported aggregation_method '{aggregation_method}'. "
            f"Supported values are: {sorted(valid_aggregations)}."
        )

    if missing_method not in valid_missing_methods:
        raise ValueError(
            f"Unsupported missing_method '{missing_method}'. "
            f"Supported values are: {sorted(valid_missing_methods)}."
        )

    logger.info(
        "Resampling dataframe | freq=%s | aggregation_method=%s | missing_method=%s",
        downsample_time,
        aggregation_method,
        missing_method,
    )

    if aggregation_method == "mean":
        df_resampled = df.resample(downsample_time).mean()
    else:
        df_resampled = df.resample(downsample_time).median()

    if missing_method == "bfill":
        df_resampled = df_resampled.bfill()
    elif missing_method == "ffill":
        df_resampled = df_resampled.ffill()
    else:
        df_resampled = df_resampled.interpolate(method="time")

    return df_resampled
