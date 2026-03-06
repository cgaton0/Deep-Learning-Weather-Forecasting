"""
Resampling utilities for time series data.
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def downsample_df(df: pd.DataFrame, freq: Optional[str]) -> pd.DataFrame:
    """
    Downsample a time-indexed DataFrame to a specified frequency.

    The function resamples the data using the mean aggregation and
    applies backward fill to handle potential missing values created
    during resampling.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a DatetimeIndex.
    freq : str or None
        Resampling frequency string (e.g., "1H", "6H", "1D").
        If None, the original dataframe is returned unchanged.

    Returns
    -------
    pd.DataFrame
        Resampled and backward-filled dataframe.
    """
    if freq is None:
        logger.info("No resampling frequency provided. Returning original dataframe.")
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "DataFrame index must be a pandas.DatetimeIndex for resampling."
        )

    logger.info("Resampling dataframe to frequency: %s", freq)

    df_resampled = df.resample(freq).mean().bfill()

    return df_resampled
