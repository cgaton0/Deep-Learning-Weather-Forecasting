"""
Scaling utilities for time series datasets.

This module provides:
- Scaling train/val/test splits using a scaler fitted on the training data.
- Inverse scaling for a single target feature when predictions are multistep.
- Persistence helpers for saving/loading scalers.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

from src.utils import ensure_dir, project_path

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def scale_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler: Optional[TransformerMixin] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TransformerMixin]:
    """
    Scale train/validation/test splits using a scaler fitted on the training set.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe.
    val_df : pd.DataFrame
        Validation dataframe.
    test_df : pd.DataFrame
        Test dataframe.
    scaler : sklearn TransformerMixin, optional
        Scaler/transformer implementing fit/transform (e.g., StandardScaler).
        If None, a StandardScaler is used.

    Returns
    -------
    scaled_train_df : pd.DataFrame
        Scaled training dataframe.
    scaled_val_df : pd.DataFrame
        Scaled validation dataframe.
    scaled_test_df : pd.DataFrame
        Scaled test dataframe.
    scaler : sklearn TransformerMixin
        The fitted scaler.
    """
    if scaler is None:
        scaler = StandardScaler()

    logger.info("Fitting scaler on training split: %s", scaler.__class__.__name__)

    scaled_train = scaler.fit_transform(train_df)
    scaled_val = scaler.transform(val_df)
    scaled_test = scaler.transform(test_df)

    scaled_train_df = pd.DataFrame(
        scaled_train, columns=train_df.columns, index=train_df.index
    )
    scaled_val_df = pd.DataFrame(scaled_val, columns=val_df.columns, index=val_df.index)
    scaled_test_df = pd.DataFrame(
        scaled_test, columns=test_df.columns, index=test_df.index
    )

    return scaled_train_df, scaled_val_df, scaled_test_df, scaler


def inverse_scale_feature(
    data_scaled: np.ndarray,
    scaler: TransformerMixin,
    df_columns: pd.Index,
    feature_name: str = "T_(degC)",
) -> np.ndarray:
    """
    Inverse-scale a single feature from scaled multistep predictions.

    This is useful when you predict only one target feature (e.g., temperature),
    but the scaler was fitted on multiple features.

    The input array is expected to be shaped as (n_samples, horizon).

    Parameters
    ----------
    data_scaled : np.ndarray
        Scaled predictions of shape (n_samples, horizon).
    scaler : sklearn TransformerMixin
        Fitted scaler with an inverse_transform method (e.g., StandardScaler).
    df_columns : pd.Index
        Column names used during scaling (must include feature_name).
    feature_name : str
        Name of the feature to inverse-scale.

    Returns
    -------
    np.ndarray
        Inverse-scaled predictions with shape (n_samples, horizon).
    """
    if data_scaled.ndim != 2:
        raise ValueError(
            "data_scaled must be a 2D array of shape (n_samples, horizon)."
        )

    if feature_name not in df_columns:
        raise KeyError(f"feature_name='{feature_name}' not found in df_columns.")

    if not hasattr(scaler, "inverse_transform"):
        raise TypeError("Provided scaler does not implement inverse_transform().")

    feature_idx = int(df_columns.get_loc(feature_name))
    n_samples, horizon = data_scaled.shape

    logger.info(
        "Inverse-scaling feature '%s' (index=%d) for horizon=%d.",
        feature_name,
        feature_idx,
        horizon,
    )

    data_rescaled = np.empty_like(data_scaled, dtype=float)

    # For each horizon step, create a dummy matrix with zeros and place the target
    # feature values in the correct column, then inverse transform and extract it back.
    n_features = len(df_columns)
    dummy = np.zeros((n_samples, n_features), dtype=float)

    for h in range(horizon):
        dummy.fill(0.0)
        dummy[:, feature_idx] = data_scaled[:, h]
        inv = scaler.inverse_transform(dummy)
        data_rescaled[:, h] = inv[:, feature_idx]

    return data_rescaled


def save_scaler(scaler: TransformerMixin, path: PathLike) -> Path:
    """
    Save a fitted scaler to disk using joblib.

    Parameters
    ----------
    scaler : sklearn TransformerMixin
        Fitted scaler to persist.
    path : str or Path
        Path relative to the project root where the scaler will be saved.

    Returns
    -------
    Path
        Absolute path of the saved scaler.
    """
    abs_path = project_path(str(path))
    ensure_dir(abs_path.parent)

    logger.info("Saving scaler to: %s", abs_path)
    joblib.dump(scaler, abs_path)

    return abs_path


def load_scaler(path: PathLike) -> TransformerMixin:
    """
    Load a previously saved scaler from disk.

    Parameters
    ----------
    path : str or Path
        Path relative to the project root where the scaler is stored.

    Returns
    -------
    sklearn TransformerMixin
        Loaded scaler instance.
    """
    abs_path = project_path(str(path))
    logger.info("Loading scaler from: %s", abs_path)

    scaler = joblib.load(abs_path)
    return scaler
