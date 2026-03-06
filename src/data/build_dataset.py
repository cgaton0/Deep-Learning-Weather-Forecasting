"""
Dataset building pipeline for the Jena Climate forecasting project.

This module:
- downloads the raw dataset (if needed),
- cleans and selects features,
- optionally downsamples,
- creates sequential train/val/test splits,
- scales splits using a scaler fitted on train only,
- builds sliding windows for supervised learning,
- optionally saves processed artifacts to disk.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from src.data.download_data import download_jena_data
from src.preprocessing.clean import clean_columns
from src.preprocessing.resampling import downsample_df
from src.preprocessing.scaling import load_scaler, save_scaler, scale_splits
from src.preprocessing.splits import create_splits_df
from src.preprocessing.windowing import create_windows_df
from src.utils import ensure_dir, project_path

logger = logging.getLogger(__name__)


def load_raw_data() -> pd.DataFrame:
    """
    Load the Jena Climate CSV into a pandas DataFrame.

    The dataset is downloaded automatically if the CSV is not present yet.

    Returns
    -------
    pd.DataFrame
        Raw dataframe indexed by timestamp.
    """
    csv_path = download_jena_data()
    logger.info("Loading raw CSV: %s", csv_path)

    df = pd.read_csv(
        csv_path,
        index_col="Date Time",
        parse_dates=["Date Time"],
        date_format="%d.%m.%Y %H:%M:%S",
    )

    # Ensure chronological order (important for time series).
    df = df.sort_index()

    return df


def build_dataset(
    downsample_time: Optional[str] = "1h",
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    window_size: int = 72,
    target_size: int = 24,
    target_feature: str = "T_(degC)",
    save: bool = True,
    processed_dir: Optional[Path] = None,
    scaler_path: Optional[Path] = None,
    reuse_scaler: bool = False,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    TransformerMixin,
]:
    """
    Run the full dataset preparation pipeline.

    Steps
    -----
    1) Load raw data (download if needed)
    2) Clean and select feature columns
    3) Optional downsampling
    4) Create sequential train/val/test splits
    5) Scale splits (fit on train only)
    6) Create sliding windows for train/val/test
    7) Optionally save processed artifacts to disk

    Parameters
    ----------
    downsample_time : str or None
        Resampling frequency string (e.g., "1h"). If None, no resampling is applied.
    test_ratio : float
        Proportion of samples assigned to the test split.
    val_ratio : float
        Proportion of samples assigned to the validation split.
    window_size : int
        Number of past timesteps used as input.
    target_size : int
        Forecast horizon.
    target_feature : str
        Name of the target column used for y.
    save : bool
        Whether to persist processed artifacts to disk.
    processed_dir : Path, optional
        Output directory for processed datasets. Defaults to `data/processed/`.
    scaler_path : Path, optional
        Path for saving/loading the scaler. Defaults to `outputs/models/scaler.joblib`.
    reuse_scaler : bool
        If True and scaler_path exists, load and reuse it instead of fitting a new one.

    Returns
    -------
    x_train, y_train, x_val, y_val, x_test, y_test : np.ndarray
        Windowed datasets.
    scaler : TransformerMixin
        Fitted (or loaded) scaler.

    Raises
    ------
    ValueError
        If window_size/target_size are not positive.
    """
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    if target_size <= 0:
        raise ValueError("target_size must be a positive integer.")

    if processed_dir is None:
        processed_dir = project_path("data", "processed")

    if scaler_path is None:
        scaler_path = project_path("outputs", "models", "scaler.joblib")

    logger.info(
        "Building dataset (downsample_time=%s, window_size=%d, target_size=%d, target_feature=%s)",
        downsample_time,
        window_size,
        target_size,
        target_feature,
    )

    df = load_raw_data()

    logger.info("Cleaning columns...")
    df = clean_columns(df)

    logger.info("Downsampling with rule: %s", downsample_time)
    df = downsample_df(df, freq=downsample_time)

    logger.info("Creating sequential splits...")
    train_df, val_df, test_df = create_splits_df(
        df, test_ratio=test_ratio, val_ratio=val_ratio
    )

    logger.info("Scaling splits...")
    if reuse_scaler and Path(scaler_path).exists():
        logger.info("Reusing existing scaler from: %s", scaler_path)
        scaler = load_scaler(scaler_path)
        train_scaled, val_scaled, test_scaled, _ = scale_splits(
            train_df, val_df, test_df, scaler=scaler
        )
    else:
        train_scaled, val_scaled, test_scaled, scaler = scale_splits(
            train_df, val_df, test_df
        )

    logger.info("Creating windows...")
    x_train, y_train = create_windows_df(
        train_scaled, window_size, target_size, feature_name=target_feature
    )
    x_val, y_val = create_windows_df(
        val_scaled, window_size, target_size, feature_name=target_feature
    )
    x_test, y_test = create_windows_df(
        test_scaled, window_size, target_size, feature_name=target_feature
    )

    if save:
        ensure_dir(processed_dir)

        logger.info("Saving processed splits to: %s", processed_dir)
        train_df.to_parquet(Path(processed_dir) / "train_raw.parquet")
        val_df.to_parquet(Path(processed_dir) / "val_raw.parquet")
        test_df.to_parquet(Path(processed_dir) / "test_raw.parquet")

        train_scaled.to_parquet(Path(processed_dir) / "train_scaled.parquet")
        val_scaled.to_parquet(Path(processed_dir) / "val_scaled.parquet")
        test_scaled.to_parquet(Path(processed_dir) / "test_scaled.parquet")

        np.save(Path(processed_dir) / "x_train.npy", x_train)
        np.save(Path(processed_dir) / "y_train.npy", y_train)
        np.save(Path(processed_dir) / "x_val.npy", x_val)
        np.save(Path(processed_dir) / "y_val.npy", y_val)
        np.save(Path(processed_dir) / "x_test.npy", x_test)
        np.save(Path(processed_dir) / "y_test.npy", y_test)

        save_scaler(scaler, scaler_path)

        logger.info("Dataset processed and saved successfully.")

    return x_train, y_train, x_val, y_val, x_test, y_test, scaler


if __name__ == "__main__":
    build_dataset()
