"""
Main entry point for training and evaluating a forecasting model on the Jena Climate dataset.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.data.build_dataset import build_dataset
from src.models.build_model import model_cnn_bilstm
from src.models.evaluate import evaluate_model
from src.models.train import save_history, train_model
from src.preprocessing.scaling import load_scaler
from src.utils import ensure_dir, project_path, setup_logging

logger = logging.getLogger(__name__)

# ----------------------------
# Hyperparameters
# ----------------------------
DOWNSAMPLE_TIME = "1h"
TEST_RATIO = 0.15
VAL_RATIO = 0.15

WINDOW_SIZE = 72
TARGET_SIZE = 24
UNITS = 32
DROPOUT = 0.1
BATCH_SIZE = 128
EPOCHS = 100
SEED = 42
TARGET_FEATURE = "T_(degC)"


def load_windows(processed_dir: Path) -> Tuple[np.ndarray, ...]:
    """
    Load windowed train/val/test arrays from disk.

    Parameters
    ----------
    processed_dir : Path
        Directory containing saved numpy arrays.

    Returns
    -------
    Tuple[np.ndarray, ...]
        x_train, y_train, x_val, y_val, x_test, y_test.
    """
    x_train = np.load(processed_dir / "x_train.npy")
    y_train = np.load(processed_dir / "y_train.npy")

    x_val = np.load(processed_dir / "x_val.npy")
    y_val = np.load(processed_dir / "y_val.npy")

    x_test = np.load(processed_dir / "x_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    return x_train, y_train, x_val, y_val, x_test, y_test


def artifacts_exist(processed_dir: Path, scaler_path: Path) -> bool:
    """
    Check whether required dataset artifacts already exist on disk.
    """
    required = [
        processed_dir / "x_train.npy",
        processed_dir / "y_train.npy",
        processed_dir / "x_val.npy",
        processed_dir / "y_val.npy",
        processed_dir / "x_test.npy",
        processed_dir / "y_test.npy",
        processed_dir / "train_raw.parquet",
    ]
    return all(p.exists() for p in required) and scaler_path.exists()


def _save_metrics(results: Dict[str, Any], out_path: Path) -> None:
    """Save evaluation results (excluding large arrays) to JSON."""
    ensure_dir(out_path.parent)

    serializable: Dict[str, Any] = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            # Large arrays are stored separately (predictions).
            serializable[k] = f"<ndarray shape={v.shape} dtype={v.dtype}>"
        elif isinstance(v, (np.floating, np.integer)):
            serializable[k] = v.item()
        else:
            serializable[k] = v

    out_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def main() -> None:
    setup_logging(level=logging.INFO)

    processed_dir = project_path("data", "processed")
    scaler_path = project_path("outputs", "models", "scaler.joblib")
    outputs_models_dir = project_path("outputs", "models")
    outputs_metrics_dir = project_path("outputs", "metrics")
    outputs_predictions_dir = project_path("outputs", "predictions")

    ensure_dir(outputs_models_dir)
    ensure_dir(outputs_metrics_dir)
    ensure_dir(outputs_predictions_dir)

    # Build dataset if artifacts are missing; otherwise load from disk.
    if not artifacts_exist(processed_dir, scaler_path):
        logger.info("Processed artifacts not found. Building dataset...")

        x_train, y_train, x_val, y_val, x_test, y_test, scaler = build_dataset(
            downsample_time=DOWNSAMPLE_TIME,
            test_ratio=TEST_RATIO,
            val_ratio=VAL_RATIO,
            window_size=WINDOW_SIZE,
            target_size=TARGET_SIZE,
            target_feature=TARGET_FEATURE,
            save=True,
            processed_dir=processed_dir,
            scaler_path=scaler_path,
            reuse_scaler=False,
        )
    else:
        logger.info("Processed artifacts found. Loading from disk...")

        logger.info("Loading windowed datasets from: %s", processed_dir)
        x_train, y_train, x_val, y_val, x_test, y_test = load_windows(processed_dir)

        # Load scaler (load_scaler logs the path internally).
        scaler = load_scaler(scaler_path)

    # Columns are needed to inverse-scale the target feature.
    train_df_path = processed_dir / "train_raw.parquet"
    logger.info("Loading train columns from: %s", train_df_path)
    train_df = pd.read_parquet(train_df_path)

    logger.info("Building model...")
    input_shape = x_train.shape[1:]  # (window_size, n_features)
    output_size = y_train.shape[1]  # horizon
    model = model_cnn_bilstm(
        units=UNITS,
        dropout_rate=DROPOUT,
        input_shape=input_shape,
        output_size=output_size,
    )

    logger.info("Training model...")
    model, history = train_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        seed=SEED,
        checkpoint_dir=outputs_models_dir,
    )

    # Save history
    history_path = outputs_metrics_dir / "history.json"
    save_history(history, history_path)

    logger.info("Evaluating model...")
    results = evaluate_model(
        model=model,
        x_test=x_test,
        y_test_scaled=y_test,
        scaler=scaler,
        df_columns=train_df.columns,
        target_feature=TARGET_FEATURE,
    )

    # Save metrics and predictions
    metrics_path = outputs_metrics_dir / "metrics.json"
    _save_metrics(results, metrics_path)

    np.save(outputs_predictions_dir / "y_pred_unscaled.npy", results["y_pred_unscaled"])
    np.save(outputs_predictions_dir / "y_test_unscaled.npy", results["y_test_unscaled"])

    logger.info("Saved metrics to: %s", metrics_path)
    logger.info("Saved predictions to: %s", outputs_predictions_dir)

    logger.info(
        "Test summary | loss=%.4f rmse_scaled=%.4f rmse=%.4f mae=%.4f corr=%.4f r2=%.4f",
        results["test_loss"],
        results["test_rmse"],
        results["rmse_global"],
        results["mae_global"],
        results["corr_global"],
        results["r2_global"],
    )


if __name__ == "__main__":
    main()
