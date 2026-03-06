"""
Evaluation utilities for multi-step time series forecasting models.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src.preprocessing.scaling import inverse_scale_feature

logger = logging.getLogger(__name__)


def evaluate_predict(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """
    Evaluate a model on the scaled test set and generate scaled predictions.

    Parameters
    ----------
    model : Any
        Compiled Keras model (or compatible object exposing evaluate/predict).
    x_test : np.ndarray
        Test inputs of shape (n_samples, window_size, n_features).
    y_test : np.ndarray
        Scaled test targets of shape (n_samples, horizon).

    Returns
    -------
    test_loss : float
        Loss value returned by model.evaluate.
    test_rmse : float
        RMSE metric returned by model.evaluate (if configured in model.compile).
    y_pred_scaled : np.ndarray
        Scaled predictions of shape (n_samples, horizon).
    """
    logger.info("Evaluating model on scaled test set...")
    test_loss, test_rmse = model.evaluate(x_test, y_test, verbose=0)
    y_pred_scaled = model.predict(x_test, verbose=0)
    return float(test_loss), float(test_rmse), y_pred_scaled


def unscale_predictions(
    y_test_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    scaler: Any,
    df_columns: Any,
    target_feature: str = "T_(degC)",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse-scale test targets and predictions for a single target feature.

    Parameters
    ----------
    y_test_scaled : np.ndarray
        Scaled test targets of shape (n_samples, horizon).
    y_pred_scaled : np.ndarray
        Scaled predictions of shape (n_samples, horizon).
    scaler : Any
        Fitted scaler implementing inverse_transform.
    df_columns : Any
        Columns used when fitting the scaler (typically a pd.Index).
    target_feature : str
        Target feature name to inverse-scale.

    Returns
    -------
    y_test_unscaled : np.ndarray
        Unscaled test targets of shape (n_samples, horizon).
    y_pred_unscaled : np.ndarray
        Unscaled predictions of shape (n_samples, horizon).
    """
    y_test_unscaled = inverse_scale_feature(
        y_test_scaled, scaler=scaler, df_columns=df_columns, feature_name=target_feature
    )
    y_pred_unscaled = inverse_scale_feature(
        y_pred_scaled, scaler=scaler, df_columns=df_columns, feature_name=target_feature
    )
    return y_test_unscaled, y_pred_unscaled


def compute_global_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute global metrics by flattening all horizons.

    Metrics
    -------
    - RMSE (global)
    - MAE (global)
    - Pearson correlation (global)
    - R2 (global)

    Notes
    -----
    Flattening mixes horizons; use per-horizon metrics to understand degradation
    over the forecast horizon.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values of shape (n_samples, horizon).
    y_pred : np.ndarray
        Predicted values of shape (n_samples, horizon).

    Returns
    -------
    dict
        Dictionary with keys: rmse, mae, corr, r2.
    """
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    rmse = root_mean_squared_error(yt, yp)
    mae = mean_absolute_error(yt, yp)
    corr = float(np.corrcoef(yt, yp)[0, 1]) if yt.size > 1 else float("nan")
    r2 = float(r2_score(yt, yp)) if yt.size > 1 else float("nan")

    return {"rmse": float(rmse), "mae": float(mae), "corr": corr, "r2": r2}


def compute_horizon_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, List[float]]:
    """
    Compute per-horizon metrics (one value per forecast step).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values of shape (n_samples, horizon).
    y_pred : np.ndarray
        Predicted values of shape (n_samples, horizon).

    Returns
    -------
    dict
        Dictionary with keys: rmse_h, mae_h (lists of length horizon).
    """
    horizon = y_true.shape[1]
    rmse_h = [
        float(root_mean_squared_error(y_true[:, h], y_pred[:, h]))
        for h in range(horizon)
    ]
    mae_h = [
        float(mean_absolute_error(y_true[:, h], y_pred[:, h])) for h in range(horizon)
    ]
    return {"rmse_h": rmse_h, "mae_h": mae_h}


def evaluate_model(
    model: Any,
    x_test: np.ndarray,
    y_test_scaled: np.ndarray,
    scaler: Any,
    df_columns: Any,
    target_feature: str = "T_(degC)",
) -> Dict[str, Any]:
    """
    Full evaluation pipeline for a forecasting model.

    Steps
    -----
    1) Evaluate the model on scaled test data (loss + compiled metrics).
    2) Predict scaled outputs.
    3) Inverse-scale targets and predictions for the selected target feature.
    4) Compute global metrics and per-horizon metrics.

    Parameters
    ----------
    model : Any
        Compiled Keras model.
    x_test : np.ndarray
        Test inputs.
    y_test_scaled : np.ndarray
        Scaled test targets.
    scaler : Any
        Fitted scaler for inverse transformation.
    df_columns : Any
        Columns used during scaling (must include target_feature).
    target_feature : str
        Target feature name to inverse-scale and evaluate.

    Returns
    -------
    dict
        Contains: test_loss, test_rmse, global metrics, per-horizon metrics,
        and unscaled y_true/y_pred arrays.
    """
    test_loss, test_rmse, y_pred_scaled = evaluate_predict(model, x_test, y_test_scaled)

    y_true, y_pred = unscale_predictions(
        y_test_scaled,
        y_pred_scaled,
        scaler=scaler,
        df_columns=df_columns,
        target_feature=target_feature,
    )

    global_metrics = compute_global_metrics(y_true, y_pred)
    horizon_metrics = compute_horizon_metrics(y_true, y_pred)

    logger.info(
        "Evaluation complete | loss=%.4f rmse_scaled=%.4f rmse=%.4f mae=%.4f corr=%.4f r2=%.4f",
        test_loss,
        test_rmse,
        global_metrics["rmse"],
        global_metrics["mae"],
        global_metrics["corr"],
        global_metrics["r2"],
    )

    return {
        "test_loss": test_loss,
        "test_rmse": test_rmse,
        "rmse_global": global_metrics["rmse"],
        "mae_global": global_metrics["mae"],
        "corr_global": global_metrics["corr"],
        "r2_global": global_metrics["r2"],
        "rmse_h": horizon_metrics["rmse_h"],
        "mae_h": horizon_metrics["mae_h"],
        "y_pred_unscaled": y_pred,
        "y_test_unscaled": y_true,
    }
