import numpy as np
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from src.preprocessing.scaling import inverse_scaler


def evaluate_predict(model, x_test, y_test):
    """Evalúa el modelo y calcula las predicciones."""

    test_loss, test_rmse = model.evaluate(x_test, y_test, verbose=0)
    y_pred_scaled = model.predict(x_test, verbose=0)

    return test_loss, test_rmse, y_pred_scaled


def unscale_predictions(y_test_scaled, y_pred_scaled, scaler, df_columns):
    """Desescala test y predicciones."""

    y_test_unscaled = inverse_scaler(y_test_scaled, scaler, df_columns)
    y_pred_unscaled = inverse_scaler(y_pred_scaled, scaler, df_columns)

    return y_test_unscaled, y_pred_unscaled


def compute_global_metrics(y_true, y_pred):
    """Calcula RMSE, MAE, correlación y R2 globales."""

    rmse = root_mean_squared_error(y_true.flatten(), y_pred.flatten())
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    corr = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    r2 = r2_score(y_true.flatten(), y_pred.flatten())

    return rmse, mae, corr, r2


def compute_horizon_metrics(y_true, y_pred):
    """RMSE y MAE por horizonte (una métrica por columna)."""

    n_h = y_true.shape[1]

    rmse_h = [root_mean_squared_error(y_true[:, h], y_pred[:, h]) for h in range(n_h)]
    mae_h = [mean_absolute_error(y_true[:, h], y_pred[:, h]) for h in range(n_h)]

    return rmse_h, mae_h


def evaluate_model(model, x_test, y_test, scaler, df_columns):
    """
    Evalúa completamente el modelo:
      - evalúa el modelo
      - calcula predicciones
      - desescalado
      - métricas globales
      - métricas por horizonte
    """

    # --- Evaluación escalada ---
    test_loss, test_rmse, y_pred_scaled = evaluate_predict(model, x_test, y_test)

    # --- Desescalar ---
    y_test, y_pred = unscale_predictions(y_test, y_pred_scaled, scaler, df_columns)

    # --- Métricas globales ---
    rmse_global, mae_global, corr_global, r2_global = compute_global_metrics(
        y_test, y_pred
    )

    # --- Métricas por horizonte ---
    rmse_h, mae_h = compute_horizon_metrics(y_test, y_pred)

    # --- Return clean dict ---
    return {
        "test_loss": test_loss,
        "test_rmse": test_rmse,
        "rmse_global": rmse_global,
        "mae_global": mae_global,
        "corr_global": corr_global,
        "r2_global": r2_global,
        "rmse_h": rmse_h,
        "mae_h": mae_h,
        "y_pred_unscaled": y_pred,
        "y_test_unscaled": y_test,
    }
