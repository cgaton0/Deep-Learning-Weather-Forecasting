"""
Plotting utilities for training history and forecasting results.
"""

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from matplotlib import pyplot as plt

from src.utils import ensure_dir, project_path

PathLike = Union[str, Path]


def _finalize_figure(save_path: Optional[PathLike] = None, show: bool = True) -> None:
    """Show the current figure and optionally save it to disk."""
    if save_path is not None:
        out_path = Path(save_path)
        if not out_path.is_absolute():
            out_path = project_path(str(out_path))
        ensure_dir(out_path.parent)
        plt.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_training_history(
    history, save_path: Optional[PathLike] = None, show: bool = True
) -> None:
    """
    Plot training curves (loss and RMSE) from a Keras History object.

    Parameters
    ----------
    history : keras.callbacks.History
        Training history returned by model.fit().
    save_path : str or Path, optional
        If provided, the figure is saved to this path (relative to project root
        if a relative path is given).
    show : bool, default=True
        Whether to display the figure interactively using ``plt.show()``.
        If False, the figure is only saved to disk (if ``save_path`` is provided)
        and then closed.
    """
    hist = history.history

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(hist.get("loss", []), label="Train", marker="x")
    ax[0].plot(hist.get("val_loss", []), label="Validation", marker="x")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()

    ax[1].plot(hist.get("root_mean_squared_error", []), label="Train", marker="x")
    ax[1].plot(
        hist.get("val_root_mean_squared_error", []), label="Validation", marker="x"
    )
    ax[1].set_title("RMSE")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()

    fig.suptitle("Training history", fontsize=16)
    _finalize_figure(save_path, show)


def plot_horizon_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seed: int = 42,
    horizon_index: Optional[int] = None,
    save_path: Optional[PathLike] = None,
    show: bool = True,
) -> None:
    """
    Plot true vs predicted values for a selected forecast horizon across samples.

    Parameters
    ----------
    y_true : np.ndarray
        True targets of shape (n_samples, horizon).
    y_pred : np.ndarray
        Predictions of shape (n_samples, horizon).
    seed : int
        Random seed used when selecting a random horizon (if horizon_index is None).
    horizon_index : int, optional
        Horizon step index to plot (0-based). If None, a random horizon is selected.
    save_path : str or Path, optional
        If provided, the figure is saved to this path.
    show : bool, default=True
        Whether to display the figure interactively using ``plt.show()``.
        If False, the figure is only saved to disk (if ``save_path`` is provided)
        and then closed.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape (n_samples, horizon)."
        )
    if y_true.ndim != 2:
        raise ValueError(
            "y_true and y_pred must be 2D arrays of shape (n_samples, horizon)."
        )

    rng = np.random.default_rng(seed)
    horizon = y_true.shape[1]

    if horizon_index is None:
        horizon_index = int(rng.integers(0, horizon))
    if not 0 <= horizon_index < horizon:
        raise ValueError(f"horizon_index must be between 0 and {horizon - 1}.")

    plt.figure(figsize=(10, 5))
    plt.plot(y_true[:, horizon_index], label="True")
    plt.plot(y_pred[:, horizon_index], label="Predicted")

    plt.xlabel("Sample")
    plt.ylabel("Target value")
    plt.title(f"Forecast horizon step: h+{horizon_index + 1}")
    plt.legend()

    _finalize_figure(save_path, show)


def plot_random_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seed: int = 42,
    n_rows: int = 3,
    n_cols: int = 2,
    save_path: Optional[PathLike] = None,
    show: bool = True,
) -> None:
    """
    Visualize random multistep forecasts for several samples.

    Parameters
    ----------
    y_true : np.ndarray
        True targets of shape (n_samples, horizon).
    y_pred : np.ndarray
        Predictions of shape (n_samples, horizon).
    seed : int
        Random seed for sample selection.
    n_rows : int
        Number of subplot rows.
    n_cols : int
        Number of subplot columns.
    save_path : str or Path, optional
        If provided, the figure is saved to this path.
    show : bool, default=True
        Whether to display the figure interactively using ``plt.show()``.
        If False, the figure is only saved to disk (if ``save_path`` is provided)
        and then closed.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape (n_samples, horizon)."
        )
    if y_true.ndim != 2:
        raise ValueError(
            "y_true and y_pred must be 2D arrays of shape (n_samples, horizon)."
        )

    rng = np.random.default_rng(seed)
    n_samples = y_true.shape[0]

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(12, 10), sharex=True, sharey=True)
    ax = np.asarray(ax)

    for i in range(n_rows):
        for j in range(n_cols):
            idx = int(rng.integers(0, n_samples))
            steps = np.arange(1, y_true.shape[1] + 1)

            ax[i, j].plot(steps, y_true[idx], label="True", marker="o")
            ax[i, j].plot(steps, y_pred[idx], label="Predicted", marker="x")
            ax[i, j].set_title(f"Sample {idx}")

            if i == n_rows - 1:
                ax[i, j].set_xlabel("Horizon step")
            if j == 0:
                ax[i, j].set_ylabel("Target value")

    ax[0, 0].legend()
    fig.suptitle("Random forecast samples", fontsize=16)

    _finalize_figure(save_path, show)


def plot_metric_over_horizon(
    metric: Sequence[float],
    metric_name: str,
    save_path: Optional[PathLike] = None,
    show: bool = True,
) -> None:
    """
    Plot a metric value for each forecast horizon step.

    Parameters
    ----------
    metric : Sequence[float]
        Metric values of length equal to the forecast horizon.
    metric_name : str
        Display name for the metric (e.g., "RMSE", "MAE").
    save_path : str or Path, optional
        If provided, the figure is saved to this path.
    show : bool, default=True
        Whether to display the figure interactively using ``plt.show()``.
        If False, the figure is only saved to disk (if ``save_path`` is provided)
        and then closed.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(metric, marker="o")

    plt.xticks(range(len(metric)), range(1, len(metric) + 1))
    plt.title(f"{metric_name} over forecast horizon")
    plt.xlabel("Horizon step")
    plt.ylabel(metric_name)

    _finalize_figure(save_path, show)
