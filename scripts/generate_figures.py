"""
Generate figures from saved artifacts (history, metrics, predictions).

Usage:
    python scripts/generate_figures.py           # Save only
    python scripts/generate_figures.py --show   # Save and display
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from src.utils import ensure_dir, project_path, setup_logging
from src.visualizations.plots import (
    plot_horizon_comparison,
    plot_metric_over_horizon,
    plot_random_samples,
    plot_training_history,
)

logger = logging.getLogger(__name__)


@dataclass
class DummyHistory:
    """Lightweight wrapper to mimic tf.keras.callbacks.History."""

    history: Dict[str, Any]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_get_list(d: Dict[str, Any], key: str) -> Optional[Sequence[float]]:
    v = d.get(key)
    return v if isinstance(v, list) else None


def main(show: bool = False) -> None:
    setup_logging(level=logging.INFO)

    outputs_metrics_dir = project_path("outputs", "metrics")
    outputs_predictions_dir = project_path("outputs", "predictions")
    outputs_figures_dir = project_path("outputs", "figures")

    ensure_dir(outputs_figures_dir)

    history_path = outputs_metrics_dir / "history.json"
    metrics_path = outputs_metrics_dir / "metrics.json"
    y_true_path = outputs_predictions_dir / "y_test_unscaled.npy"
    y_pred_path = outputs_predictions_dir / "y_pred_unscaled.npy"

    logger.info("Loading artifacts...")
    history_json = _load_json(history_path)
    hist_dict = history_json.get("history", history_json)
    history = DummyHistory(history=hist_dict)

    metrics = _load_json(metrics_path)
    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)

    logger.info("Generating figures (show=%s)...", show)

    # --- Training curves ---
    plot_training_history(
        history,
        save_path=outputs_figures_dir / "training_history.png",
        show=show,
    )

    # --- Horizon metrics ---
    rmse_h = _maybe_get_list(metrics, "rmse_h")
    mae_h = _maybe_get_list(metrics, "mae_h")

    if rmse_h is not None:
        plot_metric_over_horizon(
            rmse_h,
            "RMSE",
            save_path=outputs_figures_dir / "rmse_over_horizon.png",
            show=show,
        )

    if mae_h is not None:
        plot_metric_over_horizon(
            mae_h,
            "MAE",
            save_path=outputs_figures_dir / "mae_over_horizon.png",
            show=show,
        )

    # --- Selected horizons ---
    horizon = y_true.shape[1]
    for h_idx, name in [(0, "h+1"), (11, "h+12"), (23, "h+24")]:
        if h_idx < horizon:
            plot_horizon_comparison(
                y_true,
                y_pred,
                horizon_index=h_idx,
                save_path=outputs_figures_dir / f"horizon_{name}.png",
                show=show,
            )

    # --- Random samples ---
    plot_random_samples(
        y_true,
        y_pred,
        seed=42,
        save_path=outputs_figures_dir / "random_samples.png",
        show=show,
    )

    logger.info("Figures saved in: %s", outputs_figures_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model evaluation figures.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )

    args = parser.parse_args()
    main(show=args.show)
