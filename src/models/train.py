"""
Training utilities for Keras models.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, History
from keras.models import Model

from src.utils import ensure_dir, project_path

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def set_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value used for Python, NumPy, and TensorFlow RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_callbacks(
    model_name: str,
    patience: int = 10,
    checkpoint_dir: Optional[PathLike] = None,
) -> List[Callback]:
    """
    Create training callbacks (early stopping + the best checkpoint).

    Parameters
    ----------
    model_name : str
        Name used for the checkpoint filename.
    patience : int
        Number of epochs with no improvement after which training is stopped.
    checkpoint_dir : str or Path, optional
        Directory where the best model checkpoint will be saved.
        Defaults to `outputs/models/`.

    Returns
    -------
    list of Callback
        List containing EarlyStopping and ModelCheckpoint.
    """
    if checkpoint_dir is None:
        checkpoint_dir = project_path("outputs", "models")

    checkpoint_dir = Path(checkpoint_dir)
    ensure_dir(checkpoint_dir)

    ckpt_path = checkpoint_dir / f"{model_name}.keras"

    logger.info(
        "Using callbacks: EarlyStopping(patience=%d), ModelCheckpoint(%s)",
        patience,
        ckpt_path,
    )

    return [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ModelCheckpoint(filepath=ckpt_path, monitor="val_loss", save_best_only=True),
    ]


def _history_to_dict(history: History) -> Dict[str, Any]:
    """Convert a Keras History object into a JSON-serializable dict."""
    return {
        "params": getattr(history, "params", {}),
        "epoch": getattr(history, "epoch", []),
        "history": getattr(history, "history", {}),
    }


def save_history(history: History, path: PathLike) -> Path:
    """
    Save training history to disk as JSON.

    Parameters
    ----------
    history : History
        Keras History object returned by model.fit().
    path : str or Path
        Path relative to project root or absolute path.

    Returns
    -------
    Path
        Absolute path of the saved history JSON file.
    """
    out_path = Path(path)
    if not out_path.is_absolute():
        out_path = project_path(str(out_path))

    ensure_dir(out_path.parent)
    data = _history_to_dict(history)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    logger.info("Training history saved to: %s", out_path)
    return out_path


def train_model(
    model: Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    epochs: int,
    seed: int = 42,
    patience: int = 10,
    checkpoint_dir: Optional[PathLike] = None,
    verbose: int = 1,
) -> Tuple[Model, History]:
    """
    Train a Keras model using fixed seeds and standard callbacks.

    Parameters
    ----------
    model : Model
        Compiled model to train.
    x_train, y_train : np.ndarray
        Training data.
    x_val, y_val : np.ndarray
        Validation data.
    batch_size : int
        Batch size.
    epochs : int
        Number of training epochs.
    seed : int
        Random seed for reproducibility.
    patience : int
        Early stopping patience.
    checkpoint_dir : str or Path, optional
        Directory to save best model checkpoint (defaults to `outputs/models/`).
    verbose : int
        Verbosity level passed to model.fit().

    Returns
    -------
    model : Model
        Trained model (with best weights restored by EarlyStopping).
    history : History
        Training history object.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if epochs <= 0:
        raise ValueError("epochs must be a positive integer.")

    set_seeds(seed)
    callbacks = get_callbacks(
        model.name, patience=patience, checkpoint_dir=checkpoint_dir
    )

    logger.info(
        "Training model '%s' (epochs=%d, batch_size=%d, seed=%d, shuffle=%s)",
        model.name,
        epochs,
        batch_size,
        seed,
        False,
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        shuffle=False,  # keep temporal order
        callbacks=callbacks,
        verbose=verbose,
    )

    return model, history
