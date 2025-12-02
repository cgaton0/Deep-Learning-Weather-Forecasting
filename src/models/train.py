import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.utils import project_path, ensure_dir


def set_seeds(seed):
    """Fija semillas para reproducibilidad."""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def get_callbacks(model_name, patience=10):
    """Devuelve los callbacks usados durante el entrenamiento."""
    models_dir = project_path("models")
    ensure_dir(models_dir)

    return [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ModelCheckpoint(
            models_dir / f"{model_name}.keras", monitor="val_loss", save_best_only=True
        ),
    ]


def train_model(
    model, x_train, y_train, x_val, y_val, batch_size, epochs, seed=42, patience=10
):
    """
    Entrena el modelo con callbacks y semillas fijas.
    Devuelve la historia del entrenamiento y el modelo ya entrenado.
    """

    set_seeds(seed)
    callbacks = get_callbacks(model.name, patience)

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        shuffle=False,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history
