"""
Model factory functions for the weather forecasting project.

Currently, includes a fixed CNN + BiLSTM architecture for multistep forecasting.
"""

import logging
from typing import Tuple

from keras.layers import (
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Input,
    LSTM,
    ReLU,
)
from keras.models import Model
from keras.models import Sequential

logger = logging.getLogger(__name__)


def model_cnn_bilstm(
    units: int,
    dropout_rate: float,
    input_shape: Tuple[int, int],
    output_size: int,
) -> Model:
    """
    Build a CNN + BiLSTM model for multistep time series forecasting.

    Architecture
    ------------
    Input -> Conv1D -> BatchNorm -> ReLU -> LSTM -> BiLSTM -> Dense -> Output

    Parameters
    ----------
    units : int
        Number of convolution filters and LSTM units.
    dropout_rate : float
        Dropout rate used in LSTM layers and Dense block (0 <= dropout_rate < 1).
    input_shape : tuple of int
        Shape of a single input sample: (window_size, n_features).
    output_size : int
        Forecast horizon (number of steps to predict).

    Returns
    -------
    tensorflow.keras.Model
        A compiled Keras model.

    Raises
    ------
    ValueError
        If hyperparameters are invalid.
    """
    if units <= 0:
        raise ValueError("units must be a positive integer.")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in the range [0.0, 1.0).")
    if len(input_shape) != 2 or input_shape[0] <= 0 or input_shape[1] <= 0:
        raise ValueError(
            "input_shape must be (window_size, n_features) with positive integers."
        )
    if output_size <= 0:
        raise ValueError("output_size must be a positive integer.")

    logger.info(
        "Building model CNN_BiLSTM (units=%d, dropout=%.3f, input_shape=%s, output_size=%d)",
        units,
        dropout_rate,
        input_shape,
        output_size,
    )

    model = Sequential(name="CNN_BiLSTM")
    model.add(Input(shape=input_shape))

    # CNN block
    model.add(Conv1D(filters=units, kernel_size=5, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())

    # Recurrent block
    model.add(LSTM(units, dropout=dropout_rate, return_sequences=True))
    model.add(Bidirectional(LSTM(units, dropout=dropout_rate)))

    # Dense head
    model.add(Dropout(dropout_rate))
    model.add(Dense(units, activation="relu"))
    model.add(Dropout(dropout_rate))

    # Output (multi-step regression)
    model.add(Dense(output_size))

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["root_mean_squared_error"],
    )

    return model
