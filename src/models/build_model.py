from tensorflow.keras.layers import (
    Conv1D,
    BatchNormalization,
    ReLU,
    Input,
    LSTM,
    Bidirectional,
    Dropout,
    Dense,
)
from tensorflow.keras.models import Sequential


def model_cnn_bilstm(units, dropout_rate, input_shape, output_shape):
    """
    Construye un modelo CNN + BiLSTM para predicción multihorizon.

    Args:
        units (int): Número de filtros/unidades.
        dropout_rate (float): Ratio de dropout.
        input_shape (tuple): (window_size, num_features)
        output_shape (int): Target_size.
    """

    model = Sequential(name="CNN_BiLSTM")

    # Entrada
    model.add(Input(shape=input_shape))

    # CNN
    model.add(Conv1D(filters=units, kernel_size=5, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())

    # LSTMs
    model.add(LSTM(units, dropout=dropout_rate, return_sequences=True))
    model.add(Bidirectional(LSTM(units, dropout=dropout_rate)))

    # Dense
    model.add(Dropout(dropout_rate))
    model.add(Dense(units, activation="relu"))
    model.add(Dropout(dropout_rate))

    # Salida
    model.add(Dense(output_shape))

    # Compilación
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["root_mean_squared_error"],
    )

    return model
