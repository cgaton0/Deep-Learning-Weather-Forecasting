import logging

import numpy as np
import pandas as pd

from src.data.download_data import download_jena_data
from src.preprocessing.clean import clean_columns
from src.preprocessing.resampling import downsample_df
from src.preprocessing.scaling import scaler_minmax_df, save_scaler
from src.preprocessing.splits import create_splits_df
from src.preprocessing.windowing import create_windows_df
from src.utils import ensure_dir, project_path

logging.basicConfig(level=logging.INFO)


def load_raw_data() -> pd.DataFrame:
    """Carga el archivo CSV del dataset Jena Climate en un DataFrame.
    Descarga el dataset si aún no existe
    """

    csv_path = download_jena_data()

    logging.info(f"Loading raw CSV: {csv_path}")
    df = pd.read_csv(
        csv_path,
        index_col="Date Time",
        parse_dates=["Date Time"],
        date_format="%d.%m.%Y %H:%M:%S",
    )

    return df


def make_dataset(
    downsample_time="1h",
    test_ratio=0.15,
    val_ratio=0.15,
    window_size=72,
    target_size=24,
):
    """
    Ejecuta el pipeline completo:
    1) Descargar datos
    2) Limpiar columnas + selección de variables
    3) Downsampling
    4) Splits
    5) Escalado train/val/test
    6) Windowing
    7) Guardado en /data/processed
    """

    logging.info("Loading raw dataset...")
    df = load_raw_data()

    logging.info("Cleaning columns...")
    df = clean_columns(df)

    logging.info(f"Downsampling with rule: {downsample_time}")
    df = downsample_df(df, time=downsample_time)

    logging.info("Creating splits...")
    train_df, val_df, test_df = create_splits_df(df, test_ratio, val_ratio)

    logging.info("Scaling splits...")
    train_scaled, val_scaled, test_scaled, scaler = scaler_minmax_df(
        train_df, val_df, test_df
    )

    logging.info("Creating windows...")
    x_train, y_train = create_windows_df(train_scaled, window_size, target_size)
    x_val, y_val = create_windows_df(val_scaled, window_size, target_size)
    x_test, y_test = create_windows_df(test_scaled, window_size, target_size)

    # ---------- SAVE ----------

    processed_dir = project_path("data", "processed")
    ensure_dir(processed_dir)

    # Guardar conjuntos de datos.
    train_df.to_parquet(processed_dir / "train_raw.parquet")
    val_df.to_parquet(processed_dir / "val_raw.parquet")
    test_df.to_parquet(processed_dir / "test_raw.parquet")

    train_scaled.to_parquet(processed_dir / "train_scaled.parquet")
    val_scaled.to_parquet(processed_dir / "val_scaled.parquet")
    test_scaled.to_parquet(processed_dir / "test_scaled.parquet")

    np.save(processed_dir / "x_train.npy", x_train)
    np.save(processed_dir / "y_train.npy", y_train)
    np.save(processed_dir / "x_val.npy", x_val)
    np.save(processed_dir / "y_val.npy", y_val)
    np.save(processed_dir / "x_test.npy", x_test)
    np.save(processed_dir / "y_test.npy", y_test)

    # Guardar scaler.
    save_scaler(scaler, processed_dir / "scaler.pkl")

    logging.info("Dataset processed and saved in data/processed/")

    return x_train, y_train, x_val, y_val, x_test, y_test, scaler


if __name__ == "__main__":
    make_dataset()
