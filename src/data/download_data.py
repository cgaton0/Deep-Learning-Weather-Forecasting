import logging
import zipfile

from keras.utils import get_file

from src.utils import ensure_dir, project_path

URL = "https://storage.googleapis.com/download.tensorflow.org/data/jena_climate_2009_2016.csv.zip"
ZIP_NAME = "jena_climate_2009_2016.csv.zip"
CSV_NAME = "jena_climate_2009_2016.csv"

logging.basicConfig(level=logging.INFO)


def download_jena_data():
    """
    Descarga y extrae el dataset Jena Climate dentro de data/raw/.
    Primero comprueba si ya existe.
    """

    # Carpeta de destino para el CSV extraído.
    raw_dir = project_path("data", "raw")
    ensure_dir(raw_dir)

    csv_path = raw_dir / CSV_NAME

    # Comprobar si ya existe el archivo.
    if csv_path.exists():
        logging.info(f"Dataset already present: {csv_path}")
        return csv_path

    # Descargar ZIP via Keras caché.
    logging.info("Downloading Jena Climate dataset...")
    zip_cached_path = get_file(fname="jena_climate_2009_2016.csv.zip", origin=URL)

    logging.info(f"ZIP downloaded at: {zip_cached_path}")

    # Extraemos el ZIP.
    logging.info("Extracting ZIP...")
    with zipfile.ZipFile(zip_cached_path) as z:
        z.extractall(raw_dir)

    logging.info(f"Dataset extracted into: {raw_dir}")

    return csv_path


if __name__ == "__main__":
    download_jena_data()
