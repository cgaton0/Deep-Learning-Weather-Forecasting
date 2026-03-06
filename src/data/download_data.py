"""
Download and extract the Jena Climate dataset into the project data directory.
"""

import logging
import zipfile
from pathlib import Path

from keras.utils import get_file

from src.utils import ensure_dir, project_path

URL = "https://storage.googleapis.com/download.tensorflow.org/data/jena_climate_2009_2016.csv.zip"
ZIP_NAME = "jena_climate_2009_2016.csv.zip"
CSV_NAME = "jena_climate_2009_2016.csv"

logger = logging.getLogger(__name__)


def download_jena_data() -> Path:
    """
    Download and extract the Jena Climate dataset into `data/raw/`.

    The function first checks whether the CSV already exists in the target
    directory. If not, it downloads the ZIP file using Keras' cache mechanism
    and extracts the CSV into `data/raw/`.

    Returns
    -------
    Path
        Path to the extracted CSV file under `data/raw/`.

    Raises
    ------
    FileNotFoundError
        If extraction completes but the CSV file is not found in the target directory.
    """
    raw_dir = project_path("data", "raw")
    ensure_dir(raw_dir)

    csv_path = raw_dir / CSV_NAME
    if csv_path.exists():
        logger.info("Dataset already present: %s", csv_path)
        return csv_path

    logger.info("Downloading Jena Climate dataset from: %s", URL)

    # Download ZIP via Keras cache (returns a filesystem path as string).
    zip_cached_path = Path(get_file(fname=ZIP_NAME, origin=URL))
    logger.info("ZIP downloaded to cache at: %s", zip_cached_path)

    logger.info("Extracting ZIP into: %s", raw_dir)
    with zipfile.ZipFile(zip_cached_path) as zf:
        zf.extractall(raw_dir)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Extraction completed, but '{CSV_NAME}' was not found in {raw_dir}."
        )

    logger.info("Dataset extracted successfully: %s", csv_path)
    return csv_path


if __name__ == "__main__":
    download_jena_data()
