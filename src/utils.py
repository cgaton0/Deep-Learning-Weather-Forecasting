"""
Utility functions for path management and logging configuration.
"""

import logging
from pathlib import Path
from typing import Union, Optional

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """
    Ensure that a directory exists. If it does not exist, it is created.

    Parameters
    ----------
    path : str or Path
        Directory path to create.

    Returns
    -------
    Path
        Path object of the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_root() -> Path:
    """
    Return the project root directory.

    Assumes this file is located under repo/src/.
    """
    return Path(__file__).resolve().parents[1]


def project_path(*parts: str) -> Path:
    """
    Return a Path relative to the project root.

    Parameters
    ----------
    *parts : str
        Path components to join under the project root.

    Returns
    -------
    Path
        Full path under the project root.
    """
    return project_root().joinpath(*parts)


def setup_logging(
    level: int = logging.INFO, log_file: Optional[PathLike] = None
) -> None:
    """
    Configure project-wide logging.

    Parameters
    ----------
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG).
    log_file : str or Path, optional
        If provided, logs will also be written to this file.
    """
    handlers = [logging.StreamHandler()]

    if log_file is not None:
        log_path = Path(log_file)
        ensure_dir(log_path.parent)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
