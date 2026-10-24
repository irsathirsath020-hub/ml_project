"""Shared helpers: exceptions, logging, JSON IO."""
from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np


class MLProjectError(Exception):
    """Base class for all expected, user-actionable project errors."""


class DataLoadError(MLProjectError):
    """The dataset file is missing, unreadable, empty or malformed."""


class SchemaError(MLProjectError):
    """Required columns are missing or contain invalid values."""


class InsufficientDataError(MLProjectError):
    """Too little (valid) data to train or validate reliably."""


class TrainingError(MLProjectError):
    """Model training failed."""


class ModelNotFoundError(MLProjectError):
    """No trained model exists yet."""


class ModelIncompatibleError(MLProjectError):
    """The saved model cannot be loaded or used."""


_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        _CONFIGURED = True
    return logging.getLogger(name)


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        try:
            Path(p).mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:
            raise MLProjectError(f"Cannot create directory {p}: permission denied.") from exc


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_json(obj, path: Path) -> None:
    path = Path(path)
    ensure_dirs(path.parent)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=_json_default)
    except PermissionError as exc:
        raise MLProjectError(f"Cannot write {path}: permission denied.") from exc


def load_json(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
