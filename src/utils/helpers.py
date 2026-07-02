from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models" / "saved"
DATA_DIR = ROOT_DIR / "data"


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_path_exists(path: Path | str, description: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def load_pickle(path: Path | str) -> Any:
    path = Path(path)
    ensure_path_exists(path, "Pickle file")
    return joblib.load(path)


def save_pickle(obj: Any, path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(obj, path)


def load_json(path: Path | str) -> Any:
    path = Path(path)
    ensure_path_exists(path, "JSON file")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def format_prediction_response(model_name: str, prediction: float, metric: str) -> dict:
    return {
        "model": model_name,
        "prediction": round(float(prediction), 3),
        "metric": metric,
        "status": "success",
    }