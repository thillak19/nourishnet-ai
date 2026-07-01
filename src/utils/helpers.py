"""
NourishNet AI - Utility Helpers
"""

import json
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def format_prediction_response(model_name: str, prediction: float, metric: str) -> dict:
    return {
        "model": model_name,
        "prediction": round(prediction, 3),
        "metric": metric,
        "status": "success"
    }