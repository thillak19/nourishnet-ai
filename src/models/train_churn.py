from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

CUISINES = [
    "Indian",
    "Chinese",
    "Italian",
    "Fast Food",
    "South Indian",
    "Continental",
    "Desserts",
    "Beverages",
]
WEATHER = ["Sunny", "Rain", "Storm", "Cloudy"]
TRAFFIC = ["Low", "Medium", "High"]
VEHICLES = ["Bike", "Scooter", "Bicycle"]
FOOD_ITEMS = [
    "Paneer Tikka",
    "Margherita Pizza",
    "Sushi Roll",
    "Chocolate Donut",
    "Veg Fried Rice",
    "Ice Cream Cone",
]
USER_IDS = [f"user_{i:03d}" for i in range(1, 101)]

random.seed(42)


def generate_order_row(order_id: int) -> dict:
    cuisine = random.choice(CUISINES)
    weather = random.choice(WEATHER)
    traffic_level = random.choice(TRAFFIC)
    vehicle_type = random.choice(VEHICLES)
    is_peak_hour = random.random() < 0.35
    distance_km = round(random.uniform(0.5, 12.0), 2)
    restaurant_prep_time_min = random.randint(5, 30)
    order_value_inr = round(random.uniform(100, 1500), 2)
    customer_total_past_orders = random.randint(0, 75)
    customer_is_premium = int(random.random() < 0.25)
    customer_complaints_90d = random.randint(0, 3)

    weather_penalty = {"Sunny": 0, "Cloudy": 4, "Rain": 6, "Storm": 14}[weather]
    traffic_penalty = {"Low": 0, "Medium": 7, "High": 15}[traffic_level]
    peak_penalty = 8 if is_peak_hour else 0

    delivery_time_min = round(
        10
        + 2.5 * distance_km
        + 0.7 * restaurant_prep_time_min
        + weather_penalty
        + traffic_penalty
        + peak_penalty
        + random.uniform(-3, 3),
        1,
    )
    customer_rating = random.randint(1, 5)
    churned = int(random.random() < 0.12 + 0.05 * (customer_complaints_90d > 0))

    return {
        "order_id": order_id,
        "user_id": random.choice(USER_IDS),
        "food_name": random.choice(FOOD_ITEMS),
        "distance_km": distance_km,
        "restaurant_prep_time_min": restaurant_prep_time_min,
        "hour_of_day": random.randint(8, 23),
        "is_peak_hour": int(is_peak_hour),
        "order_value_inr": order_value_inr,
        "customer_total_past_orders": customer_total_past_orders,
        "customer_is_premium": customer_is_premium,
        "customer_complaints_90d": customer_complaints_90d,
        "cuisine_type": cuisine,
        "weather": weather,
        "traffic_level": traffic_level,
        "vehicle_type": vehicle_type,
        "delivery_time_min": delivery_time_min,
        "customer_rating": customer_rating,
        "churned": churned,
        "rating": random.randint(1, 5),
    }


def generate_dataset(n: int = 2000, output_file: Path | str | None = None) -> Path:
    output_path = Path(output_file) if output_file is not None else RAW_DIR / "orders.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "order_id",
        "user_id",
        "food_name",
        "distance_km",
        "restaurant_prep_time_min",
        "hour_of_day",
        "is_peak_hour",
        "order_value_inr",
        "customer_total_past_orders",
        "customer_is_premium",
        "customer_complaints_90d",
        "cuisine_type",
        "weather",
        "traffic_level",
        "vehicle_type",
        "delivery_time_min",
        "customer_rating",
        "churned",
        "rating",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for order_id in range(1, n + 1):
            writer.writerow(generate_order_row(order_id))

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic order dataset for NourishNet AI.")
    parser.add_argument("--count", type=int, default=2000, help="Number of orders to generate.")
    parser.add_argument("--output", type=Path, default=RAW_DIR / "orders.csv", help="Output CSV path.")
    args = parser.parse_args()

    output_path = generate_dataset(args.count, args.output)
    print(f"Generated dataset at: {output_path}")


if __name__ == "__main__":
    main()
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "cv"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
MODEL_DIR = ROOT_DIR / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_CLASSES = [
    "pizza",
    "sushi",
    "ice_cream",
    "samosa",
    "fried_rice",
    "donuts",
    "omelette",
]
EPOCHS = 10
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_subset(dataset, selected_indices):
    targets = np.array(dataset.targets)
    mask = np.isin(targets, selected_indices)
    return Subset(dataset, np.where(mask)[0])


def build_model(num_classes: int) -> torch.nn.Module:
    model = torchvision.models.mobilenet_v2(weights=None)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model.to(DEVICE)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def main():
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        raise FileNotFoundError("Expected CV dataset folders: data/cv/train and data/cv/val")

    print(f"Device: {DEVICE}")

    train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(VAL_DIR, transform=transform)

    class_to_idx = train_dataset.class_to_idx
    available_classes = [c for c in SELECTED_CLASSES if c in class_to_idx]
    if not available_classes:
        raise RuntimeError("No selected classes were found in the CV dataset.")

    selected_indices = [class_to_idx[c] for c in available_classes]

    train_loader = DataLoader(
        get_subset(train_dataset, selected_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        get_subset(val_dataset, selected_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(len(available_classes))
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_acc = evaluate(model, val_loader)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "best_food_classifier.pth")

    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Saved best model to {MODEL_DIR / 'best_food_classifier.pth'}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.helpers import ensure_dir, save_json, save_pickle

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "raw" / "orders.csv"
MODEL_DIR = ROOT_DIR / "models" / "saved"
ensure_dir(MODEL_DIR)

NUMERIC_FEATURES = [
    "distance_km",
    "delivery_time_min",
    "customer_rating",
    "customer_total_past_orders",
    "customer_complaints_90d",
    "order_value_inr",
]
CATEGORICAL_FEATURES = ["customer_is_premium"]
TARGET = "churned"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Expected column '{TARGET}' in {DATA_PATH}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def build_preprocessor():
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def evaluate(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)

    print(f"{name:22s} | Acc: {acc:.3f} | Prec: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
    return {
        "model": name,
        "accuracy": round(acc, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "auc": round(auc, 3),
    }


def train_models(X_train, X_test, y_train, y_test):
    preprocessor = build_preprocessor()
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42
        ),
    }

    results = []
    best_model = None
    best_f1 = -1
    best_name = None

    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        result = evaluate(name, y_test, preds, proba)
        results.append(result)

        if result["f1"] > best_f1:
            best_f1 = result["f1"]
            best_model = pipe
            best_name = name

    save_pickle(best_model, MODEL_DIR / "churn_best_model.pkl")
    print(f"\nBest churn model: {best_name} (saved)")

    print("\nConfusion Matrix & Classification Report (best model):")
    preds = best_model.predict(X_test)
    print(confusion_matrix(y_test, preds))
    print(
        classification_report(
            y_test,
            preds,
            target_names=["Retained", "Churned"],
            zero_division=0,
        )
    )

    return results, best_name


def tune_random_forest(X_train, y_train):
    preprocessor = build_preprocessor()
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [8, 10, 12],
        "model__min_samples_split": [2, 5],
    }
    pipe = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(class_weight="balanced", random_state=42)),
        ]
    )
    grid = GridSearchCV(pipe, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV F1: {grid.best_score_:.3f}")
    save_pickle(grid.best_estimator_, MODEL_DIR / "churn_tuned_model.pkl")
    return grid.best_estimator_


if __name__ == "__main__":
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)} | Churn rate (train): {y_train.mean():.3f}\n")

    results, _ = train_models(X_train, X_test, y_train, y_test)

    print("\n--- Hyperparameter Tuning ---")
    tune_random_forest(X_train, y_train)

    save_json(results, MODEL_DIR / "churn_results.json")
    print("\nResults saved to models/saved/churn_results.json")
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