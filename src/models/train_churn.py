"""
NourishNet AI - Customer Churn Prediction
Binary classification: will a customer churn based on recent order experience
and historical behavior. Compares Logistic Regression, Random Forest, and
Gradient Boosting Classifier.
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

DATA_PATH = Path("data/raw/orders.csv")
MODEL_DIR = Path("models/saved")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_FEATURES = [
    "distance_km", "delivery_time_min", "customer_rating",
    "customer_total_past_orders", "customer_complaints_90d", "order_value_inr",
]
CATEGORICAL_FEATURES = ["customer_is_premium"]
TARGET = "churned"


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def evaluate(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    print(f"{name:22s} | Acc: {acc:.3f} | Prec: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
    return {"model": name, "accuracy": round(acc, 3), "precision": round(prec, 3),
            "recall": round(rec, 3), "f1": round(f1, 3), "auc": round(auc, 3)}


def train_models(X_train, X_test, y_train, y_test):
    preprocessor = build_preprocessor()
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42),
    }

    results = []
    best_model, best_f1, best_name = None, -1, None

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

    joblib.dump(best_model, MODEL_DIR / "churn_best_model.pkl")
    print(f"\nBest churn model: {best_name} (saved)")

    print("\nConfusion Matrix & Classification Report (best model):")
    preds = best_model.predict(X_test)
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, target_names=["Retained", "Churned"]))

    return results, best_name


if __name__ == "__main__":
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)} | Churn rate (train): {y_train.mean():.3f}\n")

    results, best_name = train_models(X_train, X_test, y_train, y_test)

    with open(MODEL_DIR / "churn_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to models/saved/churn_results.json")