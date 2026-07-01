"""
NourishNet AI - Delivery Time Prediction
Compares classical ML (Linear Regression, Random Forest, Gradient Boosting)
against a Deep Learning model (Keras MLP) for regression on delivery time.
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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path("data/raw/orders.csv")
MODEL_DIR = Path("models/saved")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_FEATURES = [
    "distance_km", "restaurant_prep_time_min", "hour_of_day",
    "is_peak_hour", "order_value_inr", "customer_total_past_orders",
    "customer_is_premium", "customer_complaints_90d",
]
CATEGORICAL_FEATURES = ["cuisine_type", "weather", "traffic_level", "vehicle_type"]
TARGET = "delivery_time_min"


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name:25s} | MAE: {mae:6.2f} min | RMSE: {rmse:6.2f} min | R2: {r2:.3f}")
    return {"model": name, "mae": round(mae, 3), "rmse": round(rmse, 3), "r2": round(r2, 3)}


def train_classical_models(X_train, X_test, y_train, y_test):
    results = []
    preprocessor = build_preprocessor()

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42),
    }

    best_model = None
    best_r2 = -np.inf
    best_name = None

    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        result = evaluate(name, y_test, preds)
        results.append(result)

        if result["r2"] > best_r2:
            best_r2 = result["r2"]
            best_model = pipe
            best_name = name

    joblib.dump(best_model, MODEL_DIR / "delivery_time_best_classical.pkl")
    print(f"\nBest classical model: {best_name} (saved)")
    return results, best_name


def train_deep_model(X_train, X_test, y_train, y_test):
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        print("TensorFlow not installed — skipping DL comparison. Run: pip install tensorflow")
        return None

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
        X_test_proc = X_test_proc.toarray()

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train_proc.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    early_stop = keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    model.fit(
        X_train_proc, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0,
    )

    preds = model.predict(X_test_proc, verbose=0).flatten()
    result = evaluate("DeepLearning_MLP", y_test, preds)

    model.save(MODEL_DIR / "delivery_time_dl_model.keras")
    joblib.dump(preprocessor, MODEL_DIR / "delivery_time_dl_preprocessor.pkl")
    return result


if __name__ == "__main__":
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}\n")
    print("--- Classical ML Models ---")
    classical_results, best_classical = train_classical_models(X_train, X_test, y_train, y_test)

    print("\n--- Deep Learning Model ---")
    dl_result = train_deep_model(X_train, X_test, y_train, y_test)

    all_results = classical_results + ([dl_result] if dl_result else [])
    with open(MODEL_DIR / "delivery_time_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nAll results saved to models/saved/delivery_time_results.json")