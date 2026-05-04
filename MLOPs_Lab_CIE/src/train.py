import os
import json
import math
import pickle

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "training_data.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X  = df[["server_region", "concurrent_players", "packet_size_kb", "is_ranked_match"]]
y  = df["latency_ms"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── MLflow Setup ──────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("file:///" + os.path.join(BASE_DIR, "mlruns").replace("\\", "/"))
mlflow.set_experiment("gamelag-latency-ms")

# ── Model Configs ─────────────────────────────────────────────────────────────
models_cfg = [
    {
        "name": "SVR",
        "model": SVR(kernel="rbf", C=1.0, epsilon=0.1),
        "params": {"kernel": "rbf", "C": 1.0, "epsilon": 0.1}
    },
    {
        "name": "RandomForest",
        "model": RandomForestRegressor(n_estimators=100, random_state=42),
        "params": {"n_estimators": 100, "random_state": 42}
    }
]

results = []

for cfg in models_cfg:
    with mlflow.start_run(run_name=cfg["name"]):
        # Log params
        for k, v in cfg["params"].items():
            mlflow.log_param(k, v)

        # Tag
        mlflow.set_tag("priority", "high")

        # Train
        cfg["model"].fit(X_train, y_train)
        y_pred = cfg["model"].predict(X_test)

        # Metrics
        mae  = float(mean_absolute_error(y_test, y_pred))
        rmse = float(math.sqrt(mean_squared_error(y_test, y_pred)))

        mlflow.log_metric("mae",  mae)
        mlflow.log_metric("rmse", rmse)

        # Log model artifact
        mlflow.sklearn.log_model(cfg["model"], artifact_path=cfg["name"])

        print(f"{cfg['name']} — MAE: {mae:.4f}  RMSE: {rmse:.4f}")

        results.append({"name": cfg["name"], "mae": mae, "rmse": rmse,
                        "model_obj": cfg["model"]})

# ── Select Best Model by MAE ──────────────────────────────────────────────────
best = min(results, key=lambda x: x["mae"])

# Save best model to disk (needed for Task 3 & 4)
with open(os.path.join(MODELS_DIR, "best_model.pkl"), "wb") as f:
    pickle.dump(best["model_obj"], f)

print(f"\nBest model: {best['name']}  (MAE={best['mae']:.4f})")

# ── Save JSON ─────────────────────────────────────────────────────────────────
output = {
    "experiment_name": "gamelag-latency-ms",
    "models": [
        {"name": r["name"], "mae": round(r["mae"], 4), "rmse": round(r["rmse"], 4)}
        for r in results
    ],
    "best_model": best["name"],
    "best_metric_name": "mae",
    "best_metric_value": round(best["mae"], 4)
}

out_path = os.path.join(RESULTS_DIR, "step1_s1.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved: {out_path}")