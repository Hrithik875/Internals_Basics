import os
import json
import math
import pickle
import itertools

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "training_data.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── MLflow ────────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("file:///" + os.path.join(BASE_DIR, "mlruns").replace("\\", "/"))
mlflow.set_experiment("gamelag-latency-ms")

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X  = df[["server_region", "concurrent_players", "packet_size_kb", "is_ranked_match"]]
y  = df["latency_ms"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Parameter Grid (exactly as in exam) ──────────────────────────────────────
param_grid = {
    "n_estimators":    [50, 150, 250],
    "max_depth":       [5, 10, 20],
    "min_samples_split": [2, 3, 5]
}

# Build all combinations
keys   = list(param_grid.keys())
values = list(param_grid.values())
combinations = list(itertools.product(*values))
total_trials = len(combinations)
print(f"Total trials: {total_trials}")

# ── KFold ─────────────────────────────────────────────────────────────────────
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# ── Grid Search with Nested MLflow Runs ───────────────────────────────────────
best_mae       = float("inf")
best_cv_mae    = float("inf")
best_params    = {}
best_model_obj = None

with mlflow.start_run(run_name="tuning-gamelag") as parent_run:
    parent_run_id = parent_run.info.run_id

    for combo in combinations:
        params = dict(zip(keys, combo))

        with mlflow.start_run(run_name=str(params), nested=True):
            model = RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                random_state=42
            )

            # 5-fold CV MAE
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=kf,
                scoring="neg_mean_absolute_error"
            )
            cv_mae = float(-cv_scores.mean())

            # Train on full train set → test MAE
            model.fit(X_train, y_train)
            y_pred   = model.predict(X_test)
            test_mae = float(mean_absolute_error(y_test, y_pred))
            test_rmse = float(math.sqrt(mean_squared_error(y_test, y_pred)))

            # Log to MLflow
            mlflow.log_params(params)
            mlflow.log_metric("cv_mae",   cv_mae)
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("test_rmse", test_rmse)

            print(f"Params: {params} | CV MAE: {cv_mae:.4f} | Test MAE: {test_mae:.4f}")

            if test_mae < best_mae:
                best_mae       = test_mae
                best_cv_mae    = cv_mae
                best_params    = params
                best_model_obj = model

# ── Save Best Tuned Model ─────────────────────────────────────────────────────
with open(os.path.join(MODELS_DIR, "best_model.pkl"), "wb") as f:
    pickle.dump(best_model_obj, f)

print(f"\nBest params: {best_params}")
print(f"Best test MAE: {best_mae:.4f} | Best CV MAE: {best_cv_mae:.4f}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
output = {
    "search_type":     "grid",
    "n_folds":         N_FOLDS,
    "total_trials":    total_trials,
    "best_params":     best_params,
    "best_mae":        round(best_mae,    4),
    "best_cv_mae":     round(best_cv_mae, 4),
    "parent_run_name": "tuning-gamelag"
}

out_path = os.path.join(RESULTS_DIR, "step2_s2.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved: {out_path}")