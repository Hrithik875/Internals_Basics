import os
import json
import pickle

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "training_data.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── MLflow Setup ──────────────────────────────────────────────────────────────
TRACKING_URI = "file:///" + os.path.join(BASE_DIR, "mlruns").replace("\\", "/")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("gamelag-latency-ms")

client = MlflowClient(tracking_uri=TRACKING_URI)

# ── Load Data + Best Model ────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X  = df[["server_region", "concurrent_players", "packet_size_kb", "is_ranked_match"]]
y  = df["latency_ms"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with open(os.path.join(MODELS_DIR, "best_model.pkl"), "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)
mae    = float(mean_absolute_error(y_test, y_pred))

# ── Log model in a new MLflow run and register it ────────────────────────────
REGISTERED_NAME = "gamelag-latency-ms-predictor"

with mlflow.start_run(run_name="register-best-model") as run:
    run_id = run.info.run_id

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("mae", mae)
    mlflow.set_tag("priority", "high")

    # Log and register in one step
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=REGISTERED_NAME
    )

print(f"Run ID: {run_id}")

# ── Get the version that was just registered ──────────────────────────────────
import time
time.sleep(2)  # give MLflow a moment to register

versions = client.get_latest_versions(REGISTERED_NAME)
latest   = sorted(versions, key=lambda v: int(v.version))[-1]
version  = int(latest.version)

print(f"Registered model: {REGISTERED_NAME}")
print(f"Version: {version}")
print(f"MAE: {mae:.4f}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
output = {
    "registered_model_name": REGISTERED_NAME,
    "version":               version,
    "run_id":                run_id,
    "source_metric":         "mae",
    "source_metric_value":   round(mae, 4)
}

out_path = os.path.join(RESULTS_DIR, "step4_s6.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved: {out_path}")
print(json.dumps(output, indent=2))