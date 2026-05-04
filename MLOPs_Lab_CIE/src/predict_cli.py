import os
import json
import pickle
import argparse
import numpy as np

# ── Argument Parser ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GameLag Latency Predictor")
parser.add_argument("--server_region",      type=float, required=True)
parser.add_argument("--concurrent_players", type=float, required=True)
parser.add_argument("--packet_size_kb",     type=float, required=True)
parser.add_argument("--is_ranked_match",    type=float, required=True)
args = parser.parse_args()

# ── Load Model ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ── Predict ───────────────────────────────────────────────────────────────────
features = np.array([[
    args.server_region,
    args.concurrent_players,
    args.packet_size_kb,
    args.is_ranked_match
]])

prediction = float(model.predict(features)[0])

# ── Save JSON ─────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

output = {
    "image_name": "gamelag-predictor",
    "image_tag":  "v1",
    "base_image": "python:3.12-slim",
    "test_input": {
        "server_region":      int(args.server_region),
        "concurrent_players": int(args.concurrent_players),
        "packet_size_kb":     args.packet_size_kb,
        "is_ranked_match":    int(args.is_ranked_match)
    },
    "prediction": round(prediction, 4)
}

out_path = os.path.join(RESULTS_DIR, "step3_s3.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))