from pathlib import Path
import pickle
import numpy as np
from typing import Dict   # ✅ This fixes the Dict error
import random

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
DT_PATH = MODEL_DIR / "decision_tree_model.pkl"
RF_PATH = MODEL_DIR / "random_forest_model.pkl"

def load_model(path: Path):
    if not path.exists():
        print(f"⚠️ Model file not found: {path}. Using random dummy predictions instead.")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

# Load models at import time
decision_tree_model = load_model(DT_PATH)
random_forest_model = load_model(RF_PATH)

def extract_features(cdr: Dict) -> np.ndarray:
    base_features = [
        len(str(cdr.get("caller", ""))) % 10,
        len(str(cdr.get("callee", ""))) % 10,
        float(cdr.get("duration", 0)),
        int(bool(cdr.get("roaming", False))),
        0 if cdr.get("call_type", "MO") == "MO" else 1
    ]
    while len(base_features) < 20:
        base_features.append(0)
    return np.array(base_features).reshape(1, -1)

def is_fraudulent(cdr: Dict) -> Dict:
    # Ensure 'destination' is always filled
    if not cdr.get("destination"):
        cdr["destination"] = cdr.get("callee", "UNKNOWN")

    features = extract_features(cdr)

    try:
        # Decision Tree check
        if decision_tree_model:
            dt_pred = int(decision_tree_model.predict(features)[0])
            if dt_pred == 1:
                if random_forest_model:
                    rf_pred = int(random_forest_model.predict(features)[0])
                    if rf_pred == 1:
                        return {"fraud": True, "model": "decision_tree+random_forest"}
                    else:
                        return {"fraud": False, "model": "decision_tree_failed_rf"}
                else:
                    return {"fraud": True, "model": "decision_tree_only"}

        # Random Forest check
        if random_forest_model:
            rf_pred = int(random_forest_model.predict(features)[0])
            if rf_pred == 1:
                return {"fraud": True, "model": "random_forest_only"}

    except Exception as e:
        print(f"⚠️ Model prediction failed for CDR {cdr.get('call_id')}: {e}")

    # Dummy random fallback
    if random.random() < 0.25:
        return {"fraud": True, "model": "dummy_random"}

    return {"fraud": False, "model": "none"}
