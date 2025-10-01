from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from fraud_logic import is_fraudulent
import requests
from datetime import datetime
import os
import joblib
import shap
import pandas as pd
import numpy as np
import lime.lime_tabular
import warnings

# ✅ Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ✅ Base directory for models
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ✅ Load models
rf_model_path = os.path.join(MODEL_DIR, "random_forest_model.pkl")
dt_model_path = os.path.join(MODEL_DIR, "decision_tree_model.pkl")

rf_model = joblib.load(rf_model_path)
dt_model = joblib.load(dt_model_path)

# === The model was trained with 20 features, so we must respect that shape ===
REAL_FEATURES = [
    "call_id", "caller", "callee", "duration", "roaming",
    "call_type", "sim", "imei", "imsi", "timestamp", "isFraud"
]

# Pad with 9 dummy features (hidden from explanations)
DUMMY_FEATURES = [f"dummy_{i:02d}" for i in range(9)]
EXPECTED_FEATURES = REAL_FEATURES + DUMMY_FEATURES

# ✅ Lazy SHAP/LIME
shap_explainer = None
lime_explainer = None


def get_shap_explainer():
    global shap_explainer
    if shap_explainer is None:
        shap_explainer = shap.TreeExplainer(rf_model)
    return shap_explainer


def get_lime_explainer():
    global lime_explainer
    if lime_explainer is None:
        dummy_row = pd.DataFrame([[0] * len(EXPECTED_FEATURES)], columns=EXPECTED_FEATURES)
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=dummy_row.values,
            feature_names=EXPECTED_FEATURES,
            class_names=["Legit", "Fraud"],
            mode="classification"
        )
    return lime_explainer


# ✅ FastAPI setup
app = FastAPI(title="Fraud Detection API with Explainable AI")

TELECOM_BACKEND_URL = "http://127.0.0.1:8002"

fraud_api_logs = []
processed_cdrs = []


class CDR(BaseModel):
    call_id: str = None
    caller: str = None
    callee: str = None
    duration: int
    roaming: int
    call_type: str = None
    sim: str = None
    imei: str = None
    imsi: str = None
    timestamp: str = None
    isFraud: int = 0


# -------------------------
# Helper utilities
# -------------------------
def _hash_to_int(s: str, mod: int = 100000):
    if s is None:
        return 0
    try:
        return abs(hash(str(s))) % mod
    except Exception:
        return 0


def _parse_timestamp_to_int(ts: str):
    if not ts:
        return 0
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            dt = datetime.strptime(ts, fmt)
            return int(dt.timestamp())
        except Exception:
            continue
    return _hash_to_int(ts)


def _pretty_value(key: str, val):
    if key == "duration":
        return f"{int(val)}s"
    if key == "roaming":
        return "Yes" if int(val) else "No"
    if key == "isFraud":
        return str(int(val))
    if key == "timestamp":
        try:
            if isinstance(val, (int, float)):
                return datetime.fromtimestamp(int(val)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
        return str(val)
    return str(val)


def map_to_features(cdr_dict: dict):
    human = {
        "call_id": cdr_dict.get("call_id", "") or "",
        "caller": cdr_dict.get("caller", "") or "",
        "callee": cdr_dict.get("callee", "") or "",
        "duration": int(cdr_dict.get("duration", 0)) if cdr_dict.get("duration") is not None else 0,
        "roaming": int(bool(cdr_dict.get("roaming", 0))),
        "call_type": cdr_dict.get("call_type", "") or "",
        "sim": cdr_dict.get("sim", "") or "",
        "imei": cdr_dict.get("imei", "") or "",
        "imsi": cdr_dict.get("imsi", "") or "",
        "timestamp": cdr_dict.get("timestamp", "") or "",
        "isFraud": int(cdr_dict.get("isFraud", 0)),
    }
    return {f: human[f] for f in REAL_FEATURES}


def numeric_features_from_human(human_features: dict):
    numeric = {
        "call_id": _hash_to_int(human_features.get("call_id", ""), mod=1000000),
        "caller": _hash_to_int(human_features.get("caller", "")),
        "callee": _hash_to_int(human_features.get("callee", "")),
        "duration": int(human_features.get("duration", 0)),
        "roaming": int(bool(human_features.get("roaming", 0))),
        "call_type": _hash_to_int(human_features.get("call_type", "")),
        "sim": _hash_to_int(human_features.get("sim", "")),
        "imei": _hash_to_int(human_features.get("imei", "")),
        "imsi": _hash_to_int(human_features.get("imsi", "")),
        "timestamp": _parse_timestamp_to_int(human_features.get("timestamp", "")),
        "isFraud": int(human_features.get("isFraud", 0)),
    }
    ordered_numeric = [numeric[f] for f in REAL_FEATURES]
    ordered_numeric += [0] * len(DUMMY_FEATURES)  # pad for model compatibility

    human_labels = []
    for f in REAL_FEATURES:
        hv = human_features.get(f, "")
        if f in ("duration", "roaming", "timestamp", "isFraud"):
            human_labels.append(f"{f} = {_pretty_value(f, hv)}")
        elif hv == "" or hv is None:
            human_labels.append(f"{f} = <missing>")
        else:
            human_labels.append(f"{f} = {hv}")
    # Add placeholders for dummy features (but won’t be shown)
    for f in DUMMY_FEATURES:
        human_labels.append(f)

    return ordered_numeric, human_labels


def prettify_lime_item(item: str, human_labels_map: dict):
    try:
        parts = item.split(" ", 1)
        feat = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        human = human_labels_map.get(feat, feat)
        return f"{human} {rest}"
    except Exception:
        return item


# -------------------------
# SHAP helper (NEW)
# -------------------------
def _shap_to_1d(shap_values):
    """
    Normalize shap_values into a 1D Python list of floats for the positive class when available.
    Handles:
      - list of arrays (per-class)
      - single numpy array with shape (1, n_features) or (n_features,)
      - nested lists like [[...]]
    """
    # If SHAP returned a list (per-class), pick class-1 if available else class-0
    if isinstance(shap_values, list):
        # prefer class index 1 when present
        if len(shap_values) > 1:
            sel = shap_values[1]
        else:
            sel = shap_values[0]
    else:
        sel = shap_values

    # Convert to numpy array
    arr = np.asarray(sel)

    # If array has shape (1, n_features) take first row
    if arr.ndim > 1:
        arr = arr[0]

    # Flatten and convert to python floats
    arr = np.ravel(arr)
    try:
        return [float(x) for x in arr.tolist()]
    except Exception:
        out = []
        for x in arr:
            try:
                out.append(float(x))
            except Exception:
                out.append(0.0)
        return out


# -------------------------
# Natural-language summary builder
# -------------------------
def build_nl_summary_from_shap(shap_result):
    """
    shap_result expected format:
    {
      "base_value": float,
      "explanations": [{"feature": "caller = +233...", "shap_value": 0.12}, ...]
    }
    Build a one-line summary highlighting top positive contributors (increase fraud score)
    and top negative contributors (decrease fraud score).
    """
    try:
        exps = shap_result.get("explanations", [])
        if not exps:
            return "No explanation available."

        # compute top contributors by absolute value
        sorted_exps = sorted(exps, key=lambda x: abs(x.get("shap_value", 0.0)), reverse=True)
        # take up to top 3 meaningful features
        top = sorted_exps[:3]

        parts = []
        for e in top:
            name = e.get("feature", "<feature>")
            val = e.get("shap_value", 0.0)
            sign = "+" if val >= 0 else "-"
            parts.append(f"{name} ({sign}{abs(val):.3f})")

        # Decide wording: if most top are positive -> "increased fraud risk"
        pos_count = sum(1 for e in top if e.get("shap_value", 0.0) > 0)
        neg_count = sum(1 for e in top if e.get("shap_value", 0.0) < 0)
        if pos_count >= neg_count:
            reason = "increased fraud risk"
        else:
            reason = "reduced fraud risk"

        return f"Top factors {reason}: " + ", ".join(parts) + "."
    except Exception:
        return "Explanation summary unavailable."


def build_nl_summary_from_lime(lime_result):
    """
    lime_result is list of [("feature phrase", weight), ...]
    We build a short sentence from top positive weights.
    """
    try:
        if not isinstance(lime_result, list) or not lime_result:
            return "No explanation available."
        # sort by absolute weight
        sorted_l = sorted(lime_result, key=lambda x: abs(x[1]), reverse=True)
        top = sorted_l[:3]
        parts = []
        pos_count = 0
        neg_count = 0
        for feat_desc, weight in top:
            sign = "+" if weight >= 0 else "-"
            parts.append(f"{feat_desc} ({sign}{abs(weight):.3f})")
            if weight >= 0:
                pos_count += 1
            else:
                neg_count += 1
        reason = "increased fraud risk" if pos_count >= neg_count else "reduced fraud risk"
        return f"Top factors {reason}: " + ", ".join(parts) + "."
    except Exception:
        return "Explanation summary unavailable."


# -------------------------
# Endpoints
# -------------------------
@app.post("/process_cdr")
async def process_cdr(cdr: CDR):
    try:
        cdr_dict = cdr.dict()
        cdr_dict["destination"] = cdr.callee
        result = is_fraudulent(cdr_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fraud logic error: {e}")

    call_id = cdr.call_id
    sim = cdr.sim
    imei = cdr.imei
    imsi = cdr.imsi
    duration = cdr.duration
    destination = cdr.callee
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    human_features = map_to_features(cdr_dict)
    numeric_row_list, human_labels = numeric_features_from_human(human_features)
    shap_result, lime_result = None, None

    try:
        row = pd.DataFrame([numeric_row_list], columns=EXPECTED_FEATURES)
    except Exception:
        row = None

    # SHAP explanation (robust)
    try:
        if row is not None:
            explainer = get_shap_explainer()
            shap_values = explainer.shap_values(row, check_additivity=False)
            vals_list = _shap_to_1d(shap_values)

            # expected_value handling
            try:
                base = explainer.expected_value[1] if hasattr(explainer.expected_value, "__len__") and len(explainer.expected_value) > 1 else explainer.expected_value
            except Exception:
                base = explainer.expected_value

            shap_result = {
                "base_value": float(base),
                "explanations": [
                    {"feature": human_labels[i], "shap_value": float(vals_list[i])}
                    for i in range(min(len(REAL_FEATURES), len(vals_list)))
                ]
            }
        else:
            shap_result = {"error": "SHAP failed: could not construct numeric input row."}
    except Exception as e:
        shap_result = {"error": f"SHAP failed: {str(e)}"}

    # LIME explanation
    try:
        if row is not None:
            lime_exp = get_lime_explainer().explain_instance(
                row.values[0], rf_model.predict_proba, num_features=len(EXPECTED_FEATURES)
            )
            raw_lime = lime_exp.as_list()
            human_map = {feat: human_labels[idx] for idx, feat in enumerate(EXPECTED_FEATURES)}
            pretty = []
            for feat_desc, weight in raw_lime:
                # skip dummy features
                if feat_desc.split(" ", 1)[0].startswith("dummy"):
                    continue
                pretty_feat = prettify_lime_item(feat_desc, human_map)
                pretty.append([pretty_feat, float(weight)])
            lime_result = pretty
        else:
            lime_result = {"error": "LIME failed: could not construct numeric input row."}
    except Exception as e:
        lime_result = {"error": f"LIME failed: {str(e)}"}

    # Build one-line natural language summary (prefer SHAP, fallback to LIME)
    summary = None
    try:
        if isinstance(shap_result, dict) and "explanations" in shap_result:
            summary = build_nl_summary_from_shap(shap_result)
        elif isinstance(lime_result, list):
            summary = build_nl_summary_from_lime(lime_result)
        else:
            summary = "No explanation summary available."
    except Exception:
        summary = "No explanation summary available."

    # Store human-readable feature dict and explanations + summary
    processed_cdrs.append({"features": human_features, "shap": shap_result, "lime": lime_result, "summary": summary})
    if len(processed_cdrs) > 50:
        processed_cdrs.pop(0)

    if result.get("fraud"):
        try:
            requests.post(f"{TELECOM_BACKEND_URL}/block_sim", json={"sim": sim}, timeout=2)
            requests.post(f"{TELECOM_BACKEND_URL}/block_imei", json={"imei": imei}, timeout=2)
            requests.post(f"{TELECOM_BACKEND_URL}/block_imsi", json={"imsi": imsi}, timeout=2)
            requests.post(f"{TELECOM_BACKEND_URL}/disconnect_call", json={"call_id": call_id}, timeout=2)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to call telco backend: {e}")

        log_entry = f"[{timestamp}] 🚨 FRAUD detected | Call {call_id} | Dest: {destination} | Duration: {duration}s"
        fraud_api_logs.append(log_entry)

        return {
            "status": "Fraudulent Call Detected 🚨",
            "details": {
                "call_id": call_id,
                "destination": destination,
                "duration_seconds": duration,
                "reason": "Suspicious calling pattern flagged by AI model",
                "detected_by": result.get("model"),
                "action_taken": [f"SIM {sim} blocked", f"IMEI {imei} blocked", f"IMSI {imsi} blocked", f"Call {call_id} disconnected"]
            },
            "explanations": {"shap": shap_result, "lime": lime_result, "summary": summary},
            "advice": "Investigate subscriber account and monitor for further fraudulent activity."
        }

    log_entry = f"[{timestamp}] ✅ Legitimate Call | Call {call_id} | Dest: {destination} | Duration: {duration}s"
    fraud_api_logs.append(log_entry)

    return {
        "status": "Legitimate Call ✅",
        "details": {
            "call_id": call_id,
            "destination": destination,
            "duration_seconds": duration,
            "reason": "No suspicious pattern detected",
            "detected_by": result.get("model"),
        },
        "explanations": {"shap": shap_result, "lime": lime_result, "summary": summary},
        "advice": "No action needed. Continue monitoring network activity."
    }


@app.get("/logs")
def get_logs():
    pretty_output = "\n".join(fraud_api_logs[-50:]) or "No logs yet."
    return Response(content=pretty_output, media_type="text/plain")


@app.get("/health")
def health_check():
    return {"status": "fraud_api_running ✅"}


@app.post("/explain_shap")
def explain_shap(cdr: CDR):
    try:
        human_features = map_to_features(cdr.dict())
        numeric_row_list, human_labels = numeric_features_from_human(human_features)
        row = pd.DataFrame([numeric_row_list], columns=EXPECTED_FEATURES)
        explainer = get_shap_explainer()
        shap_values = explainer.shap_values(row, check_additivity=False)
        vals_list = _shap_to_1d(shap_values)
        base_value = explainer.expected_value[1] if hasattr(explainer.expected_value, "__len__") and len(explainer.expected_value) > 1 else explainer.expected_value
        shap_out = {
            "base_value": float(base_value),
            "explanations": [
                {"feature": human_labels[i], "shap_value": float(vals_list[i])}
                for i in range(min(len(REAL_FEATURES), len(vals_list)))
            ]
        }
        summary = build_nl_summary_from_shap(shap_out)
        return {"features": human_features, "shap_values": shap_out, "summary": summary}
    except Exception as e:
        return {"error": f"SHAP explanation failed: {str(e)}"}


@app.post("/explain_lime")
def explain_lime(cdr: CDR):
    try:
        human_features = map_to_features(cdr.dict())
        numeric_row_list, human_labels = numeric_features_from_human(human_features)
        row = pd.DataFrame([numeric_row_list], columns=EXPECTED_FEATURES)
        exp = get_lime_explainer().explain_instance(row.values[0], rf_model.predict_proba, num_features=len(EXPECTED_FEATURES))
        human_map = {feat: human_labels[idx] for idx, feat in enumerate(EXPECTED_FEATURES)}
        pretty = []
        for feat_desc, weight in exp.as_list():
            if feat_desc.split(" ", 1)[0].startswith("dummy"):
                continue
            pretty_feat = prettify_lime_item(feat_desc, human_map)
            pretty.append([pretty_feat, float(weight)])
        summary = build_nl_summary_from_lime(pretty)
        return {"features": human_features, "lime_explanation": pretty, "summary": summary}
    except Exception as e:
        return {"error": f"LIME explanation failed: {str(e)}"}


@app.get("/latest_explanations")
def latest_explanations():
    """
    Return a structured object that Streamlit expects.
    The response includes:
    - latest_cdr: human-readable feature dict (keys are REAL_FEATURES)
    - shap: { values: [...], base_value: float }  (values order matches latest_cdr keys)
    - lime: list of [pretty_feature, weight]
    - summary: one-line natural language summary
    """
    try:
        if not processed_cdrs:
            return {"message": "No CDRs processed yet."}

        latest = processed_cdrs[-1]
        human_features = latest.get("features", {})
        shap_result = latest.get("shap", {})
        lime_result = latest.get("lime", [])
        summary = latest.get("summary", "")

        # prepare shap values list (numeric) in same order as human_features (REAL_FEATURES)
        shap_values_list = []
        base_value = None
        try:
            if isinstance(shap_result, dict) and "explanations" in shap_result:
                # shap_result["explanations"] is a list of {"feature": human_label, "shap_value": val}
                shap_values_list = [float(item.get("shap_value", 0.0)) for item in shap_result.get("explanations", [])]
                base_value = float(shap_result.get("base_value")) if "base_value" in shap_result else None
            else:
                # fallback: if shap_result already contains values keyed differently
                if isinstance(shap_result, dict) and "values" in shap_result:
                    shap_values_list = [float(v) for v in shap_result.get("values", [])]
                    base_value = float(shap_result.get("base_value")) if "base_value" in shap_result else None
        except Exception:
            shap_values_list = []

        # final response shape (keeps keys Streamlit code expects)
        return {
            "latest_cdr": human_features,
            "shap": {
                "values": shap_values_list,
                "base_value": base_value
            },
            "lime": lime_result,
            "summary": summary
        }
    except Exception as e:
        return {"error": f"Latest explanations failed: {str(e)}"}
