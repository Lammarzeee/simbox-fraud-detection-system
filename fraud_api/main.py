from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from fraud_logic import is_fraudulent
import requests
from datetime import datetime

app = FastAPI(title="Fraud Detection API")

TELECOM_BACKEND_URL = "http://127.0.0.1:8002"

# ✅ In-memory logs for fraud API
fraud_api_logs = []


# ✅ Updated CDR model to match dummy telco data
class CDR(BaseModel):
    call_id: str
    caller: str
    callee: str
    duration: int
    roaming: int
    call_type: str
    sim: str
    imei: str
    imsi: str
    timestamp: str
    isFraud: int = 0  # optional default


@app.post("/process_cdr")
async def process_cdr(cdr: CDR):
    try:
        # ✅ Map callee -> destination before fraud logic
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
    destination = cdr.callee  # ✅ now using callee directly
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Fraudulent case ---
    if result.get("fraud"):
        try:
            requests.post(f"{TELECOM_BACKEND_URL}/block_sim", json={"sim": sim}, timeout=2)
            requests.post(f"{TELECOM_BACKEND_URL}/block_imei", json={"imei": imei}, timeout=2)
            requests.post(f"{TELECOM_BACKEND_URL}/block_imsi", json={"imsi": imsi}, timeout=2)
            requests.post(f"{TELECOM_BACKEND_URL}/disconnect_call", json={"call_id": call_id}, timeout=2)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to call telco backend: {e}")

        log_entry = f"[{timestamp}] 🚨 FRAUD detected | Call {call_id} | Dest: {destination} | Duration: {duration}s | Actions: SIM {sim}, IMEI {imei}, IMSI {imsi} blocked + Call disconnected"
        fraud_api_logs.append(log_entry)

        return {
            "status": "Fraudulent Call Detected 🚨",
            "details": {
                "call_id": call_id,
                "destination": destination,
                "duration_seconds": duration,
                "reason": "Suspicious calling pattern flagged by AI model",
                "detected_by": result.get("model"),
                "action_taken": [
                    f"SIM {sim} blocked",
                    f"IMEI {imei} blocked",
                    f"IMSI {imsi} blocked",
                    f"Call {call_id} disconnected"
                ]
            },
            "advice": "Investigate subscriber account and monitor for further fraudulent activity."
        }

    # --- Legitimate case ---
    log_entry = f"[{timestamp}] ✅ Legitimate Call | Call {call_id} | Dest: {destination} | Duration: {duration}s | Detected by: {result.get('model')}"
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
        "advice": "No action needed. Continue monitoring network activity."
    }


# --- Pretty Logs endpoint ---
@app.get("/logs")
def get_logs():
    """Return last 50 fraud detection logs in plain text."""
    pretty_output = "\n".join(fraud_api_logs[-50:]) or "No logs yet."
    return Response(content=pretty_output, media_type="text/plain")


@app.get("/health")
def health_check():
    return {"status": "fraud_api_running ✅"}
