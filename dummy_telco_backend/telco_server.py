from fastapi import FastAPI, Request, Response
import httpx
import asyncio
from datetime import datetime
import json
import os

# ✅ Explicitly enable Swagger UI and ReDoc
app = FastAPI(
    title="Dummy Telco Server",
    description="Simulated Telco server for testing fraud detection system",
    version="1.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc"      # ReDoc alternative UI
)

fraud_api_url = "http://127.0.0.1:8001/process_cdr"

# Path to your live CDR file (CSV or JSONL)
LIVE_CDR_FILE = "live_cdrs.jsonl"

# In-memory logs (pretty + detailed)
action_log = []


async def send_with_retry(client, cdr, retries=1, delay=2):
    try:
        return await client.post(fraud_api_url, json=cdr, timeout=10.0)
    except Exception as e:
        if retries > 0:
            await asyncio.sleep(delay)
            return await send_with_retry(client, cdr, retries - 1, delay)
        return e


def read_live_cdrs():
    """Read the latest CDRs from the file."""
    if not os.path.exists(LIVE_CDR_FILE):
        return []

    cdrs = []
    _, ext = os.path.splitext(LIVE_CDR_FILE.lower())
    try:
        if ext in [".jsonl", ".ndjson"]:
            with open(LIVE_CDR_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        cdrs.append(json.loads(line))
        else:  # assume CSV
            import csv
            with open(LIVE_CDR_FILE, newline='', encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cdrs.append(row)
    except Exception as e:
        print(f"Error reading live CDRs: {e}")
    return cdrs


# ✅ NEW: Direct Ingest Endpoint for Streamlit/Dummy server
@app.post("/ingest_cdr")
async def ingest_cdr(request: Request):
    """Receive raw CDR from client, forward to Fraud API, log result."""
    cdr = await request.json()
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(fraud_api_url, json=cdr, timeout=10.0)
            fraud_response = resp.json()
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            action_log.append(
                f"[{ts}] 📥 Ingested CDR {cdr.get('call_id', 'unknown')} → Fraud API: {fraud_response.get('status', 'N/A')}"
            )
            return fraud_response
        except Exception as e:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            action_log.append(
                f"[{ts}] ❌ Failed to process CDR {cdr.get('call_id', 'unknown')} | Error: {e}"
            )
            return {"status": "❌ Error forwarding to Fraud API", "detail": str(e)}


@app.get("/send-cdrs")
async def send_cdrs():
    results = []
    cdr_samples = read_live_cdrs()  # dynamically load live CDRs

    if not cdr_samples:
        return {"status": "No CDRs found to send"}

    # Ensure 'destination' exists in each CDR
    for cdr in cdr_samples:
        if "destination" not in cdr:
            cdr["destination"] = cdr.get("callee") or cdr.get("caller") or "UNKNOWN"

    async with httpx.AsyncClient() as client:
        tasks = [send_with_retry(client, cdr, retries=1, delay=2) for cdr in cdr_samples]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    for cdr, resp in zip(cdr_samples, responses):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if isinstance(resp, Exception):
            log_entry = f"[{timestamp}] ❌ Failed to send CDR {cdr.get('call_id', 'unknown')} | Error: {resp}"
            action_log.append(log_entry)
            results.append({
                "cdr": cdr,
                "timestamp": timestamp,
                "fraud_api_response": {
                    "status": "Connection error ❌",
                    "message": "Could not reach Fraud API, even after retry.",
                    "technical_detail": str(resp)
                }
            })
        else:
            try:
                fraud_response = resp.json()
                log_entry = f"[{timestamp}] 📤 Sent CDR {cdr.get('call_id', 'unknown')} OK | Fraud API response: {fraud_response.get('status', 'N/A')}"
                action_log.append(log_entry)
                results.append({
                    "cdr": cdr,
                    "timestamp": timestamp,
                    "fraud_api_response": fraud_response
                })
            except Exception as e:
                log_entry = f"[{timestamp}] ⚠️ Invalid response for CDR {cdr.get('call_id', 'unknown')} | Error: {e}"
                action_log.append(log_entry)
                results.append({
                    "cdr": cdr,
                    "timestamp": timestamp,
                    "fraud_api_response": {
                        "status": "Invalid response from Fraud API ⚠️",
                        "message": "Fraud API returned unexpected data.",
                        "technical_detail": str(e),
                        "raw_response": resp.text
                    }
                })

        await asyncio.sleep(1)

    return {"sent_cdrs": results}


# --- Safe blocking endpoints with logging ---
@app.post("/block_sim")
def block_sim(sim: dict):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not sim or "sim" not in sim:
        return {"status": "❌ SIM key missing in request"}
    status = f"[{timestamp}] 🔒 SIM {sim['sim']} blocked"
    action_log.append(status)
    return {"status": status}


@app.post("/block_imei")
def block_imei(imei: dict):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not imei or "imei" not in imei:
        return {"status": "❌ IMEI key missing in request"}
    status = f"[{timestamp}] 🔒 IMEI {imei['imei']} blocked"
    action_log.append(status)
    return {"status": status}


@app.post("/block_imsi")
def block_imsi(imsi: dict):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not imsi or "imsi" not in imsi:
        return {"status": "❌ IMSI key missing in request"}
    status = f"[{timestamp}] 🔒 IMSI {imsi['imsi']} blocked"
    action_log.append(status)
    return {"status": status}


@app.post("/disconnect_call")
def disconnect_call(call: dict):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not call or "call_id" not in call:
        return {"status": "❌ call_id key missing in request"}
    status = f"[{timestamp}] ☎️ Call {call['call_id']} disconnected"
    action_log.append(status)
    return {"status": status}


# --- Pretty Logs endpoint ---
@app.get("/logs")
def logs():
    """Return last 50 telco server logs in plain text."""
    pretty_output = "\n".join(action_log[-50:]) or "No logs yet."
    return Response(content=pretty_output, media_type="text/plain")


# --- Status endpoint ---
@app.get("/status")
def status():
    return {"recent_actions": action_log[-10:]}  # last 10 actions


@app.get("/health")
def health():
    return {"status": "telco_server_running ✅"}
