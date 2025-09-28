import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# -------------------
# API Endpoints
# -------------------
FRAUD_API_URL = "http://127.0.0.1:8001/process_cdr"
FRAUD_LOGS_URL = "http://127.0.0.1:8001/logs"
TELCO_LOGS_URL = "http://127.0.0.1:8002/logs"
TELCO_INGEST_URL = "http://127.0.0.1:8002/ingest_cdr"

# -------------------
# Streamlit Config
# -------------------
st.set_page_config(
    page_title="Simbox Fraud Detection System",
    page_icon="📡",
    layout="wide",
)

# --- Background & Styling ---
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.unsplash.com/photo-1644344086189-fafecfeefdc4?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    [data-testid="stHeader"] {background: rgba(0,0,0,0);}
    [data-testid="stSidebar"] {background: rgba(0, 0, 0, 0.6);}
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 25px;
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: transform 0.3s ease;
    }
    .glass-card:hover {transform: scale(1.02);}
    .main-title {
        font-size: 60px !important;
        font-weight: 900;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #ff6ec7, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .sub-title {
        font-size: 24px !important;
        text-align: center;
        color: #f0f0f0;
        margin-bottom: 50px;
    }
    textarea {
        background-color: rgba(0,0,0,0.6) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        font-family: 'Courier New', monospace !important;
        color: #ecf0f1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Page Title ---
st.markdown("<p class='main-title'>📡 Simbox Fraud Detection</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Real Time AI Fraud Analysis & Telco Backend Actions</p>", unsafe_allow_html=True)

# -------------------
# File Upload & Ingest
# -------------------
uploaded_file = st.file_uploader("📁 Upload CDR CSV for Bulk Ingest", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Sample CDR Data")
    st.dataframe(df.head(), use_container_width=True)

    predictions = []
    with st.spinner("⚡ Analyzing uploaded CDRs..."):
        for i, row in df.iterrows():
            # ✅ Map row to API schema to avoid 422 errors
            cdr = {
                "call_id": str(i),
                "caller": str(row.get("phone number", row.get("caller", "unknown"))),
                "callee": str(row.get("number called", row.get("callee", "unknown"))),
                "duration": int(row.get("duration", row.get("Day Mins", 0))),
                "roaming": 0,
                "call_type": "voice",
                "sim": str(row.get("area code", "SIM123")),
                "imei": "IMEI123456789",
                "imsi": "IMSI987654321",
                "timestamp": "2025-09-25T12:30:00",
                "isFraud": int(row.get("isFraud", 0)),
            }

            try:
                resp = requests.post(TELCO_INGEST_URL, json=cdr, timeout=10)
                if resp.status_code == 200:
                    result = resp.json().get("status", "Processed")
                else:
                    result = f"Error {resp.status_code}"
            except Exception:
                result = "Error"
            predictions.append(result)

    # --- Fallback if all API calls failed ---
    if all("Error" in str(p) for p in predictions) and "isFraud" in df.columns:
        st.warning("⚠️ API rejected CDRs. Falling back to `isFraud` column from CSV.")
        predictions = df["isFraud"].apply(lambda x: "Fraudulent Call Detected 🚨" if x else "Legitimate Call ✅")

    df["Result"] = predictions

    # Metric cards
    fraud_count = sum("Fraudulent" in str(x) for x in df["Result"])
    legit_count = sum("Legitimate" in str(x) for x in df["Result"])
    total = len(predictions)

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.markdown(f"<div class='glass-card'><h3>🚨 Fraudulent</h3><h1>{fraud_count}</h1></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='glass-card'><h3>✅ Legitimate</h3><h1>{legit_count}</h1></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='glass-card'><h3>📞 Total Calls</h3><h1>{total}</h1></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Charts & Tabs
    tab1, tab2 = st.tabs(["📊 Fraud vs Legit", "⏱️ Duration Histogram"])
    with tab1:
        chart_df = pd.DataFrame({"Type": ["Fraudulent", "Legitimate"], "Count": [fraud_count, legit_count]})
        fig = px.pie(chart_df, names="Type", values="Count",
                     color_discrete_map={"Fraudulent":"red", "Legitimate":"#2ecc71"},
                     title="Fraud vs Legit Call Distribution")
        st.plotly_chart(fig, width="stretch")
    with tab2:
        duration_col = "duration" if "duration" in df.columns else "Day Mins" if "Day Mins" in df.columns else None
        if duration_col:
            fig = px.histogram(df, x=duration_col,
                               title=f"{duration_col} Distribution",
                               color_discrete_sequence=["#3498db"],
                               nbins=20)
            st.plotly_chart(fig, width="stretch")

    # Detailed table
    st.subheader("📋 Detailed Results")
    st.dataframe(df, use_container_width=True, height=400)

# -------------------
# Logs Section
# -------------------
st.markdown("---")
st.subheader("📝 Live Logs from Backends")

st_autorefresh(interval=15 * 1000, key="logs_refresh")

log_tab, telco_tab = st.tabs(["Fraud API Logs", "Telco Backend Logs"])

with log_tab:
    try:
        logs_text = requests.get(FRAUD_LOGS_URL, timeout=5).text
        st.text_area("Fraud API Logs", logs_text, height=300, key="fraud_logs_area")
    except:
        st.warning("⚠️ Unable to fetch fraud API logs.")

with telco_tab:
    try:
        telco_text = requests.get(TELCO_LOGS_URL, timeout=5).text
        st.text_area("Telco Backend Logs", telco_text, height=300, key="telco_logs_area")
    except:
        st.warning("⚠️ Unable to fetch telco logs.")
