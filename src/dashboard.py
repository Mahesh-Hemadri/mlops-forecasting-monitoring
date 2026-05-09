import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="MLOps Dashboard",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 MLOps Monitoring Dashboard")
st.success("✅ Real-time ML Monitoring System Active")

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("logs/predictions.csv")

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="mixed",
    errors="coerce"
)

df = df.dropna(subset=["timestamp"])
df = df.sort_values("timestamp")

# -------------------------
# METRICS
# -------------------------
latest_pred = df["prediction"].iloc[-1]

rmse = np.sqrt(
    ((df["sales"] - df["prediction"]) ** 2).mean()
)

# -------------------------
# DRIFT DETECTION
# -------------------------
train_df = pd.read_csv("data/processed/cleaned.csv")

stat, p_value = ks_2samp(
    train_df["sales"],
    df["sales"]
)

drift_detected = p_value < 0.05

# =========================
# RETRAINING STATUS
# =========================

st.subheader("🔄 Retraining Status")

try:

    with open(
        "logs/retraining_status.txt",
        "r"
    ) as f:

        status = f.read()

    if "retrained" in status.lower():
        st.success(status)
    else:
        st.info(status)

except:
    st.warning(
        "No retraining status available"
    )

# =========================
# KPI SECTION
# =========================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Predictions",
    len(df)
)

col2.metric(
    "Latest Prediction",
    f"{latest_pred:.2f}"
)

col3.metric(
    "RMSE",
    f"{rmse:.2f}"
)

st.divider()

# =========================
# DRIFT + MODEL INFO
# =========================

left, right = st.columns(2)

with left:
    st.subheader("🧠 Drift Detection")

    if drift_detected:
        st.error(
            f"⚠️ Drift Detected (p-value={p_value:.5f})"
        )
    else:
        st.success(
            f"✅ No Drift (p-value={p_value:.5f})"
        )

with right:
    st.subheader("📦 Model Info")

    st.info("""
    Model: XGBoost Regressor
    
    Deployment: AWS EC2
    
    API: FastAPI
    
    Monitoring: Streamlit
    """)

st.divider()

# =========================
# CHARTS SIDE BY SIDE
# =========================

chart1, chart2 = st.columns(2)

# -------------------------
# Prediction Trend
# -------------------------

with chart1:

    st.subheader("📈 Predictions Trend")

    fig, ax = plt.subplots(figsize=(6,4))

    ax.plot(
        df["timestamp"],
        df["prediction"],
        marker='o'
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Prediction")
    ax.grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

# -------------------------
# Actual vs Predicted
# -------------------------

with chart2:

    st.subheader("📉 Actual vs Predicted")

    fig, ax = plt.subplots(figsize=(6,4))

    ax.plot(
        df["timestamp"],
        df["sales"],
        label="Actual",
        marker='o'
    )

    ax.plot(
        df["timestamp"],
        df["prediction"],
        label="Predicted",
        marker='x'
    )

    ax.legend()

    ax.set_xlabel("Time")
    ax.set_ylabel("Sales")
    ax.grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

st.divider()

# =========================
# SUMMARY
# =========================

st.subheader("📊 System Summary")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.write(f"• Total Predictions: {len(df)}")
    st.write(f"• Latest Prediction: {latest_pred:.2f}")

with summary_col2:
    st.write(f"• RMSE: {rmse:.2f}")
    st.write(
        f"• Drift Status: {'Detected' if drift_detected else 'Stable'}"
    )