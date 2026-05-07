import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

st.set_page_config(page_title="MLOps Dashboard", layout="wide")

st.title("🚀 MLOps Monitoring Dashboard")

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv("logs/predictions.csv")

# Fix timestamps
df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
df = df.dropna(subset=["timestamp"])
df = df.sort_values("timestamp")

# -------------------------
# Basic Metrics
# -------------------------
st.subheader("📊 System Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Predictions", len(df))

latest_pred = df["prediction"].iloc[-1]
col2.metric("Latest Prediction", round(latest_pred, 2))

rmse = np.sqrt(((df["sales"] - df["prediction"]) ** 2).mean())
col3.metric("RMSE", round(rmse, 2))

# -------------------------
# Drift Detection
# -------------------------
st.subheader("🧠 Drift Detection")

try:
    train_df = pd.read_csv("data/processed/cleaned.csv")

    stat, p_value = ks_2samp(train_df["sales"], df["sales"])

    if p_value < 0.05:
        st.error(f"⚠️ Drift Detected (p-value={p_value:.5f})")
        drift_detected = True
    else:
        st.success(f"✅ No Drift (p-value={p_value:.5f})")
        drift_detected = False

except:
    st.warning("Training data not found — skipping drift detection")
    drift_detected = False

# -------------------------
# Model Info
# -------------------------
st.subheader("🧠 Model Info")

st.write("""
- Model: XGBoost Regressor  
- Deployment: AWS EC2  
- API: FastAPI  
- Monitoring: Streamlit  
""")

# -------------------------
# Predictions Over Time
# -------------------------
st.subheader("📈 Predictions Over Time")

fig, ax = plt.subplots()
ax.plot(df["timestamp"], df["prediction"], marker='o')
ax.set_title("Predictions Trend")
ax.set_xlabel("Time")
ax.set_ylabel("Prediction")
ax.grid(True)
st.pyplot(fig)

# -------------------------
# Actual vs Predicted
# -------------------------
st.subheader("📉 Actual vs Predicted")

fig, ax = plt.subplots()
ax.plot(df["timestamp"], df["sales"], label="Actual", marker='o')
ax.plot(df["timestamp"], df["prediction"], label="Predicted", marker='x')
ax.legend()
ax.set_title("Model Performance")
ax.set_xlabel("Time")
ax.set_ylabel("Sales")
ax.grid(True)
st.pyplot(fig)

# -------------------------
# Summary
# -------------------------
st.subheader("📊 System Summary")

st.write(f"""
- Total Predictions: {len(df)}  
- Latest Prediction: {latest_pred:.2f}  
- RMSE: {rmse:.2f}  
- Drift Status: {"Detected" if drift_detected else "Stable"}  
""")