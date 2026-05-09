import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import ks_2samp


st.set_page_config(
    page_title="MLOps Dashboard",
    layout="wide"
)


PREDICTION_LOGS = "logs/predictions.csv"
TRAINING_DATA = "data/processed/cleaned.csv"


def load_prediction_logs():

    logs = pd.read_csv(PREDICTION_LOGS)

    logs["timestamp"] = pd.to_datetime(
        logs["timestamp"],
        format="mixed",
        errors="coerce"
    )

    logs = logs.dropna(subset=["timestamp"])

    return logs.sort_values("timestamp")


def load_training_data():
    return pd.read_csv(TRAINING_DATA)


def calculate_rmse(actual, predicted):
    return np.sqrt(((actual - predicted) ** 2).mean())


def detect_drift(reference_sales, recent_sales):

    _, p_value = ks_2samp(
        reference_sales,
        recent_sales
    )

    return p_value < 0.05, p_value


def load_retraining_status():

    try:
        with open(
            "logs/retraining_status.txt",
            "r",
            encoding="utf-8"
        ) as file:

            return file.read()

    except FileNotFoundError:
        return "No retraining activity found"


prediction_logs = load_prediction_logs()
historical_sales = load_training_data()

latest_prediction = prediction_logs["prediction"].iloc[-1]

rmse = calculate_rmse(
    prediction_logs["sales"],
    prediction_logs["prediction"]
)

drift_detected, p_value = detect_drift(
    historical_sales["sales"],
    prediction_logs["sales"]
)

retraining_status = load_retraining_status()


st.title("MLOps Monitoring Dashboard")

st.caption(
    "FastAPI • XGBoost • Streamlit • AWS EC2"
)

st.success(
    "Monitoring pipeline active"
)


metric_1, metric_2, metric_3 = st.columns(3)

metric_1.metric(
    "Total Predictions",
    len(prediction_logs)
)

metric_2.metric(
    "Latest Prediction",
    round(latest_prediction, 2)
)

metric_3.metric(
    "RMSE",
    round(rmse, 2)
)


st.divider()


left_panel, right_panel = st.columns(2)


with left_panel:

    st.subheader("Drift Detection")

    if drift_detected:
        st.error(
            f"Distribution shift detected (p-value={p_value:.5f})"
        )
    else:
        st.success(
            f"System stable (p-value={p_value:.5f})"
        )


with right_panel:

    st.subheader("Retraining Status")

    if "retrained" in retraining_status.lower():
        st.success(retraining_status)
    else:
        st.info(retraining_status)


st.divider()


chart_left, chart_right = st.columns(2)


with chart_left:

    st.subheader("Prediction Trend")

    fig, ax = plt.subplots(figsize=(5, 3))

    ax.plot(
        prediction_logs["timestamp"],
        prediction_logs["prediction"],
        marker="o",
        linewidth=2
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Prediction")
    ax.grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)


with chart_right:

    st.subheader("Actual vs Predicted")

    fig, ax = plt.subplots(figsize=(5, 3))

    ax.plot(
        prediction_logs["timestamp"],
        prediction_logs["sales"],
        label="Actual",
        marker="o"
    )

    ax.plot(
        prediction_logs["timestamp"],
        prediction_logs["prediction"],
        label="Predicted",
        marker="x"
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Sales")
    ax.legend()

    ax.grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)


st.divider()


summary_left, summary_right = st.columns(2)

with summary_left:
    st.write(f"Predictions processed: {len(prediction_logs)}")
    st.write(f"Latest forecast: {latest_prediction:.2f}")

with summary_right:
    st.write(f"Current RMSE: {rmse:.2f}")
    st.write(
        f"Drift status: {'Detected' if drift_detected else 'Stable'}"
    )