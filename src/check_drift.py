import subprocess

import pandas as pd
from scipy.stats import ks_2samp


DRIFT_THRESHOLD = 0.05
MIN_SAMPLES_FOR_RETRAINING = 10


def load_reference_data():
    return pd.read_csv("data/processed/cleaned.csv")


def load_recent_predictions():
    return pd.read_csv("logs/predictions.csv")


def detect_drift(reference_sales, recent_sales):

    stat, p_value = ks_2samp(
        reference_sales,
        recent_sales
    )

    drift_detected = p_value < DRIFT_THRESHOLD

    return drift_detected, p_value


def save_status(message):
    with open(
        "logs/retraining_status.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(message)


def retrain_model():
    print("Retraining model...")
    subprocess.run(["python", "src/train.py"])
    print("Model updated successfully")


if __name__ == "__main__":

    historical_sales = load_reference_data()
    recent_predictions = load_recent_predictions().tail(10)
    

    if len(recent_predictions) < MIN_SAMPLES_FOR_RETRAINING:
        print("Not enough prediction data for drift analysis")
        save_status("Waiting for more production predictions")
        exit()

    drift_detected, p_value = detect_drift(
        historical_sales["sales"],
        recent_predictions["sales"]
    )

    print(f"P-value: {p_value:.5f}")

    if drift_detected:
        print("Distribution shift detected")

        retrain_model()

        save_status(
            f"Model retrained automatically (p-value={p_value:.5f})"
        )

    else:
        print("System stable")

        save_status(
            f"No drift detected (p-value={p_value:.5f})"
        )