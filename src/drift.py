import pandas as pd
from scipy.stats import ks_2samp
import subprocess


def load_data():
    train_df = pd.read_csv("data/processed/cleaned.csv")
    prod_df = pd.read_csv("logs/predictions.csv")
    return train_df, prod_df






def check_drift(train_df, prod_df):
    stat, p_value = ks_2samp(train_df["sales"], prod_df["sales"])

    print(f"KS Statistic: {stat}")
    print(f"P-value: {p_value}")

    if p_value < 0.05:
        print("⚠️ Drift detected!")

        # simple safeguard
        if len(prod_df) > 50:
            print("Triggering retraining...")
            subprocess.run(["python", "src/train.py"])
            print("✅ Model retrained and updated")
        else:
            print("Not enough production data to retrain")


if __name__ == "__main__":
    train_df, prod_df = load_data()
    check_drift(train_df, prod_df)