import pandas as pd
from scipy.stats import ks_2samp


def load_data():
    train_df = pd.read_csv("data/processed/cleaned.csv")
    prod_df = pd.read_csv("logs/predictions.csv")
    return train_df, prod_df


def check_drift(train_df, prod_df):
    # compare sales distribution
    stat, p_value = ks_2samp(train_df["sales"], prod_df["sales"])

    print(f"KS Statistic: {stat}")
    print(f"P-value: {p_value}")

    if p_value < 0.05:
        print("⚠️ Drift detected!")
    else:
        print("✅ No drift detected")


if __name__ == "__main__":
    train_df, prod_df = load_data()
    check_drift(train_df, prod_df)