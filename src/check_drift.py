import pandas as pd
from scipy.stats import ks_2samp
import subprocess

# -------------------------
# Load Training Data
# -------------------------
train_df = pd.read_csv(
    "data/processed/cleaned.csv"
)

# -------------------------
# Load Prediction Logs
# -------------------------
pred_df = pd.read_csv(
    "logs/predictions.csv"
)

# -------------------------
# Drift Detection
# -------------------------
stat, p_value = ks_2samp(
    train_df["sales"],
    pred_df["sales"]
)

print(f"KS Statistic: {stat}")
print(f"P-value: {p_value}")

# -------------------------
# Retraining Trigger
# -------------------------
if p_value < 0.05:

    print("⚠️ Drift detected!")
    print("🚀 Triggering retraining...")

    subprocess.run(
        ["python", "src/train.py"]
    )

    # Save retraining status
    with open(
        "logs/retraining_status.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "✅ Model retrained successfully"
        )

    print("✅ Model retrained successfully")

else:

    # Save stable status
    with open(
        "logs/retraining_status.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "✅ System stable - no retraining needed"
        )

    print("✅ No drift detected")