import joblib
import pandas as pd
from xgboost import XGBRegressor

from src.feature_engineering import create_features


MODEL_PATH = "models/model.pkl"


def load_training_data():
    return pd.read_csv(
        "data/processed/cleaned.csv",
        parse_dates=["date"]
    )


def build_model():
    return XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )


def train():

    sales_data = load_training_data()

    feature_data = create_features(sales_data)

    X = feature_data.drop(columns=["sales", "date"])
    y = feature_data["sales"]

    model = build_model()

    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)

    print("Model training complete")


if __name__ == "__main__":
    train()