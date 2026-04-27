import pandas as pd
from xgboost import XGBRegressor
import joblib

from src.feature_engineering import create_features

def train():
    df = pd.read_csv("data/processed/cleaned.csv", parse_dates=['date'])

    df = create_features(df)

    X = df.drop(columns=['sales', 'date'])
    y = df['sales']

    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)

    joblib.dump(model, "models/model.pkl")
    print("✅ Model trained and saved")

if __name__ == "__main__":
    train()