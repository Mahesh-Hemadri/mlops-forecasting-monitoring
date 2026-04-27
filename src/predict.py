import pandas as pd
import joblib
from src.feature_engineering import create_features

model = joblib.load("models/model.pkl")

def predict(input_df):
    df = create_features(input_df)
    X = df.drop(columns=['sales', 'date'])
    preds = model.predict(X)
    return preds

if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned.csv", parse_dates=['date'])
    predictions = predict(df)

    print(predictions[:5])