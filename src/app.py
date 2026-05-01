from fastapi import FastAPI
import joblib
import pandas as pd
from src.feature_engineering import create_features
from src.logger import log_prediction

app = FastAPI()

# load model once
model = joblib.load("models/model.pkl")


@app.get("/")
def home():
    return {"message": "ML Model API is running"}


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    # add basic features
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    # TEMP FIX: dummy lag values
    df['lag_1'] = data.get("sales", 0)
    df['lag_7'] = data.get("sales", 0)

    X = df[['day_of_week', 'month', 'lag_1', 'lag_7']]

    preds = model.predict(X)
    pred = float(preds[0])

    # log it
    log_prediction(data, pred)

    return {"prediction": pred}
    