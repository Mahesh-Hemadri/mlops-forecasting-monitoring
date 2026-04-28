from fastapi import FastAPI
import joblib
import pandas as pd
from src.feature_engineering import create_features

app = FastAPI()

# load model once
model = joblib.load("models/model.pkl")


@app.get("/")
def home():
    return {"message": "ML Model API is running"}


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    df = create_features(df)

    X = df.drop(columns=['sales', 'date'], errors='ignore')

    preds = model.predict(X)

    return {"prediction": float(preds[0])}