import joblib
import pandas as pd
from fastapi import FastAPI

from src.prediction_logger import log_prediction


app = FastAPI()

model = joblib.load("models/model.pkl")


def prepare_input(payload):

    frame = pd.DataFrame([payload])

    frame["date"] = pd.to_datetime(frame["date"])

    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["month"] = frame["date"].dt.month

    sales_value = payload.get("sales", 0)

    frame["lag_1"] = sales_value
    frame["lag_7"] = sales_value

    return frame[
        ["day_of_week", "month", "lag_1", "lag_7"]
    ]


@app.get("/")
def health_check():
    return {"status": "API running"}


@app.post("/predict")
def predict(payload: dict):

    model_input = prepare_input(payload)

    prediction = float(
        model.predict(model_input)[0]
    )

    log_prediction(payload, prediction)

    return {
        "prediction": round(prediction, 2)
    }