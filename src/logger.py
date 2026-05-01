import pandas as pd
import os
from datetime import datetime

LOG_FILE = "logs/predictions.csv"


def log_prediction(input_data, prediction):
    log_entry = {
        "timestamp": datetime.now(),
        "date": input_data.get("date"),
        "sales": input_data.get("sales"),
        "prediction": prediction
    }

    df = pd.DataFrame([log_entry])

    if not os.path.exists("logs"):
        os.makedirs("logs")

    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)