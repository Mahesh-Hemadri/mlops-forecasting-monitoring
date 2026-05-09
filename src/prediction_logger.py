from datetime import datetime
from pathlib import Path

import pandas as pd


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "predictions.csv"


def log_prediction(payload, prediction):

    LOG_DIR.mkdir(exist_ok=True)

    log_row = {
        "timestamp": datetime.now(),
        "date": payload.get("date"),
        "sales": payload.get("sales"),
        "prediction": prediction
    }

    log_df = pd.DataFrame([log_row])

    write_mode = "a" if LOG_FILE.exists() else "w"

    log_df.to_csv(
        LOG_FILE,
        mode=write_mode,
        header=not LOG_FILE.exists(),
        index=False
    )