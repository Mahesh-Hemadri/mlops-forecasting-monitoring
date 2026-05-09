import pandas as pd


def create_features(data):

    dataset = data.copy()

    dataset["day_of_week"] = dataset["date"].dt.dayofweek
    dataset["month"] = dataset["date"].dt.month

    dataset["lag_1"] = dataset["sales"].shift(1)
    dataset["lag_7"] = dataset["sales"].shift(7)

    return dataset.dropna()