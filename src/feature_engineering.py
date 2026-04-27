import pandas as pd

def create_features(df):
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    df['lag_1'] = df['sales'].shift(1)
    df['lag_7'] = df['sales'].shift(7)

    df = df.dropna()
    return df