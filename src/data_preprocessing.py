import pandas as pd

def load_data(path):
    df = pd.read_csv(path, parse_dates=['date'])
    return df

def filter_data(df):
    df = df[(df['store_nbr'] == 1) & (df['family'] == 'GROCERY I')]
    return df

def preprocess(df):
    df = df.sort_values('date')
    df = df[['date', 'sales']]
    df = df.dropna()
    return df

if __name__ == "__main__":
    df = load_data("data/raw/train.csv")
    df = filter_data(df)
    df = preprocess(df)

    df.to_csv("data/processed/cleaned.csv", index=False)
    print("✅ Data preprocessing completed")