import pandas as pd

def preprocessa_dati(df):
    df = df.dropna()
    if 'type' not in df.columns:
        df['type'] = 'unknown'
    df = pd.get_dummies(df, columns=['type'])
    return df