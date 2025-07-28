# logic/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import PROCESSED_DATA_DIR
import os

def preprocess(df, file_label=""):
    df = df.dropna()
    if 'type' not in df.columns:
        df["type"] = file_label
    if 'type' in df.columns:
        encoder = LabelEncoder()
        df['type'] = encoder.fit_transform(df['type'])  # red → 0, white → 1
    features = df.drop(columns=["quality"])
    target = df["quality"]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    df_scaled["quality"] = target.values
    df_scaled["type"] = df["type"]
    return df_scaled

def save_processed(df, filename):
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DATA_DIR, filename)
    df.to_csv(out_path, index=False)
    return out_path
