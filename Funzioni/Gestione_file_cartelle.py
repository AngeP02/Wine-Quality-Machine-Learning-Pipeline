# logic/file_manager.py

import os
import pandas as pd
from config import RAW_DATA_DIR

def save_uploaded_file(uploaded_file):
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    file_path = os.path.join(RAW_DATA_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def load_csv_files():
    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]
    datasets = []
    for file in files:
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, file), sep=';')
        datasets.append((file, df))
    return datasets
