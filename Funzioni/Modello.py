# logic/training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from config import MODEL_DIR, MODEL_FILENAME

def train_model(df):
    X = df.drop(columns=["quality", "type"])
    y = df["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, MODEL_FILENAME), "wb") as f:
        pickle.dump(model, f)
