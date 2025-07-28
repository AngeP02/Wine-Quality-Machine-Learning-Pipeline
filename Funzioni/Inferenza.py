# logic/inference.py

import pickle
import pandas as pd
from config import MODEL_DIR, MODEL_FILENAME

def load_model():
    with open(f"{MODEL_DIR}/{MODEL_FILENAME}", "rb") as f:
        model = pickle.load(f)
    return model

def make_prediction(model, input_data: dict):
    df = pd.DataFrame([input_data])
    return model.predict(df)[0]
