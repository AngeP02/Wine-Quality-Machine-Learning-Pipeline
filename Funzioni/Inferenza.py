import pandas as pd
import os
import pickle
import numpy as np

def get_header_from_processed(processed_dir):
    try:
        df = pd.read_csv(os.path.join(processed_dir, "wine_processed.csv"))
        return [col for col in df.columns if col not in ['quality', 'wine_type']]
    except:
        return []

def predici(user_input_dict, models_dir):
    input_df = pd.DataFrame([user_input_dict])
    with open(os.path.join(models_dir, "wine_model.pkl"), "rb") as f:
        model = pickle.load(f)
    prediction = model.predict(input_df)
    return int(prediction[0])

# --- Inferenza.py ---
import pickle
import pandas as pd
import os

MODEL_DIR = "./simulated_cloud_storage/models"

def predici_qualita(nuovo_vino: dict):
    modello_path = os.path.join(MODEL_DIR, "model.pkl")

    with open(modello_path, "rb") as f:
        model = pickle.load(f)

    df_nuovo = pd.DataFrame([nuovo_vino])

    predizione = model.predict(df_nuovo)
    return predizione[0]
