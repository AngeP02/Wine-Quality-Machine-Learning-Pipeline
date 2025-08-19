import os
import io
import json
import pandas as pd
import joblib
from google.cloud import storage
from flask import Request, jsonify

MODEL_BUCKET = os.getenv("MODEL_BUCKET", "wine-dataset-modello")

def _scarica_modello(model_name: str):
    """Scarica il modello dal bucket e lo restituisce come dizionario."""
    storage_client = storage.Client()
    blob = storage_client.bucket(MODEL_BUCKET).blob(model_name)
    local_path = f"/tmp/{model_name}"  # /tmp in Cloud Functions è scrivibile
    blob.download_to_filename(local_path)
    modello = joblib.load(local_path)
    return modello

def predict(request: Request):
    """
    request: oggetto Flask Request con JSON tipo:
    {
        "instances": [
            [8.9,0.62,0.19,3.9,0.17,51,148,0.9986,3.17,0.93,9.2]
        ]
    }
    """
    try:
        request_json = request.get_json()
        if not request_json or "instances" not in request_json:
            return jsonify({"error": "JSON invalido, manca 'instances'"}), 400

        # Prendi l'ultimo modello salvato (puoi decidere di parametrizzarlo)
        model_name = "wine-dataset_unito_balanced_rf_classifier.joblib"
        saved = _scarica_modello(model_name)

        pipe = saved["model"]
        feature_cols = saved["feature_cols"]

        # Creiamo un DataFrame coerente con le colonne di training
        df = pd.DataFrame(request_json["instances"], columns=feature_cols)

        # Predizione
        preds = pipe.predict(df).tolist()
        return jsonify({"predictions": preds}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
