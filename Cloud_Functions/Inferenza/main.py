import os
import io
import json
import joblib
import pandas as pd
from google.cloud import storage
from flask import Request

BUCKET_MODELLI = os.getenv("MODEL_BUCKET", "wine-dataset-modello")


def carica_modello(job_id: str):
    client = storage.Client()
    bucket = client.bucket(BUCKET_MODELLI)
    modelli = [b for b in bucket.list_blobs() if job_id in b.name and b.name.endswith(".joblib")]
    if not modelli:
        raise FileNotFoundError(f"Nessun modello trovato per job_id {job_id}")
    modello_blob = modelli[0]
    dati_binari = modello_blob.download_as_bytes()
    modello_salvato = joblib.load(io.BytesIO(dati_binari))
    return modello_salvato


def inferenza(request: Request):
    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return json.dumps({"error": "Corpo della richiesta deve essere JSON"}), 400
        job_id = request_json.get("job_id")
        features_input = request_json.get("features")

        if not job_id or not features_input:
            return json.dumps({"error": "Devi specificare 'job_id' e 'features'"}), 400
        modello_salvato = carica_modello(job_id)
        modello = modello_salvato["modello"]
        features_col = modello_salvato["feature_cols"]
        df_input = pd.DataFrame([features_input], columns=features_col)
        predizione = modello.predict(df_input)[0]
        return json.dumps({"qualita_predetta": int(predizione)}), 200
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)}), 404
    except Exception as e:
        return json.dumps({"error": f"Errore durante l'inferenza: {e}"}), 500