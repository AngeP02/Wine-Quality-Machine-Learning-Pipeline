#Preprocessing

import os, io, json, logging, datetime
from google.cloud import storage
import pandas as pd

OUTPUT_BUCKET = os.getenv("OUTPUT_BUCKET", "wine-dataset-preprocessati")
STATUS_BUCKET = os.getenv("STATUS_BUCKET", "wine-dataset-status")

def _scrivi_stato(storage_client, job_id, stato, extra=None, reset=False):
    bucket = storage_client.bucket(STATUS_BUCKET)
    status_file = f"status_preprocessing_{job_id}.json"
    blob = bucket.blob(status_file)
    if reset or not blob.exists():
        contenuto = {
            "fase": "preprocessing",
            "cronologia": []
        }
    else:
        contenuto = json.loads(blob.download_as_text())
    contenuto.setdefault("cronologia", [])
    nuovo_step = {
        "stato": stato,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }
    if extra:
        nuovo_step.update(extra)
    contenuto["cronologia"].append(nuovo_step)
    contenuto["ultimo_stato"] = nuovo_step
    contenuto["job_id"] = job_id
    blob.upload_from_string(
        json.dumps(contenuto, indent=2),
        content_type="application/json"
    )

def preprocessing(event, context):
    storage_client = storage.Client()
    nome_file = event['name']
    nome_bucket = event['bucket']
    parti_nomi = nome_file.split("_")
    if len(parti_nomi) >= 3:
        job_id = parti_nomi[2]
    else:
        job_id = "unknown"
    try:
        _scrivi_stato(storage_client, job_id, "Preprocessing avviato", {"file_sorgente": nome_file}, reset=True)
        blob_sorgente = storage_client.bucket(nome_bucket).blob(nome_file)
        df = pd.read_csv(io.BytesIO(blob_sorgente.download_as_bytes()), sep=',')
        colonne_numeriche = df.select_dtypes(include=['number']).columns.tolist()
        colonne_feature = [c for c in colonne_numeriche if c != "quality"]
        df[colonne_feature] = df[colonne_feature].fillna(df[colonne_feature].mean(numeric_only=True))
        df[colonne_feature] = (df[colonne_feature] - df[colonne_feature].mean()) / df[colonne_feature].std(ddof=0)
        nome_file_processato = nome_file.rsplit(".", 1)[0] + "_processesato.csv"
        blob_di_output = storage_client.bucket(OUTPUT_BUCKET).blob(nome_file_processato)
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        blob_di_output.upload_from_file(buffer, content_type='text/csv')
        _scrivi_stato(storage_client, job_id, "Preprocessing completato", {
            "file_sorgente": nome_file,
            "file_processato": nome_file_processato
        })
    except Exception as e:
        logging.exception("Errore durante il preprocessing")
        _scrivi_stato(storage_client, job_id,"Errore nel pre processing", {"message": str(e), "file_sorgente": nome_file})
        raise
