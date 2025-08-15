# main.py (Cloud Function preprocessing)
import os, io, json, logging, datetime
from google.cloud import storage
import pandas as pd

OUTPUT_BUCKET = os.getenv("OUTPUT_BUCKET", "wine-dataset-preprocessati")
STATUS_BUCKET = os.getenv("STATUS_BUCKET", "wine-dataset-status")
STATUS_FILE   = os.getenv("STATUS_FILE",   "status_preprocessing.json")

def _scrivi_stato(storage_client, stato, extra=None):
    bucket = storage_client.bucket(STATUS_BUCKET)
    blob = bucket.blob(STATUS_FILE)
    contenuto = {
        "stato": stato,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }
    if extra:
        contenuto.update(extra)
    blob.upload_from_string(json.dumps(contenuto), content_type="application/json")

def preprocessing(event, context):
    storage_client = storage.Client()
    file_name = event['name']
    bucket_name = event['bucket']
    try:
        _scrivi_stato(storage_client, "avviato", {"file_sorgente": file_name})
        src_blob = storage_client.bucket(bucket_name).blob(file_name)
        df = pd.read_csv(io.BytesIO(src_blob.download_as_bytes()), sep=',')
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != "quality"]
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean(numeric_only=True))
        df[feature_cols] = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std(ddof=0)
        processed_file_name = file_name.rsplit(".", 1)[0] + "_processesato.csv"
        out_blob = storage_client.bucket(OUTPUT_BUCKET).blob(processed_file_name)
        buff = io.BytesIO()
        df.to_csv(buff, index=False)
        buff.seek(0)
        out_blob.upload_from_file(buff, content_type='text/csv')
        _scrivi_stato(storage_client, "Pre processing completato", {
            "file_sorgente": file_name,
            "file_processato": processed_file_name
        })
    except Exception as e:
        logging.exception("Errore durante il preprocessing")
        _scrivi_stato(storage_client, "Errore nel pre processing", {"message": str(e), "file_sorgente": file_name})
        raise
