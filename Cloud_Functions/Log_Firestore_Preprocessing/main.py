#Log Firestore Preprocessing

import json
from google.cloud import storage, firestore

storage_client = storage.Client()
db = firestore.Client()

def status_to_firestore(event, context):
    nome_bucket = event['bucket']
    nome_file = event['name']
    if not nome_file.startswith("status_preprocessing_"):
        return
    job_id = nome_file.replace("status_preprocessing_", "").replace(".json", "")

    blob = storage_client.bucket(nome_bucket).blob(nome_file)
    dati = json.loads(blob.download_as_text())
    dati["job_id"] = job_id
    db.collection("pipeline_status").document(f"preprocessing_{job_id}").set(dati)
    id_documento = dati.get("file_processato") or dati.get("timestamp") or job_id
    db.collection("pipeline_status_history").document(f"preprocessing_{id_documento}").set(dati)
