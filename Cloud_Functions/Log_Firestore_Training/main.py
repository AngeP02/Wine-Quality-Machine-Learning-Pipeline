#Log Firestore Training

import json
from google.cloud import storage, firestore

storage_client = storage.Client()
db = firestore.Client()

def status_to_firestore_training(event, context):
    nome_bucket = event["bucket"]
    nome_file = event["name"]
    if not nome_file.startswith("status_training_"):
        return
    job_id = nome_file.replace("status_training_", "").replace(".json", "")
    blob = storage_client.bucket(nome_bucket).blob(nome_file)
    dati = json.loads(blob.download_as_text())
    cronologia = dati.get("cronologia", [])
    for step in cronologia:
        if "confusion_matrix" in step:
            step["confusion_matrix"] = [
                {"row": i, "values": row} for i, row in enumerate(step["confusion_matrix"])
            ]
    db.collection("pipeline_status").document(f"training_latest_{job_id}").set({
        "fase": "training",
        "job_id": job_id,
        "cronologia": cronologia
    })
    db.collection("pipeline_status_history").document(f"training_{job_id}").set({
        "fase": "training",
        "job_id": job_id,
        "cronologia": cronologia
    })
