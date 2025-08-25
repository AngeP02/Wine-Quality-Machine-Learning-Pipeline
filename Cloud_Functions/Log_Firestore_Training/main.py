#Log Firestore Training

import json
from google.cloud import storage, firestore

STATUS_FILE = "status_training.json"

storage_client = storage.Client()
db = firestore.Client()

def status_to_firestore_training(event, context):
    bucket_name = event["bucket"]
    file_name = event["name"]

    if file_name != STATUS_FILE:
        return

    # scarico lo status file dal bucket
    blob = storage_client.bucket(bucket_name).blob(file_name)
    data = json.loads(blob.download_as_text())

    # estraggo solo la cronologia
    cronologia = data.get("cronologia", [])

    # fix per confusion_matrix (Firestore non accetta liste di liste)
    for step in cronologia:
        if "confusion_matrix" in step:
            step["confusion_matrix"] = [
                {"row": i, "values": row} for i, row in enumerate(step["confusion_matrix"])
            ]

    # documento "corrente"
    db.collection("pipeline_status").document("training_latest").set({
        "fase": "training",
        "cronologia": cronologia
    })

    # storico
    ultimo = cronologia[-1] if cronologia else {}
    doc_id = ultimo.get("modello") or ultimo.get("timestamp") or "unknown"

    db.collection("pipeline_status_history").document(f"training_{doc_id}").set({
        "fase": "training",
        "cronologia": cronologia
    })
