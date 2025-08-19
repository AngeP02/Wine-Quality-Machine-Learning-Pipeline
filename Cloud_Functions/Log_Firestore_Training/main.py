import json
from google.cloud import storage, firestore

STATUS_FILE ="status_training.json"

storage_client = storage.Client()
db = firestore.Client()

def status_to_firestore_training(event, context):
    bucket_name = event["bucket"]
    file_name = event["name"]
    if file_name != STATUS_FILE:
        return
    blob = storage_client.bucket(bucket_name).blob(file_name)
    data = json.loads(blob.download_as_text())
    db.collection("pipeline_status").document("training_latest").set(data)
    doc_id = data.get("modello") or data.get("timestamp") or "unknown"
    db.collection("pipeline_status_history").document(f"training_{doc_id}").set(data)
