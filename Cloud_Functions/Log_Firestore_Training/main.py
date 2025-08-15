import os, json
from google.cloud import storage, firestore

STATUS_FILE = os.getenv("STATUS_FILE", "status_training.json")

storage_client = storage.Client()
db = firestore.Client()

def status_to_firestore_training(event, context):
    """
    Trigger: finalize su bucket 'wine-dataset-status'
    Copia lo stato del training in Firestore (latest + history)
    """
    bucket_name = event["bucket"]
    file_name = event["name"]

    if file_name != STATUS_FILE:
        return  # ignora altri file di stato

    blob = storage_client.bucket(bucket_name).blob(file_name)
    data = json.loads(blob.download_as_text())

    # Latest
    db.collection("pipeline_status").document("training_latest").set(data)

    # History (usa modello o timestamp come id)
    doc_id = data.get("modello") or data.get("timestamp") or "unknown"
    db.collection("pipeline_status_history").document(f"training_{doc_id}").set(data)
