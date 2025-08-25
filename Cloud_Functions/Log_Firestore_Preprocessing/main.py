#Log Firestore Preprocessing

import json
from google.cloud import storage, firestore

STATUS_FILE = "status_preprocessing.json"

storage_client = storage.Client()
db = firestore.Client()

def status_to_firestore(event, context):
    bucket_name = event['bucket']
    file_name = event['name']
    if file_name != STATUS_FILE:
        return
    blob = storage_client.bucket(bucket_name).blob(file_name)
    data = json.loads(blob.download_as_text())
    db.collection("pipeline_status").document("preprocessing_latest").set(data)
    doc_id = data.get("file_processato") or data.get("timestamp")
    if doc_id:
        db.collection("pipeline_status_history").document(doc_id).set(data)
