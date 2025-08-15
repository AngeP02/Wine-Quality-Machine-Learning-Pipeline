import json
from google.cloud import storage


def leggi_stato_preprocessing():
    client = storage.Client()
    bucket = client.bucket("wine-pipeline-status")
    blob = bucket.blob("status_preprocessing.json")
    if not blob.exists():
        return None
    contenuto = blob.download_as_text()
    return json.loads(contenuto)
