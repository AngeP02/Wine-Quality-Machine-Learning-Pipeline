#Utils

from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError, Forbidden, NotFound
import datetime
import uuid

def carica_file_su_gcs(bucket_name, file_obj, destination_blob_name):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        # Riposiziona il cursore all'inizio
        file_obj.seek(0)

        # Caricamento file
        blob.upload_from_file(file_obj)

        # Verifica se il file esiste davvero nel bucket
        if blob.exists():
            return True
        else:
            return False

    except (GoogleAPIError, Forbidden, NotFound) as e:
        print(f"Errore durante upload: {e}")
        return False


def genera_nome_univoco(nome_base):
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    estensione = nome_base.split(".")[-1]
    nome_senza_ext = ".".join(nome_base.split(".")[:-1])
    nuovo_nome = f"{nome_senza_ext}_{timestamp}_{unique_id}.{estensione}"
    return nuovo_nome
