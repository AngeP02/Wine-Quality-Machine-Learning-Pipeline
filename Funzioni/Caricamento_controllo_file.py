from google.cloud import storage
import datetime
import uuid

def carica_file_su_gcs(bucket_name, file_obj, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    file_obj.seek(0)
    blob.upload_from_file(file_obj)
    return True

def genera_nome_univoco(nome_base):
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    estensione = nome_base.split(".")[-1]
    nome_senza_ext = ".".join(nome_base.split(".")[:-1])
    nuovo_nome = f"{nome_senza_ext}_{timestamp}_{unique_id}.{estensione}"
    return nuovo_nome