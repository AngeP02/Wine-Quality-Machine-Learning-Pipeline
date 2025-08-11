# preprocess/main.py
from google.cloud import storage
import pandas as pd
import io
import os

storage_client = storage.Client()

OUTPUT_BUCKET = os.environ.get("PREPROCESSED_BUCKET", "wine-dataset-preprocessed")


def preprocess(data, context):
    """Triggered by a change to a Cloud Storage bucket.
    data is the Cloud Functions event payload."""
    bucket_name = data['bucket']
    name = data['name']
    print(f"preprocess triggered for: gs://{bucket_name}/{name}")

    src_bucket = storage_client.bucket(bucket_name)
    blob = src_bucket.blob(name)
    raw_bytes = blob.download_as_bytes()

    # prova a leggere CSV: prova con separatore , e ; automatico
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine='python')
    except Exception:
        # fallback comma
        df = pd.read_csv(io.BytesIO(raw_bytes))

    # ESEMPIO di pre-processing minimale:
    # - rimuovi righe/colonne vuote
    # - strip colonne nome
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(how='all')  # rimuovi righe tutte NaN
    df = df.dropna(axis=1, how='all')  # rimuovi colonne tutte NaN

    # Assicurati che ci sia la colonna tipo_vino (già dalla GUI la aggiungi,
    # ma se manca aggiungila come unknown)
    if "tipo_vino" not in df.columns:
        df["tipo_vino"] = "unknown"

    # Altri controlli possibili (es. quality presente)
    if "quality" not in df.columns:
        print(f"File {name} non contiene colonna 'quality'. Skippato.")
        return

    # Salva su bucket preprocessato con prefisso/prezzo (mantieni nome unico)
    out_bucket = storage_client.bucket(OUTPUT_BUCKET)
    out_name = f"preprocessed_{name}"
    out_blob = out_bucket.blob(out_name)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    out_blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")
    print(f"Preprocessed file written to gs://{OUTPUT_BUCKET}/{out_name}")
