import pandas as pd
import os
from google.cloud import storage
import io
import logging

OUTPUT_BUCKET = "wine-dataset-preprocessati"


def preprocessing(event, context):
    try:
        logging.info("Preprocessing started")
        file_name = event['name']
        bucket_name = event['bucket']
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        data = blob.download_as_bytes()
        df = pd.read_csv(io.BytesIO(data), sep=';')
        df = df.fillna(df.mean(numeric_only=True))
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
        output_bucket_name = "wine-dataset-preprocessati"
        output_bucket = storage_client.bucket(output_bucket_name)
        processed_file_name = f"{os.path.basename(file_name)}"
        output_blob = output_bucket.blob(processed_file_name)
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        output_blob.upload_from_file(buffer, content_type='text/csv')
        logging.info("Preprocessing completed successfully. File caricato su bucket")
    except Exception as e:
        logging.error(f"Errore durante il preprocessing: {e}")
        raise