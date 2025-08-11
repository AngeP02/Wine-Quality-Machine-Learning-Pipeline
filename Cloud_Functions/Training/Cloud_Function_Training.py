import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from google.cloud import storage
import os
import io

def download_csv_from_gcs(gcs_path):
    # gcs_path esempio: gs://bucket-name/file.csv
    client = storage.Client()
    _, bucket_name, *path_parts = gcs_path.split("/")
    blob_path = "/".join(path_parts)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    data = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(data))
    return df

def upload_model_to_gcs(model, gcs_path):
    client = storage.Client()
    _, bucket_name, *path_parts = gcs_path.split("/")
    blob_path = "/".join(path_parts)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    # Serializza modello in memoria
    model_bytes = joblib.dumps(model)
    blob.upload_from_string(model_bytes, content_type='application/octet-stream')
    print(f"Modello salvato su {gcs_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Percorso GCS del CSV preprocessato")
    parser.add_argument("--output_path", required=True, help="Percorso GCS per salvare modello")
    args = parser.parse_args()

    print(f"Scarico dati da {args.input_path} ...")
    df = download_csv_from_gcs(args.input_path)

    # Assumiamo la colonna 'quality' come target
    X = df.drop(columns=["quality"])
    y = df["quality"]

    print("Avvio training modello RandomForest...")
    model = RandomForestClassifier()
    model.fit(X, y)

    # Nome file modello con timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    model_path = os.path.join(args.output_path, f"rf_model_{timestamp}.pkl")

    print(f"Salvo modello su {model_path} ...")
    upload_model_to_gcs(model, model_path)

    print("Training completato!")

if __name__ == "__main__":
    main()
