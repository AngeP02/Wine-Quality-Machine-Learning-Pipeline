# trainer/train.py
import os
import argparse
import pandas as pd
from google.cloud import storage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def download_from_gcs(gcs_uri, local_path="/tmp/data.csv"):
    client = storage.Client()
    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1]
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    print(f"Downloaded {gcs_uri} to {local_path}")
    return local_path

def upload_to_gcs(local_path, gcs_uri):
    client = storage.Client()
    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1]
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to {gcs_uri}")

def train_and_save(data_path, model_gcs_dir):
    # scarica se gs://
    if data_path.startswith("gs://"):
        local_csv = "/tmp/data.csv"
        download_from_gcs(data_path, local_csv)
        data_path = local_csv

    df = pd.read_csv(data_path)
    # semplice cleaning: rimuovi righe con target mancante
    df = df.dropna(subset=["quality"])
    X = df.drop("quality", axis=1)
    y = df["quality"]

    # se ci sono colonne non numeriche, converti o drop — qui semplice coerente:
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("Classification report:\n", classification_report(y_test, preds))

    # salva modello localmente e poi carica su GCS
    local_model_path = "/tmp/model.joblib"
    joblib.dump(clf, local_model_path)

    # scegli un nome nel GCS: model_gcs_dir + /model.joblib
    if model_gcs_dir.startswith("gs://"):
        gcs_model_path = model_gcs_dir.rstrip("/") + "/model.joblib"
        upload_to_gcs(local_model_path, gcs_model_path)
    else:
        # salva locale (non consigliato in cloud runtime)
        os.makedirs(model_gcs_dir, exist_ok=True)
        joblib.dump(clf, os.path.join(model_gcs_dir, "model.joblib"))
        print(f"Saved model locally to {model_gcs_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="gs://... path to preprocess csv")
    parser.add_argument("--model_gcs_dir", required=True, help="gs://... target folder to save model")
    args = parser.parse_args()
    train_and_save(args.data_path, args.model_gcs_dir)
