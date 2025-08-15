import os, io, json, logging, datetime
from google.cloud import storage
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# === ENV VARS ===
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "wine-dataset-modello")
STATUS_BUCKET = os.getenv("STATUS_BUCKET", "wine-dataset-status")
STATUS_FILE   = os.getenv("STATUS_FILE", "status_training.json")

def _scrivi_stato(storage_client, stato, extra=None):
    """Scrive lo stato corrente in un JSON 'latest' su STATUS_BUCKET/STATUS_FILE"""
    bucket = storage_client.bucket(STATUS_BUCKET)
    blob = bucket.blob(STATUS_FILE)
    contenuto = {
        "stato": stato,
        "fase": "training",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }
    if extra:
        contenuto.update(extra)
    blob.upload_from_string(json.dumps(contenuto), content_type="application/json")

def training(event, context):
    """
    Trigger: finalize su bucket 'wine-dataset-preprocessati'
    event['bucket'] = bucket sorgente (preprocessati)
    event['name']   = path file CSV preprocessato
    """
    storage_client = storage.Client()
    file_name = event['name']
    bucket_name = event['bucket']

    try:
        _scrivi_stato(storage_client, "Addestramento avviato", {
            "file_sorgente_processato": file_name
        })

        # Scarica CSV preprocessato
        src_blob = storage_client.bucket(bucket_name).blob(file_name)
        df = pd.read_csv(io.BytesIO(src_blob.download_as_bytes()))

        if "quality" not in df.columns:
            raise ValueError("Manca la colonna 'quality' nel dataset preprocessato.")

        # Feature e target
        X = df.drop("quality", axis=1)
        y = df["quality"]

        # Separiamo colonne numeriche e categoriche
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()  # tipo_vino, ecc.

        # Creiamo preprocessore
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
            ]
        )

        # Class weights per bilanciare classi rare
        class_weights = {k: 1.0 for k in y.unique()}

        # Pipeline modello
        clf = Pipeline([
            ("preprocessor", preprocessor),
            ("rf", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                class_weight=class_weights
            ))
        ])

        # Split stratificato
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train
        clf.fit(X_train, y_train)

        # Predizioni e metriche
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Salvataggio modello
        base = file_name.rsplit(".", 1)[0]
        model_filename = f"{base}_rf_classifier.joblib"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_model_path = os.path.join(script_dir, model_filename)
        joblib.dump({"model": clf, "feature_cols": X.columns.tolist()}, local_model_path)

        # Caricamento modello e metriche su bucket
        out_blob = storage_client.bucket(MODEL_BUCKET).blob(model_filename)
        out_blob.upload_from_filename(local_model_path, content_type="application/octet-stream")

        metrics_blob = storage_client.bucket(MODEL_BUCKET).blob(f"{base}_metrics.json")
        metrics_blob.upload_from_string(json.dumps({"classification_report": report, "confusion_matrix": cm}),
                                        content_type="application/json")

        _scrivi_stato(storage_client, "Addestramento completato", {
            "file_processato": file_name,
            "modello": model_filename,
            "metriche": {"classification_report": report, "confusion_matrix": cm}
        })

    except Exception as e:
        logging.exception("Errore durante l'addestramento")
        _scrivi_stato(storage_client, "Errore nell'addestramento", {
            "message": str(e),
            "file_sorgente_processato": file_name
        })
        raise
