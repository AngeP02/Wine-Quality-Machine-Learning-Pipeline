import os, io, json, logging, datetime
from google.cloud import storage
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from collections import Counter

MODEL_BUCKET = os.getenv("MODEL_BUCKET", "wine-dataset-modello")
STATUS_BUCKET = os.getenv("STATUS_BUCKET", "wine-dataset-status")
STATUS_FILE = os.getenv("STATUS_FILE", "status_training.json")


def _scrivi_stato(storage_client, stato, extra=None):
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


def valuta_sbilanciamento(y):
    counter = Counter(y)
    total = len(y)
    class_ratios = {k: v / total for k, v in counter.items()}
    rare_classes = sum(1 for ratio in class_ratios.values() if ratio < 0.05)
    min_ratio = min(class_ratios.values())
    logging.info(f"Distribuzione classi: {dict(counter)}")
    logging.info(f"Classi rare (< 5%): {rare_classes}")
    logging.info(f"Classe più rara: {min_ratio:.3f}")
    return {
        'class_counts': counter,
        'class_ratios': class_ratios,
        'rare_classes': rare_classes,
        'min_ratio': min_ratio,
        'is_severely_imbalanced': rare_classes > 0 and min_ratio < 0.02
    }


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, Counter):
        return dict(obj)
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    else:
        return obj


def training(event, context):
    storage_client = storage.Client()
    file_name = event['name']
    bucket_name = event['bucket']
    try:
        _scrivi_stato(storage_client, "Addestramento avviato", {
            "file_sorgente_processato": file_name
        })
        src_blob = storage_client.bucket(bucket_name).blob(file_name)
        df = pd.read_csv(io.BytesIO(src_blob.download_as_bytes()))
        if "quality" not in df.columns:
            raise ValueError("Manca la colonna 'quality' nel dataset preprocessato.")
        X = df.drop("quality", axis=1)
        y = df["quality"]
        imbalance_info = valuta_sbilanciamento(y)
        _scrivi_stato(storage_client, "Analisi distribuzione completata", {
            "distribuzione_classi": imbalance_info['class_counts'],
            "classi_rare": imbalance_info['rare_classes']
        })
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
            ]
        )
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError as e:
            logging.warning(f"Split stratificato fallito: {e}. Uso split normale.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        model = BalancedRandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2
        )

        final_pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        final_pipeline.fit(X_train, y_train)

        y_pred = final_pipeline.predict(X_test)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        base = file_name.rsplit(".", 1)[0]
        model_filename = f"{base}_balanced_rf_classifier.joblib"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_model_path = os.path.join(script_dir, model_filename)
        joblib.dump({
            "model": final_pipeline,
            "feature_cols": X.columns.tolist(),
            "model_type": "balanced_rf",
            "class_distribution": imbalance_info['class_counts'],
            "preprocessing_info": {
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols
            }
        }, local_model_path)
        storage_client.bucket(MODEL_BUCKET).blob(model_filename).upload_from_filename(
            local_model_path, content_type="application/octet-stream"
        )

        detailed_metrics = {
            "best_model": "balanced_rf",
            "balanced_accuracy": balanced_acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "classification_report": report,
            "confusion_matrix": cm,
            "class_distribution": imbalance_info['class_counts'],
            "imbalance_analysis": imbalance_info
        }
        metrics_blob = storage_client.bucket(MODEL_BUCKET).blob(f"{base}_detailed_metrics.json")
        metrics_blob.upload_from_string(json.dumps(sanitize_for_json(detailed_metrics), indent=2),
                                        content_type="application/json")

        _scrivi_stato(storage_client, "Addestramento completato", {
            "file_processato": file_name,
            "modello": model_filename,
            "miglior_modello": "balanced_rf",
            "balanced_accuracy": balanced_acc,
            "metriche": detailed_metrics
        })

        if os.path.exists(local_model_path):
            os.remove(local_model_path)

    except Exception as e:
        logging.exception("Errore durante l'addestramento")
        _scrivi_stato(storage_client, "Errore nell'addestramento", {
            "message": str(e),
            "file_sorgente_processato": file_name
        })
        raise