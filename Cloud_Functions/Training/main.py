#Training

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


def _scrivi_stato(storage_client, job_id, stato, extra=None, reset=False):
    bucket = storage_client.bucket(STATUS_BUCKET)
    status_file = f"status_training_{job_id}.json"
    blob = bucket.blob(status_file)

    if reset or not blob.exists():
        contenuto = {
            "fase": "training",
            "cronologia": []
        }
    else:
        contenuto = json.loads(blob.download_as_text())

    contenuto.setdefault("cronologia", [])

    nuovo_step = {
        "stato": stato,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }
    if extra:
        nuovo_step.update(extra)

    contenuto["cronologia"].append(nuovo_step)
    contenuto["ultimo_stato"] = nuovo_step
    contenuto["job_id"] = job_id


    blob.upload_from_string(
        json.dumps(contenuto, indent=2),
        content_type="application/json"
    )





def valuta_sbilanciamento(y):
    conteggio = Counter(y)
    totale = len(y)
    rapporti_classe = {k: v / totale for k, v in conteggio.items()}
    classi_rare = sum(1 for ratio in rapporti_classe.values() if ratio < 0.05)
    rapporto_minimo = min(rapporti_classe.values())
    return {
        'class_counts': dict(conteggio),
        'rapporti_classe': rapporti_classe,
        'classi_rare': classi_rare,
        'rapporto_minimo': rapporto_minimo,
        'is_severely_imbalanced': classi_rare > 0 and rapporto_minimo < 0.02
    }


def pulizia_json(obj):
    if isinstance(obj, dict):
        return {k: pulizia_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [pulizia_json(v) for v in obj]
    elif isinstance(obj, Counter):
        return dict(obj)
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    else:
        return obj


def training(event, context):
    client_storage = storage.Client()
    nome_file = event['name']
    nome_bucket = event['bucket']

    parti_nomi = nome_file.split("_")
    if len(parti_nomi) >= 3:
        job_id = parti_nomi[2]
    else:
        job_id = "unknown"
    try:
        _scrivi_stato(client_storage, job_id,"Addestramento avviato", {"file_sorgente": nome_file}, reset=True)
        blob_sorgente = client_storage.bucket(nome_bucket).blob(nome_file)
        df = pd.read_csv(io.BytesIO(blob_sorgente.download_as_bytes()))
        if "quality" not in df.columns:
            raise ValueError("Manca la colonna 'quality' nel dataset preprocessato.")

        X = df.drop("quality", axis=1)
        y = df["quality"]

        info_sbilanciamento = valuta_sbilanciamento(y)
        _scrivi_stato(client_storage, job_id,"Analisi distribuzione completata", {
            "distribuzione_classi": info_sbilanciamento['class_counts'],
            "classi_rare": info_sbilanciamento['classi_rare'],
        })

        colonne_numeriche = X.select_dtypes(include=["number"]).columns.tolist()
        colonne_categoriche = X.select_dtypes(exclude=["number"]).columns.tolist()
        pre_processore = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), colonne_numeriche),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), colonne_categoriche)
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

        modello = BalancedRandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2
        )

        pipeline_finale = ImbPipeline([
            ('pre_processore', pre_processore),
            ('classifier', modello)
        ])
        pipeline_finale.fit(X_train, y_train)

        y_pred = pipeline_finale.predict(X_test)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        base = nome_file.rsplit(".", 1)[0]
        nome_modello = f"{base}_{job_id}_balanced_rf_classifier.joblib"
        directory_script = os.path.dirname(os.path.abspath(__file__))
        percorso_locale_modello = os.path.join(directory_script, nome_modello)
        joblib.dump({
            "modello": pipeline_finale,
            "feature_cols": X.columns.tolist(),
            "model_type": "balanced_rf",
            "class_distribution": info_sbilanciamento['class_counts'],
            "preprocessing_info": {
                "colonne_numeriche": colonne_numeriche,
                "colonne_categoriche": colonne_categoriche
            }
        }, percorso_locale_modello)
        client_storage.bucket(MODEL_BUCKET).blob(nome_modello).upload_from_filename(
            percorso_locale_modello, content_type="application/octet-stream"
        )

        metriche_dettagliate = {
            "best_model": "balanced_rf",
            "balanced_accuracy": balanced_acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "classification_report": report,
            "confusion_matrix": cm,
            "class_distribution": info_sbilanciamento['class_counts'],
            "imbalance_analysis": info_sbilanciamento
        }
        blob_metriche = client_storage.bucket(MODEL_BUCKET).blob(f"{base}_detailed_metrics.json")
        blob_metriche.upload_from_string(json.dumps(pulizia_json(metriche_dettagliate), indent=2),
                                        content_type="application/json")
        _scrivi_stato(client_storage,job_id, "Addestramento completato", {
            "file_processato": nome_file,
            "modello": nome_modello,
            "balanced_accuracy": balanced_acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "confusion_matrix": cm
        })

        if os.path.exists(percorso_locale_modello):
            os.remove(percorso_locale_modello)

    except Exception as e:
        logging.exception("Errore durante l'addestramento")
        _scrivi_stato(client_storage, job_id,"Errore nell'addestramento", {
            "message": str(e),
            "file_sorgente": nome_file
        })
        raise
