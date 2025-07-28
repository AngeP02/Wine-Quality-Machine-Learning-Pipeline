# backend_simulation.py
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

STORAGE_DIR = "./simulated_cloud_storage"
RAW_DIR = os.path.join(STORAGE_DIR, "wine_raw_data")
PROCESSED_DIR = os.path.join(STORAGE_DIR, "wine_processed_data")
MODEL_DIR = os.path.join(STORAGE_DIR, "wine_models")
LOG_DIR = os.path.join(STORAGE_DIR, "wine_logs")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def log(msg):
    print(msg)
    with open(os.path.join(LOG_DIR, "pipeline.log"), "a") as f:
        f.write(msg + "\n")

def upload_to_fake_bucket(file):
    file_path = os.path.join(RAW_DIR, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    log(f"[UPLOAD] File caricato: {file.name}")
    return file_path

def preprocessing_trigger(file_path):
    log("[PREPROCESSING] Iniziato...")
    df = pd.read_csv(file_path, sep=';')
    if 'quality' not in df.columns:
        raise ValueError("La colonna 'quality' è obbligatoria.")
    df.dropna(inplace=True)
    X = df.drop(columns=['quality'])
    y = df['quality']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df_processed = pd.DataFrame(X_scaled, columns=X.columns)
    df_processed['quality'] = y.values
    processed_path = os.path.join(PROCESSED_DIR, "processed.csv")
    df_processed.to_csv(processed_path, index=False)
    pickle.dump(scaler, open(os.path.join(PROCESSED_DIR, "scaler.pkl"), "wb"))
    log("[PREPROCESSING] Completato.")
    return processed_path

def training_trigger(processed_path):
    log("[TRAINING] Avviato...")
    df = pd.read_csv(processed_path)
    X = df.drop(columns=['quality'])
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    metrics = classification_report(y_test, y_pred, output_dict=True)
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log(f"[TRAINING] Completato. Accuracy: {acc:.2%}")
    return metrics

def predict_with_model(new_data):
    model = pickle.load(open(os.path.join(MODEL_DIR, "model.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(PROCESSED_DIR, "scaler.pkl"), "rb"))
    scaled_input = scaler.transform([new_data])
    prediction = model.predict(scaled_input)
    return prediction[0]
