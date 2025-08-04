import streamlit as st
import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def avvia_training(processed_dir, models_dir):
    data = pd.read_csv(os.path.join(processed_dir, "wine_processed.csv"))
    X = data.drop(columns=["quality", "wine_type"])
    y = data["quality"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    with open(os.path.join(models_dir, "wine_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    st.success(f"Modello addestrato con accuracy: {score:.2f}")
    return model, score, report

STORAGE_DIR = "./simulated_cloud_storage"
MODEL_DIR = os.path.join(STORAGE_DIR, "models")
PROCESSED_DIR = os.path.join(STORAGE_DIR, "processed_data")

os.makedirs(MODEL_DIR, exist_ok=True)

def train_model():
    # Carica dati già preprocessati
    for file in os.listdir(PROCESSED_DIR):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(PROCESSED_DIR, file))
            break  # Usa il primo disponibile

    X = df.drop("quality", axis=1)
    y = df["quality"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Addestramento Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Valutazione
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)

    # Salva modello
    with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    # Salva metriche per visualizzazione successiva
    with open(os.path.join(MODEL_DIR, "report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(report)

    return acc, report, y_test, y_pred
