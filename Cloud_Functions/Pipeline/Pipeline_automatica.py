import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def train_model(data_path: str, model_dir: str):
    # Carica dati
    data = pd.read_csv(data_path)

    # Assumiamo 'quality' come target
    X = data.drop("quality", axis=1)
    y = data["quality"]

    # Split (per demo, ma spesso su cloud si usa tutto per training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Definisci modello
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Addestra
    model.fit(X_train, y_train)

    # Valutazione
    preds = model.predict(X_test)
    print("Classification report:\n", classification_report(y_test, preds))

    # Salva modello
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Modello salvato in {model_path}")


if __name__ == "__main__":
    # Variabili d’ambiente o argomenti CLI
    data_path = os.environ.get("DATA_PATH", "/data/wine_quality.csv")
    model_dir = os.environ.get("MODEL_DIR", "/model")

    train_model(data_path, model_dir)
