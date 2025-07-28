import streamlit as st
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title='Pipeline ML', layout="wide")

# --- Simulated cloud storage directories ---
STORAGE_DIR = "./simulated_cloud_storage"
RAW_DIR = os.path.join(STORAGE_DIR, "wine_raw_data")
PROCESSED_DIR = os.path.join(STORAGE_DIR, "wine_processed_data")
MODEL_DIR = os.path.join(STORAGE_DIR, "wine_models")
LOG_DIR = os.path.join(STORAGE_DIR, "wine_logs")

for d in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)



def log(msg):
    if "logs" not in st.session_state:
        st.session_state.logs = []

    st.session_state.logs.append(msg)
    log_box.code("\n".join(st.session_state.logs), language="text")

    with open(os.path.join(LOG_DIR, "pipeline.log"), "a") as f:
        f.write(msg + "\n")

# Upload multiple files and save in raw_data
def upload_files(files):
    paths = []
    for file in files:
        path = os.path.join(RAW_DIR, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        log(f"[UPLOAD] File caricato: {file.name}")
        paths.append(path)
    return paths

# Check if all csv headers match
def headers_match(paths):
    headers = []
    for p in paths:
        df = pd.read_csv(p, sep=';')
        headers.append(tuple(df.columns))
    return len(set(headers)) == 1

# Add wine_type column based on filename (example: red or white)
def add_wine_type_column(df, filename):
    if "red" in filename.lower():
        df["wine_type"] = "red"
    elif "white" in filename.lower():
        df["wine_type"] = "white"
    else:
        df["wine_type"] = "unknown"
    return df

# Preprocess: merge if headers match, else separate preprocess
def preprocessing(paths):
    log("[PREPROCESSING] Avviato...")
    processed_paths = []
    if headers_match(paths):
        dfs = []
        for p in paths:
            df = pd.read_csv(p, sep=';')
            df = add_wine_type_column(df, os.path.basename(p))
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        combined.dropna(inplace=True)
        # Scaling features (tutte tranne quality e wine_type)
        features = combined.drop(columns=["quality", "wine_type"])
        target = combined["quality"]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        processed_df = pd.DataFrame(features_scaled, columns=features.columns)
        processed_df["quality"] = target.values
        processed_df["wine_type"] = combined["wine_type"].values
        processed_path = os.path.join(PROCESSED_DIR, "processed_combined.csv")
        processed_df.to_csv(processed_path, index=False)
        pickle.dump(scaler, open(os.path.join(PROCESSED_DIR, "scaler_combined.pkl"), "wb"))
        processed_paths.append(processed_path)
        log("[PREPROCESSING] Dataset uniti e preprocessati.")
    else:
        for p in paths:
            df = pd.read_csv(p, sep=';')
            df.dropna(inplace=True)
            features = df.drop(columns=["quality"])
            target = df["quality"]
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            processed_df = pd.DataFrame(features_scaled, columns=features.columns)
            processed_df["quality"] = target.values
            base = os.path.basename(p).replace(".csv", "")
            processed_path = os.path.join(PROCESSED_DIR, f"processed_{base}.csv")
            processed_df.to_csv(processed_path, index=False)
            pickle.dump(scaler, open(os.path.join(PROCESSED_DIR, f"scaler_{base}.pkl"), "wb"))
            processed_paths.append(processed_path)
        log("[PREPROCESSING] Dataset separati preprocessati.")
    log("[PREPROCESSING] Completato.")
    return processed_paths

# Train model(s) on each processed dataset
def training(processed_paths):
    metrics_all = {}
    for p in processed_paths:
        log(f"[TRAINING] Avviato per {os.path.basename(p)}")
        df = pd.read_csv(p)
        # If wine_type is present, escluderla dalle features
        if "wine_type" in df.columns:
            X = df.drop(columns=["quality", "wine_type"])
        else:
            X = df.drop(columns=["quality"])
        y = df["quality"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        base = os.path.basename(p).replace("processed_", "").replace(".csv", "")
        model_path = os.path.join(MODEL_DIR, f"model_{base}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        metrics_all[base] = {"accuracy": acc, "report": report}
        log(f"[TRAINING] Completato per {base}. Accuracy: {acc:.2%}")
    return metrics_all

# Predict using chosen model and input data
def predict(model_key, input_data):
    model_path = os.path.join(MODEL_DIR, f"model_{model_key}.pkl")
    scaler_path = os.path.join(PROCESSED_DIR, f"scaler_{model_key}.pkl") if model_key != "combined" else os.path.join(PROCESSED_DIR, "scaler_combined.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("Modello o scaler non trovati!")
        return None
    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    scaled_input = scaler.transform([input_data])
    prediction = model.predict(scaled_input)
    return prediction[0]


# ------------- STREAMLIT FRONTEND -----------------

st.subheader("Wine Quality Machine Learning Pipeline")
uploaded_files = st.file_uploader("Caricare uno o più file CSV", accept_multiple_files=True, type="csv")

if uploaded_files:
    st.session_state.logs = []
    log_box = st.empty()
    log_box.code("")
    paths = upload_files(uploaded_files)
    processed_paths = preprocessing(paths)
    metrics = training(processed_paths)

    import matplotlib.pyplot as plt

    st.markdown("Metriche dei Modelli Addestrati")

    for key, val in metrics.items():
        st.subheader(f"Dataset: {key}")

        accuracy = val["accuracy"]
        report = val["report"]

        # Estrai metrica per ogni classe (esclude 'accuracy', 'macro avg', ecc.)
        class_metrics = {k: v for k, v in report.items() if k.isdigit()}

        # Tabella metrica
        df_metrics = pd.DataFrame(class_metrics).T
        df_metrics = df_metrics[["precision", "recall", "f1-score", "support"]]
        df_metrics.columns = ["Precision", "Recall", "F1-Score", "Supporto"]
        df_metrics.index.name = "Classe/Qualità"

        col1, col2 = st.columns([2, 2])
        with col1:
            st.write(f"**Accuratezza complessiva**: {accuracy:.2%}")
            st.dataframe(df_metrics.style.format(precision=2))

        with col2:
            fig, ax = plt.subplots(figsize=(2, 2))
            df_metrics["F1-Score"].plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")
            ax.set_title("F1-Score per classe")
            ax.set_ylabel("F1-Score")
            ax.set_xlabel("Classe Qualità")
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)

    st.header("3️⃣ Inferenza")

    # Scegli modello
    modello_scelto = st.selectbox("Scegli modello per predizione", options=list(metrics.keys()))

    # Prepara form input dinamico basato sul dataset processato scelto
    sample_processed = pd.read_csv(processed_paths[0])
    # Per input considera tutte le colonne tranne 'quality' e 'wine_type'
    input_columns = [c for c in sample_processed.columns if c not in ['quality', 'wine_type']]

    st.write("Inserisci i valori per la predizione:")
    input_vals = []
    for col in input_columns:
        val = st.number_input(f"{col}", value=float(sample_processed[col].mean()))
        input_vals.append(val)

    if st.button("🚀 Predici Qualità"):
        pred = predict(modello_scelto, input_vals)
        if pred is not None:
            st.success(f"🔮 Predizione: Qualità = {pred}")
