import streamlit as st
import pandas as pd
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Wine Quality ML GUI", layout="wide")
st.title("🍷 Wine Quality Classifier - Local Version")

# Percorsi locali per test
RAW_DATA_DIR = "raw_data"
PROCESSED_DATA_DIR = "processed_data"
MODEL_DIR = "models"

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Caricamento file
st.header("1. Carica il Dataset")
uploaded_file = st.file_uploader("Carica file CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    filename = uploaded_file.name
    df.to_csv(os.path.join(RAW_DATA_DIR, filename), index=False)
    st.success("✅ File caricato con successo!")
    st.write(df.head())

    # 2. Preprocessing
    st.header("2. Preprocessing")
    with st.spinner("Esecuzione del preprocessing..."):
        time.sleep(1)
        df = df.dropna()
        st.write("✔️ Valori nulli rimossi")

        if 'quality' not in df.columns:
            st.error("La colonna 'quality' non è presente nel dataset.")
            st.stop()

        X = df.drop('quality', axis=1)
        y = df['quality']

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        X_scaled_df['quality'] = y.values

        processed_path = os.path.join(PROCESSED_DATA_DIR, "processed_" + filename)
        X_scaled_df.to_csv(processed_path, index=False)
        st.success("✅ Preprocessing completato e dati salvati localmente.")

    # 3. Training modello
    st.header("3. Addestramento del modello")
    test_size = st.slider("Scegli la dimensione del test set", 0.1, 0.5, 0.2)
    if st.button("Avvia addestramento"):
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_df.drop("quality", axis=1),
            X_scaled_df["quality"],
            test_size=test_size,
            random_state=42
        )
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        st.write("📊 **Classification Report**")
        st.dataframe(pd.DataFrame(report).transpose())

        model_path = os.path.join(MODEL_DIR, "rf_model.joblib")
        joblib.dump(model, model_path)
        st.success(f"✅ Modello addestrato e salvato localmente in {model_path}")

        # Grafico heatmap
        st.write("📈 **Matrice di Confusione**")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

# 4. Inferenza
st.header("4. Inferenza su Nuovi Dati")
if os.path.exists(os.path.join(MODEL_DIR, "rf_model.joblib")):
    model = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
    st.info("Inserisci valori per una nuova predizione:")

    input_data = {}
    if 'df' in locals():
        input_cols = df.drop("quality", axis=1).columns
        for col in input_cols:
            val = st.number_input(f"{col}", value=0.0, step=0.1, format="%.2f")
            input_data[col] = val

        if st.button("Predici Qualità Vino"):
            new_data = pd.DataFrame([input_data])
            new_data_scaled = scaler.transform(new_data)
            pred = model.predict(new_data_scaled)
            st.success(f"🎯 Qualità predetta del vino: {pred[0]}")
else:
    st.warning("⚠️ Nessun modello addestrato trovato. Addestra prima un modello.")
