import streamlit as st
from Funzioni import Preprocessing, Gestione_file_cartelle, Modello, Metriche, Inferenza
import os

st.set_page_config(page_title="ML Pipeline Locale", layout="wide")
st.subheader("Wine Quality Machine Learning Pipeline")

uploaded_files = st.file_uploader("Caricare uno o più dataset", type="csv", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        path = Gestione_file_cartelle.save_uploaded_file(file)
    st.success("File caricati")
    datasets = Gestione_file_cartelle.load_csv_files()
    combined = []
    for fname, df in datasets:
        label = "red" if "red" in fname else "white"
        df_processed = Preprocessing.preprocess(df, label)
        Preprocessing.save_processed(df_processed, f"processed_{fname}")
        combined.append(df_processed)
    st.success("Preprocessing completato")
    all_data = []
    for f in os.listdir("Dati/Preprocessati"):
        all_data.append(Preprocessing.pd.read_csv(f"Dati/Preprocessati/{f}"))
    full_df = Preprocessing.pd.concat(all_data)
    model, X_test, y_test = Modello.train_model(full_df)
    Modello.save_model(model)
    st.success("Modello addestrato")

    st.header("4. Valutazione")
    metrics_df = Metriche.evaluate_model(model, X_test, y_test)
    st.dataframe(metrics_df)

st.header("5. Invia nuovi dati per inferenza")
model = Inferenza.load_model()
if model:
    col_names = model.feature_names_in_
    inputs = {}
    for col in col_names:
        inputs[col] = st.number_input(f"{col}", value=0.0)
    if st.button("Predici qualità"):
        pred = Inferenza.make_prediction(model, inputs)
        st.success(f"Predizione: Qualità = {pred}")
