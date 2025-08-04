import streamlit as st
from Funzioni import Gestione_file_cartelle, Preprocessing, Metriche, Inferenza,Modello
import os

st.set_page_config(page_title="Wine Quality Pipeline", layout="wide")
st.subheader("Wine Quality - Machine Learning Pipeline")

RAW_DIR = "./SimulatedCloud/raw_data"
PROCESSED_DIR = "./SimulatedCloud/processed_data"
MODELS_DIR = "./SimulatedCloud/models"
LOGS_DIR = "./SimulatedCloud/logs"

for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR]: os.makedirs(d, exist_ok=True)

uploaded_files = st.file_uploader("Caricare uno o più file CSV del dataset Wine Quality:", type="csv", accept_multiple_files=True)

if uploaded_files:
    st.markdown("Log")
    header_ok, tipo_vini, dfs = Gestione_file_cartelle.verifica_header_direttamente(uploaded_files)
    if header_ok:
        for file in uploaded_files:
            Gestione_file_cartelle.salva_file_locale(file, RAW_DIR)
        df_pulito = Preprocessing.preprocessa_dfs(dfs, tipo_vini, PROCESSED_DIR)
        st.info("Avvio training automatico")
        model, score, report = Modello.avvia_training(PROCESSED_DIR, MODELS_DIR)
        st.text(report)
        Metriche.mostra_metriche(report)
        st.markdown("Inferenza")
        header = Inferenza.get_header_from_processed(PROCESSED_DIR)
        if header:
            user_input = {}
            cols = st.columns(len(header))
            for i, col_name in enumerate(header):
                user_input[col_name] = cols[i].number_input(f"{col_name}", format="%.2f")
            if st.button("Predici qualità vino"):
                predizione = Inferenza.predici(user_input, MODELS_DIR)
                st.success(f"Predizione qualità del vino: {predizione}")
        else:
            st.warning(
                "Impossibile recuperare header. Assicurarsi che il preprocessing sia stato eseguito correttamente.")
    else:
        st.error("I file hanno header differenti. Procedere separatamente.")

