import streamlit as st
import os
import pandas as pd
from Funzioni.Gestione_file_cartelle import salva_file_locale, verifica_header, unisci_dataset
from Funzioni.Preprocessing import preprocessa_dati
from Funzioni.Modello import addestra_modello
from Funzioni.Inferenza import esegui_inferenza
from Funzioni.Metriche import calcola_metriche
import pickle

# --- Directory simulate ---
RAW_DIR = "./simulated_storage/wine_raw_data"
PROC_DIR = "./simulated_storage/wine_processed_data"
MODEL_DIR = "./simulated_storage/wine_models"
LOG_DIR = "./simulated_storage/wine_logs"

for d in [RAW_DIR, PROC_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

st.set_page_config(page_title="Wine Quality ML Pipeline", layout="wide")
st.subheader("Wine Quality - ML Pipeline Automatica")

st.markdown("---")
file_caricati = st.file_uploader("Caricare uno o più file CSV (vino rosso e/o bianco)", type="csv", accept_multiple_files=True)

if file_caricati:
    with st.spinner("Salvataggio file in corso..."):
        for file in file_caricati:
            salva_file_locale(file, RAW_DIR)

    header_ok, headers_diff = verifica_header([os.path.join(RAW_DIR, f.name) for f in file_caricati])
    if not header_ok:
        st.error("I file caricati hanno header diversi:")
        for idx, header in enumerate(headers_diff):
            st.markdown(f"**Header file {idx + 1}:**")
            st.code(", ".join(header), language="text")
    else:
        st.success("File caricati con header coerente. Inizio pipeline automatica.")
        with st.spinner("Unione e preprocessing in corso..."):
            df_unito = unisci_dataset([os.path.join(RAW_DIR, f.name) for f in file_caricati])
            df_pulito = preprocessa_dati(df_unito)
            path_proc = os.path.join(PROC_DIR, "wine_processed.csv")
            df_pulito.to_csv(path_proc, index=False)
            st.success("Preprocessing completato.")
        with st.spinner("Addestramento modello in corso..."):
            modello, X_test, y_test = addestra_modello(df_pulito)
            with open(os.path.join(MODEL_DIR, "modello.pkl"), "wb") as f:
                pickle.dump(modello, f)
            st.success("Modello addestrato e salvato")
        with st.spinner("Calcolo delle metriche..."):
            metrica_test = calcola_metriche(modello, X_test, y_test)
            st.write("---")
            st.write("### Risultati sul test set:")
            st.text(metrica_test)

            # Converte in DataFrame pandas per visualizzazione tabellare
            df_report = pd.DataFrame(metrica_test).transpose()
            st.dataframe(df_report.style.format("{:.2f}"))  # numeri formattati a 2 decimali

        st.markdown("---")
        st.subheader("🤖 Inferenza su nuovi dati")
        colonne = df_pulito.drop("quality", axis=1).columns.tolist()

        with st.form("form_inferenza"):
            input_vals = []
            for col in colonne:
                val = st.number_input(f"{col}", key=col)
                input_vals.append(val)
            submitted = st.form_submit_button("Invia per predizione")

        if submitted:
            with open(os.path.join(MODEL_DIR, "modello.pkl"), "rb") as f:
                modello = pickle.load(f)
            predizione = esegui_inferenza(modello, input_vals, colonne)
            st.success(f"🎯 Qualità del vino prevista: {predizione}")