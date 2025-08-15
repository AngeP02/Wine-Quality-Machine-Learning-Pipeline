import io
import pandas as pd
import streamlit as st
from Funzioni.Caricamento_controllo_file import carica_file_su_gcs, genera_nome_univoco
import threading
from google.cloud import firestore
import time
import sys


log_buffer = io.StringIO()
sys.stdout = log_buffer

st.set_page_config(page_title="Wine Quality ML Pipeline", layout="wide")
st.subheader("Wine Quality - ML Pipeline Automatica")
st.markdown("---")

if "files_uploaded" not in st.session_state:
    st.session_state.files_uploaded = set()
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    file_caricati = st.file_uploader( "Carica uno o più file CSV", type="csv", accept_multiple_files=True, key=f"uploaded_files_{st.session_state.uploader_key}" )

with col3:
    st.markdown("<div style='height:50px;'></div>", unsafe_allow_html=True)
    if st.button("Ricarica"):
        st.session_state.files_uploaded.clear()
        st.session_state.uploader_key += 1
        if "listener_started" in st.session_state:
            del st.session_state["listener_started"]

        st.rerun()

if file_caricati:
    passo = 1
    db = firestore.Client()


    def on_snapshot(doc_snapshot, changes, read_time):
        if not doc_snapshot:
            return
        data = doc_snapshot[0].to_dict() or {}
        stato = data.get("stato")
        print(f"{passo}) {stato}")



    def start_firestore_listener():
        doc_ref = db.collection("pipeline_status").document("preprocessing_latest")
        doc_ref.on_snapshot(on_snapshot)


    if "listener_started" not in st.session_state:
        st.session_state.listener_started = True
        thread = threading.Thread(target=start_firestore_listener, daemon=True)
        thread.start()
        time.sleep(0.1)



    dfs = []
    nomi_file = []
    file_non_conformi = []
    for file in file_caricati:
        if file.name in st.session_state.files_uploaded:
            st.info(f"File già caricato in questa sessione: {file.name}")
            continue
        try:
            df = pd.read_csv(io.BytesIO(file.read()), sep=";")
        except Exception as e:
            st.error(f"Errore nel leggere {file.name}: {e}")
            continue
        if "quality" not in df.columns:
            file_non_conformi.append(file.name)
            continue
        dfs.append(df)
        nomi_file.append(file.name)
        st.session_state.files_uploaded.add(file.name)

    if file_non_conformi:
        st.error(f"I seguenti file non sono conformi (manca colonna 'quality'): {file_non_conformi}")
    elif len(dfs) == 0:
        st.error("Nessun file valido.")
    else:
        headers_set = {tuple(df.columns) for df in dfs}
        if len(headers_set) > 1:
            st.error("I file caricati non hanno lo stesso header. Caricare solo file con colonne identiche.")
        else:
            df_unito = pd.concat(dfs, ignore_index=True)
            tipo_vino = []
            for nome in nomi_file:
                if "red" in nome.lower():
                    tipo_vino.append("red")
                elif "white" in nome.lower():
                    tipo_vino.append("white")
                else:
                    tipo_vino.append("Other")
            col_tipo_vino = []
            for i, df in enumerate(dfs):
                col_tipo_vino.extend([tipo_vino[i]] * len(df))
            df_unito["tipo_vino"] = col_tipo_vino
            csv_buffer = io.StringIO()
            df_unito.to_csv(csv_buffer, index=False, sep=",")
            nome_file_unito = genera_nome_univoco("wine-dataset_unito.csv")
            esito = carica_file_su_gcs( "wine-dataset-originali", io.BytesIO(csv_buffer.getvalue().encode()), nome_file_unito )
            if esito:
                st.success(f"File caricato come {nome_file_unito}")
            else:
                st.error(f"Errore nel caricamento del file {nome_file_unito}")

    st.markdown("### Stato avanzamento pipeline")
    st.code(log_buffer.getvalue())
    passo += 1
