import io
import pandas as pd
import streamlit as st
from Funzioni.Caricamento_controllo_file import carica_file_su_gcs, genera_nome_univoco
from Logging.Gestione_log import mostra_stato_preprocessing

st.set_page_config(page_title="Wine Quality ML Pipeline", layout="wide")
st.subheader("Wine Quality - ML Pipeline Automatica")
st.markdown("---")

# Inizializza variabili di sessione se non esistono
if "files_uploaded" not in st.session_state:
    st.session_state.files_uploaded = set()
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    file_caricati = st.file_uploader(
        "Carica file CSV",
        type="csv",
        accept_multiple_files=True,
        key=f"uploaded_files_{st.session_state.uploader_key}"
    )

with col3:
    st.markdown("<div style='height:50px;'></div>", unsafe_allow_html=True)
    if st.button("Ricarica"):
        # Resetta i file caricati e incrementa la key per resettare il file_uploader
        st.session_state.files_uploaded = set()
        st.session_state.uploader_key += 1
        st.rerun()

if file_caricati:
    dfs = []
    nomi_file = []
    file_non_conformi = []

    for file in file_caricati:
        if file.name in st.session_state.files_uploaded:
            st.info(f"File già caricato in questa sessione: {file.name}")
            continue

        contenuto = file.read()
        try:
            df = pd.read_csv(io.BytesIO(contenuto), sep=";")
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
        st.warning(f"I seguenti file non contengono la colonna 'quality' e sono stati scartati: {file_non_conformi}")

    if len(dfs) == 0:
        st.error("Nessun file valido da caricare.")
    else:
        headers = [tuple(df.columns) for df in dfs]
        set_headers = set(headers)

        if len(set_headers) == 1:
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

            carica_file_su_gcs("wine-dataset-originali", io.BytesIO(csv_buffer.getvalue().encode()), nome_file_unito)
            st.success(f"File unito caricato come {nome_file_unito} su Cloud Storage.")
        else:
            for i, df in enumerate(dfs):
                nome_originale = nomi_file[i]
                if "red" in nome_originale.lower():
                    df["tipo_vino"] = "red"
                elif "white" in nome_originale.lower():
                    df["tipo_vino"] = "white"
                else:
                    df["tipo_vino"] = "Other"

                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False, sep=",")
                nome_file_unico = genera_nome_univoco(nome_originale)

                carica_file_su_gcs("wine-dataset-originali", io.BytesIO(csv_buffer.getvalue().encode()), nome_file_unico)
                st.success(f"File {nome_originale} caricato come {nome_file_unico} su Cloud Storage.")
    #mostra_stato_preprocessing()