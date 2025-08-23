import io
import pandas as pd
import streamlit as st
from Funzioni.Caricamento_controllo_file import carica_file_su_gcs, genera_nome_univoco
from google.cloud import firestore
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Wine Quality ML Pipeline", layout="wide")
st.subheader("Wine Quality - ML Pipeline Automatica")
st.markdown("---")

if "files_uploaded" not in st.session_state:
    st.session_state.files_uploaded = set()
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "id_run_corrente" not in st.session_state:
    st.session_state.id_run_corrente = None
if "log_content" not in st.session_state:
    st.session_state.log_content = ""
if "log_visibile" not in st.session_state:
    st.session_state.log_visibile = False
if "confusion_matrix_data" not in st.session_state:
    st.session_state.confusion_matrix_data = None

col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    file_caricati = st.file_uploader(
        "Carica uno o più file CSV",
        type="csv",
        accept_multiple_files=True,
        key=f"uploaded_files_{st.session_state.uploader_key}"
    )

with col3:
    st.markdown("<div style='height:50px;'></div>", unsafe_allow_html=True)
    if st.button("Ricarica Pagina"):
        st.session_state.files_uploaded.clear()
        st.session_state.uploader_key += 1
        st.session_state.id_run_corrente = None
        st.session_state.log_content = ""
        st.session_state.log_visibile = False
        st.session_state.confusion_matrix_data = None
        st.rerun()

if file_caricati:
    if not st.session_state.id_run_corrente:
        dfs = []
        nomi_file = []
        file_non_conformi = []
        for file in file_caricati:
            if file.name in st.session_state.files_uploaded:
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
            if len(st.session_state.files_uploaded) > 0:
                st.info(
                    "File già processati in questa sessione. Clicca 'Aggiorna log' per vedere lo stato o 'Ricarica Pagina' per iniziare da capo.")
                st.session_state.log_visibile = True
            else:
                st.warning("Nessun nuovo file valido selezionato.")
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
                st.session_state.id_run_corrente = nome_file_unito
                esito = carica_file_su_gcs("wine-dataset-originali", io.BytesIO(csv_buffer.getvalue().encode()),
                                           nome_file_unito)
                if esito:
                    st.success(f"File caricato correttamente come {nome_file_unito}. La pipeline è stata avviata.")
                    st.info("Cliccare su 'Aggiorna log' per monitorare l'avanzamento.")
                    st.session_state.log_visibile = True
                    st.session_state.log_content = f"Pipeline avviata per il file: {st.session_state.id_run_corrente}\n"
                else:
                    st.error(f"Errore nel caricamento del file {nome_file_unito}")
                    st.session_state.id_run_corrente = None

if st.session_state.log_visibile:
    # Usa le colonne per allineare titolo e pulsante
    col_titolo, col_bottone = st.columns([4, 1])
    with col_titolo:
        st.markdown("#### Stato avanzamento pipeline")
    with col_bottone:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Aggiorna log"):
            id_run_atteso = st.session_state.id_run_corrente
            if not id_run_atteso:
                st.warning("Nessuna esecuzione attiva. Carica un file per iniziare.")
            else:
                db = firestore.Client()
                log_lines = [f"Log per l'esecuzione di {id_run_atteso}\n", "FASE DI PREPROCESSING\n"]
                doc_ref_prep = db.collection("pipeline_status").document("preprocessing_latest")
                doc_prep = doc_ref_prep.get()
                nome_file_processato = None
                if doc_prep.exists:
                    data_prep = doc_prep.to_dict()
                    if data_prep.get("file_sorgente") == id_run_atteso:
                        stato_prep = data_prep.get("stato", "Stato non disponibile")
                        log_lines.append(f"Stato: {stato_prep}")
                        if "file_processato" in data_prep:
                            log_lines.append(f"File salvato come: {data_prep['file_processato']}")
                        nome_file_processato = data_prep.get("file_processato")
                    else:
                        log_lines.append(f"In attesa che la pipeline processi il file {id_run_atteso}...")
                else:
                    log_lines.append("Nessuna informazione di preprocessing disponibile. In attesa...")
                log_lines.append("\nFASE DI ADDESTRAMENTO E VALUTAZIONE\n")
                doc_ref_train = db.collection("pipeline_status").document("training_latest")
                doc_train = doc_ref_train.get()
                st.session_state.confusion_matrix_data = None  # Resetta il dato prima di ricaricare
                if doc_train.exists and nome_file_processato:
                    data_train = doc_train.to_dict()
                    cronologia = data_train.get("cronologia", [])
                    if cronologia and cronologia[0].get("file_sorgente") == nome_file_processato:
                        log_lines.append("Cronologia degli Eventi:")
                        for step in cronologia:
                            ts = step.get("timestamp", "N/A")
                            stato = step.get("stato", "Sconosciuto")
                            log_lines.append(f"- [{ts}] {stato}")
                        ultimo_step_completo = next(
                            (step for step in reversed(cronologia) if "balanced_accuracy" in step), None)
                        if ultimo_step_completo:
                            log_lines.append("\nRisultati Finali:")
                            log_lines.append(f"- Balanced Accuracy: {ultimo_step_completo['balanced_accuracy']:.3f}")
                            log_lines.append(f"- F1 Macro: {ultimo_step_completo['f1_macro']:.3f}")
                            log_lines.append(f"- F1 Weighted: {ultimo_step_completo['f1_weighted']:.3f}")
                            if "confusion_matrix" in ultimo_step_completo:
                                st.session_state.confusion_matrix_data = ultimo_step_completo["confusion_matrix"]
                        else:
                            log_lines.append("\nTraining in corso. Le metriche saranno disponibili al termine.")
                    else:
                        log_lines.append(f"In attesa che la pipeline di training inizi per il file corretto...")
                else:
                    log_lines.append("Nessuna informazione di training disponibile. In attesa...")
                st.session_state.log_content = "\n".join(log_lines)

    # Crea due colonne per affiancare il log e il grafico
    col_log, col_grafico = st.columns([3, 2])

    with col_log:
        st.code(st.session_state.log_content, language="text")

    with col_grafico:
        # Il grafico viene disegnato solo se i dati della matrice sono disponibili nella session_state
        if st.session_state.confusion_matrix_data:
            st.code("Matrice di Confusione")
            cm_data = [row_dict["values"] for row_dict in st.session_state.confusion_matrix_data]
            num_classi = len(cm_data)
            if num_classi == 6:
                etichette_originali = [3 + i for i in range(num_classi)]
            elif num_classi == 7:
                etichette_originali = [3 + i for i in range(num_classi)]
            else:
                etichette_originali = [i for i in range(num_classi)]
            df_cm = pd.DataFrame(cm_data, columns=etichette_originali, index=etichette_originali)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={"size": 10})
            ax.set_xlabel("Classe Predetta", fontsize=6)
            ax.set_ylabel("Classe Reale", fontsize=6)
            st.pyplot(fig)
            plt.close(fig)
