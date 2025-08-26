import io
import pandas as pd
import streamlit as st
from Utils import carica_file_su_gcs, genera_nome_univoco
from google.cloud import firestore
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import storage
import joblib

st.set_page_config(page_title="Wine Quality ML Pipeline", layout="wide")
st.subheader("Wine Quality - ML Pipeline Automatica")
st.markdown("---")

if "file_caricati" not in st.session_state:
    st.session_state.file_caricati = set()
if "chiave_caricamento" not in st.session_state:
    st.session_state.chiave_caricamento = 0
if "id_run_corrente" not in st.session_state:
    st.session_state.id_run_corrente = None
if "contenuto_log" not in st.session_state:
    st.session_state.contenuto_log = ""
if "log_visibile" not in st.session_state:
    st.session_state.log_visibile = False
if "dati_confusion_matrix" not in st.session_state:
    st.session_state.dati_confusion_matrix = None

col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    lista_file_caricati = st.file_uploader(
        "Carica uno o più file CSV",
        type="csv",
        accept_multiple_files=True,
        key=f"file_caricati_id_{st.session_state.chiave_caricamento}"
    )

with col3:
    st.markdown("<div style='height:50px;'></div>", unsafe_allow_html=True)
    if st.button("Ricarica Pagina"):
        st.session_state.file_caricati.clear()
        st.session_state.chiave_caricamento += 1
        st.session_state.id_run_corrente = None
        st.session_state.contenuto_log = ""
        st.session_state.log_visibile = False
        st.session_state.dati_confusion_matrix = None
        st.rerun()

if lista_file_caricati:
    if not st.session_state.id_run_corrente:
        lista_dataframe = []
        nomi_file = []
        file_non_conformi = []
        for file in lista_file_caricati:
            if file.name in st.session_state.file_caricati:
                continue
            try:
                elemento_corrente = pd.read_csv(io.BytesIO(file.read()), sep=";")
            except Exception as e:
                st.error(f"Errore nel leggere {file.name}: {e}")
                continue
            if "quality" not in elemento_corrente.columns:
                file_non_conformi.append(file.name)
                continue
            lista_dataframe.append(elemento_corrente)
            nomi_file.append(file.name)
            st.session_state.file_caricati.add(file.name)

        if file_non_conformi:
            st.error(f"Nei seguenti file manca la colonna 'quality': {file_non_conformi}")
        elif len(lista_dataframe) == 0:
            if len(st.session_state.file_caricati) > 0:
                st.info("File già processati in questa sessione. Cliccare su 'Aggiorna log' per vedere lo stato o 'Ricarica Pagina' per iniziare da capo.")
                st.session_state.log_visibile = True
            else:
                st.warning("Nessun nuovo file valido selezionato.")
        else:
            headers_set = set()
            for elemento_corrente in lista_dataframe:
                intestazioni_correnti = tuple(elemento_corrente.columns)
                headers_set.add(intestazioni_correnti)
            if len(headers_set) > 1:
                st.error("I file caricati non hanno lo stesso header. Caricare solo file con set di colonne identiche.")
            else:
                df_unito = pd.concat(lista_dataframe, ignore_index=True)
                tipo_vino = []
                for nome in nomi_file:
                    if "red" in nome.lower():
                        tipo_vino.append("red")
                    elif "white" in nome.lower():
                        tipo_vino.append("white")
                    else:
                        tipo_vino.append("other")
                colonna_tipo_vino = []
                for indice, elemento_corrente in enumerate(lista_dataframe):
                    colonna_tipo_vino.extend([tipo_vino[indice]] * len(elemento_corrente))
                df_unito["tipo_vino"] = colonna_tipo_vino
                csv_buffer = io.StringIO()
                df_unito.to_csv(csv_buffer, index=False, sep=",")
                nome_file_unito = genera_nome_univoco("wine-dataset_unito.csv")
                st.session_state.id_run_corrente = nome_file_unito
                esito = carica_file_su_gcs("wine-dataset-originali", io.BytesIO(csv_buffer.getvalue().encode()), nome_file_unito)
                if esito:
                    st.success(f"File caricato correttamente come {nome_file_unito}. La pipeline è stata avviata.")
                    st.info("Cliccare su 'Aggiorna log' per monitorare l'avanzamento.")
                    st.session_state.log_visibile = True
                    st.session_state.contenuto_log = f"Pipeline avviata per il file: {st.session_state.id_run_corrente}\n"
                else:
                    st.error(f"Errore nel caricamento del file {nome_file_unito}")
                    st.session_state.id_run_corrente = None

if st.session_state.log_visibile:
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
                riga_del_log = [f"Log per l'esecuzione di {id_run_atteso}\n", "FASE DI PREPROCESSING\n"]
                documento_recente_stato_preprocessing = db.collection("pipeline_status").document("preprocessing_latest")
                documento_recuperato = documento_recente_stato_preprocessing.get()
                nome_file_processato = None
                if documento_recuperato.exists:
                    dati_documento_recuperato = documento_recuperato.to_dict()
                    if dati_documento_recuperato.get("file_sorgente") == id_run_atteso:
                        valore_stato_preprocessing = dati_documento_recuperato.get("stato", "Stato non disponibile")
                        riga_del_log.append(f"Stato: {valore_stato_preprocessing}")
                        if "file_processato" in dati_documento_recuperato:
                            riga_del_log.append(f"File salvato come: {dati_documento_recuperato['file_processato']}")
                        nome_file_processato = dati_documento_recuperato.get("file_processato")
                    else:
                        riga_del_log.append(f"In attesa che la pipeline processi il file {id_run_atteso}...")
                else:
                    riga_del_log.append("Nessuna informazione di preprocessing disponibile. In attesa...")


                riga_del_log.append("\nFASE DI ADDESTRAMENTO E VALUTAZIONE\n")
                riferimento_file_addestramento_recente_firestore = db.collection("pipeline_status").document("training_latest")
                documento_recuperato_training = riferimento_file_addestramento_recente_firestore.get()
                st.session_state.dati_confusion_matrix = None
                if documento_recuperato_training.exists and nome_file_processato:
                    dati_training = documento_recuperato_training.to_dict()
                    cronologia = dati_training.get("cronologia", [])
                    if cronologia and cronologia[0].get("file_sorgente") == nome_file_processato:
                        riga_del_log.append("Cronologia degli Eventi:")
                        for passo in cronologia:
                            timestamp = passo.get("timestamp", "N/A")
                            stato_addestramento = passo.get("stato", "Sconosciuto")
                            riga_del_log.append(f"- [{timestamp}] {stato_addestramento}")
                        ultimo_step_completo = next(
                            (step for step in reversed(cronologia) if "balanced_accuracy" in step), None)
                        if ultimo_step_completo:
                            riga_del_log.append("\nRisultati Finali:")
                            riga_del_log.append(f"- Balanced Accuracy: {ultimo_step_completo['balanced_accuracy']:.3f}")
                            riga_del_log.append(f"- F1 Macro (media non pesata): {ultimo_step_completo['f1_macro']:.3f}")
                            riga_del_log.append(f"- F1 Weighted (media pesata): {ultimo_step_completo['f1_weighted']:.3f}")
                            if "confusion_matrix" in ultimo_step_completo:
                                st.session_state.dati_confusion_matrix = ultimo_step_completo["confusion_matrix"]
                        else:
                            riga_del_log.append("\nTraining in corso. Le metriche saranno disponibili al termine.")
                    else:
                        riga_del_log.append(f"In attesa che la pipeline di training inizi per il file corretto...")
                else:
                    riga_del_log.append("Nessuna informazione di training disponibile. In attesa...")
                st.session_state.contenuto_log = "\n".join(riga_del_log)

    col_log, col_grafico = st.columns([3, 2])
    with col_log:
        st.code(st.session_state.contenuto_log, language="text")
    with col_grafico:
        if st.session_state.dati_confusion_matrix:
            st.code("Matrice di Confusione")
            dati_matrice_confusione = []
            for riga_cm in st.session_state.dati_confusion_matrix:
                dati_matrice_confusione.append(riga_cm["values"])
            num_classi = len(dati_matrice_confusione)
            if num_classi == 6:
                etichette_originali = [3 + i for i in range(num_classi)]
            elif num_classi == 7:
                etichette_originali = [3 + i for i in range(num_classi)]
            else:
                etichette_originali = [i for i in range(num_classi)]
            dataframe_confusion_matrix = pd.DataFrame(dati_matrice_confusione, columns=etichette_originali, index=etichette_originali)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(dataframe_confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={"size": 10})
            ax.set_xlabel("Classe Predetta", fontsize=6)
            ax.set_ylabel("Classe Reale", fontsize=6)
            st.pyplot(fig)
            plt.close(fig)

if st.session_state.dati_confusion_matrix:
    st.markdown("---")
    st.markdown("#### Inferenza sul Modello Addestrato")
    BUCKET_MODELLI = "wine-dataset-modello"
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_MODELLI)
    file_nel_bucket = list(bucket.list_blobs())
    modelli = [b for b in file_nel_bucket if b.name.endswith(".joblib")]
    if not modelli:
        st.warning("Nessun modello trovato nel bucket.")
    else:
        modelli.sort(key=lambda b: b.time_created, reverse=True)
        ultimo_modello = modelli[0]
        st.info(f"Modello caricato: **{ultimo_modello.name}**")
        dati_binari_modello = ultimo_modello.download_as_bytes()
        modello_salvato = joblib.load(io.BytesIO(dati_binari_modello))
        modello = modello_salvato["model"]
        features = modello_salvato["feature_cols"]
        if features:
            st.write("Inserire i valori delle feature")
            valori_input = {}
            tipo_vino_selezionato = "red"
            with st.form("Inferenza"):
                for feat in features:
                    if feat == "tipo_vino":
                        tipo_vino_selezionato = st.selectbox("Tipo Vino", ["red", "white", "other"])
                    else:
                        valori_input[feat] = st.number_input(f"{feat}", value=0.0, step=0.1, format="%.5f")
                confermati = st.form_submit_button("Predizione")

            if confermati:
                X = pd.DataFrame([valori_input], columns=[f for f in features if f != "tipo_vino"])
                if "tipo_vino" in features:
                    X["tipo_vino"] = tipo_vino_selezionato
                predizione = modello.predict(X)[0]
                st.success(f"Qualità Predetta: **{predizione}**")