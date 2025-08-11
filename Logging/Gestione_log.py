from google.cloud import logging
import streamlit as st

def get_logs_runtime_from_gcp(max_entries=10):
    client = logging.Client()
    entries = client.list_entries(order_by=logging.DESCENDING, page_size=max_entries)
    logs = []
    for entry in entries:
        if hasattr(entry, "text_payload") and entry.text_payload:
            logs.append(entry.text_payload)
        elif hasattr(entry, "json_payload") and entry.json_payload:
            logs.append(str(entry.json_payload))
        elif hasattr(entry, "proto_payload") and entry.proto_payload:
            logs.append(str(entry.proto_payload))
        else:
            logs.append(str(entry))
    return logs

def mostra_stato_preprocessing():
    logs = get_logs_runtime_from_gcp(max_entries=20)
    logs_lower = [log.lower() for log in logs]

    if any("completed" in log or "success" in log for log in logs_lower):
        st.success("Preprocessing completato con successo! Il file è stato caricato su Cloud Storage.")
    elif any("error" in log or "failed" in log for log in logs_lower):
        st.error("Errore durante il preprocessing, controlla i log.")
    else:
        st.warning("Preprocessing in corso o nessun log recente trovato.")
    with st.expander("Mostra log dettagliati"):
        for log in logs:
            st.text(log)
    st.write(f"Numero log: {len(logs)}")
    st.write(logs)

