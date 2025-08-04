import os
import pandas as pd
import streamlit as st

def salva_file_locale(file, dest_dir):
    path = os.path.join(dest_dir, file.name)
    with open(path, "wb") as f:
        f.write(file.getbuffer())
    st.success("Header validato. Avvio preprocessing")

def verifica_header_direttamente(uploaded_files):
    header = None
    tipo_vini = {}
    dfs = {}

    for file in uploaded_files:
        df = pd.read_csv(file, sep=";")
        if header is None:
            header = list(df.columns)
        elif list(df.columns) != header:
            return False, None, None  # Header diverso

        tipo = "rosso" if "red" in file.name.lower() else "bianco"
        tipo_vini[file.name] = tipo
        dfs[file.name] = df

    return True, tipo_vini, dfs