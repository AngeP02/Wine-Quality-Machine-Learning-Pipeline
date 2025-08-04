import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import streamlit as st
import pickle

def preprocessa_dfs(dfs_dict, tipo_vini, processed_dir):
    df_totale = pd.DataFrame()
    for fname, df in dfs_dict.items():
        df["wine_type"] = tipo_vini[fname]
        df_totale = pd.concat([df_totale, df], ignore_index=True)

    df_totale.dropna(inplace=True)

    features = df_totale.drop(columns=["quality", "wine_type"])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(scaled_features, columns=features.columns)
    df_scaled["quality"] = df_totale["quality"].values
    df_scaled["wine_type"] = df_totale["wine_type"].values

    output_path = os.path.join(processed_dir, "wine_processed.csv")
    df_scaled.to_csv(output_path, index=False)
    st.success(f"Preprocessing completato. Dati salvati in {output_path}.")
    return df_scaled

def preprocessa_dati(df):
    # Rimuove valori nulli
    df = df.dropna()

    # Encoding variabile "tipo" se presente
    if "tipo" in df.columns:
        df["tipo"] = df["tipo"].map({"red": 0, "white": 1})

    return df