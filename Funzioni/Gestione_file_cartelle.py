
import os
import pandas as pd

def salva_file_locale(file, dest_dir):
    path = os.path.join(dest_dir, file.name)
    with open(path, "wb") as f:
        f.write(file.getbuffer())

def leggi_csv(path):
    return pd.read_csv(path, sep=";")

def verifica_header(lista_path):
    headers = [pd.read_csv(p, nrows=1, sep=";").columns.tolist() for p in lista_path]
    return all(h == headers[0] for h in headers), headers

def unisci_dataset(lista_path):
    df_finale = pd.DataFrame()
    for p in lista_path:
        nome = os.path.basename(p)
        tipo = "red" if "red" in nome.lower() else "white"
        df = pd.read_csv(p, sep=";")
        df['type'] = tipo
        df_finale = pd.concat([df_finale, df], ignore_index=True)
    return df_finale

