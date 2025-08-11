import pandas as pd

def esegui_inferenza(modello, valori_input, colonne):
    X_nuovo = pd.DataFrame([valori_input], columns=colonne)
    pred = modello.predict(X_nuovo)
    return int(pred[0])