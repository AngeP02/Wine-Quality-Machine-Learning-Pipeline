import os
from main import preprocessing  # importa la funzione

class Event:
    def __init__(self, bucket, name):
        self.bucket = bucket
        self.name = name

def test_preprocessing():
    # Simuliamo un file caricato su bucket raw
    event = {
        "bucket": "wine-dataset-originali",  # Sostituisci con il nome reale del tuo bucket
        "name": "winequality-red.csv" # Nome file caricato
    }
    context = None  # Non serve per il test base

    print(">>> Avvio test preprocessing...")
    preprocessing(event, context)
    print(">>> Test completato.")

if __name__ == "__main__":
    test_preprocessing()
