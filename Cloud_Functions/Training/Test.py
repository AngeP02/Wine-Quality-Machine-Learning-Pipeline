import os
from main import training


def test_training():
    event = {
        "bucket": "wine-dataset-preprocessati",
        "name":"wine-dataset_unito_60963e3cf51f4efba2844806da68e46b_20250830T220637_processesato.csv"
    }
    context = None

    print(">>> Avvio test training...")
    training(event, context)
    print(">>> Test completato.")


if __name__ == "__main__":
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise EnvironmentError("Imposta GOOGLE_APPLICATION_CREDENTIALS con il percorso della chiave di servizio JSON")
    test_training()
