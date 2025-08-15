import os
from main import training  # il file che hai incollato lo salvi come main_training.py


def test_training():
    # Evento simulato
    event = {
        "bucket": "wine-dataset-preprocessati",  # bucket sorgente
        "name": "wine-dataset_unito_20250815T221944_0c17f0_processesato.csv"  # file CSV preprocessato
    }
    context = None

    print(">>> Avvio test training...")
    training(event, context)
    print(">>> Test completato.")


if __name__ == "__main__":
    # Assicurati di avere le credenziali attive
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise EnvironmentError("Imposta GOOGLE_APPLICATION_CREDENTIALS con il percorso della chiave di servizio JSON")

    test_training()
