import os
from main import training  # il file che hai incollato lo salvi come main_training.py


def test_training():
    # Evento simulato
    event = {
        "bucket": "wine-dataset-preprocessati",
        "name": "wine-dataset_unito_20250819T152345_f9d355_processesato.csv"
    }
    context = None

    print(">>> Avvio test training...")
    training(event, context)
    print(">>> Test completato.")


if __name__ == "__main__":
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise EnvironmentError("Imposta GOOGLE_APPLICATION_CREDENTIALS con il percorso della chiave di servizio JSON")
    test_training()
