import json
from types import SimpleNamespace
from main import predict  # importa la funzione di Cloud Function

class DummyRequest:
    """Simula l'oggetto Request di Flask per test locale"""
    def __init__(self, json_data):
        self._json = json_data

    def get_json(self, silent=False):
        return self._json

def test_inferenza():
    # Istanza di esempio (usa le tue colonne del dataset)
    instances = [{
        "fixed acidity": 8.9,
        "volatile acidity": 0.62,
        "citric acid": 0.19,
        "residual sugar": 3.9,
        "chlorides": 0.17,
        "free sulfur dioxide": 51,
        "total sulfur dioxide": 148,
        "density": 0.9986,
        "pH": 3.17,
        "sulphates": 0.93,
        "alcohol": 9.2,
        "tipo_vino": "red"  # <- aggiungi la colonna categorica
    }]

    request = DummyRequest({"instances": instances})
    response, status_code, headers = predict(request)

    print("Status Code:", status_code)
    print("Response JSON:", response)

if __name__ == "__main__":
    import os
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise EnvironmentError("Imposta GOOGLE_APPLICATION_CREDENTIALS con il percorso della chiave JSON")
    test_inferenza()
