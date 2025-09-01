import requests

URL_INFERENZA = "https://us-central1-pipeline-ml-wine-dataset.cloudfunctions.net/inferenza"

JOB_ID = "fe771fb53e5b4ccdbb4ccf76773b8d95"

features = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.7,
    "citric acid": 0.0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
    "tipo_vino": "red"
}

payload = {
    "job_id": JOB_ID,
    "features": features
}

response = requests.post(URL_INFERENZA, json=payload)

if response.status_code == 200:
    print("Predizione:", response.json()["qualita_predetta"])
else:
    print("Errore:", response.status_code, response.json())
