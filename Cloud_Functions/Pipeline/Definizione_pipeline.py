from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aiplatform_gapic

PROJECT_ID = "pipeline-ml-wine-dataset"
LOCATION = "us-central1"
BUCKET_NAME = "wine-dataset-originali"


CSV_FILENAME = "path/del/file_preprocessato.csv"
DISPLAY_NAME = "wine-quality-dataset"
MODEL_DISPLAY_NAME = "wine-quality-model"

aiplatform.init(project=PROJECT_ID, location=LOCATION)

def crea_dataset_tabular(display_name: str, gcs_uri: str) -> aiplatform.TabularDataset:
    print(f"Creo dataset con dati da: {gcs_uri}")
    dataset = aiplatform.TabularDataset.create(
        display_name=display_name,
        gcs_source=[gcs_uri]
    )
    print(f"Dataset creato: {dataset.resource_name}")
    return dataset

def avvia_training_automl(dataset: aiplatform.TabularDataset, target_column: str, model_display_name: str) -> aiplatform.Model:
    training_job = aiplatform.AutoMLTabularTrainingJob(
        display_name=model_display_name,
        optimization_prediction_type="classification",
        optimization_objective="minimize-log-loss",  # per classificazione multiclasse
    )

    model = training_job.run(
        dataset=dataset,
        target_column=target_column,
        sync=True,  # Attendi il completamento (in locale ok, in cloud conviene asincrono)

    )
    print(f"Training completato. Modello creato: {model.resource_name}")
    return model


def deploy_model(model: aiplatform.Model, endpoint_display_name: str) -> aiplatform.Endpoint:
    print("Deploy del modello in corso...")
    endpoint = model.deploy(
        deployed_model_display_name=endpoint_display_name,
        machine_type="n1-standard-4"  # Puoi cambiare se vuoi
    )
    print(f"Modello deployato con endpoint: {endpoint.resource_name}")
    return endpoint


def main(file_name: str):
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    gcs_uri = f"gs://{BUCKET_NAME}/{file_name}"

    # 1) Crea dataset da CSV pre-processato
    dataset = crea_dataset_tabular(display_name="Wine Quality Dataset", gcs_uri=gcs_uri)

    # 2) Avvia training AutoML
    model = avvia_training_automl(dataset=dataset, target_column="quality", model_display_name="Wine Quality Model")

    # 3) Deploy modello e endpoint
    endpoint = deploy_model(model=model, endpoint_display_name="Wine Quality Endpoint")

    print("Pipeline completata con successo.")


if __name__ == "__main__":
    main("wine-dataset_unito_20250810T154334_8ed44c.csv")

