# Wine Quality ML Pipeline

An automated, serverless Machine Learning pipeline for wine quality classification, built on Google Cloud Platform.

This project was developed as part of the *Distributed Systems and Cloud Computing* course. It demonstrates how to design and deploy an event-driven ML workflow that automates data preprocessing, model training, experiment tracking, and inference.

## Overview

Wine Quality ML Pipeline provides an automated workflow for training and deploying Machine Learning models using the Wine Quality dataset.

Users can upload CSV files through a Streamlit web interface. Once a dataset is uploaded, the system automatically triggers the subsequent stages of the pipeline:

1. Data preprocessing and cleaning.
2. Model training and evaluation.
3. Storage of the trained model and evaluation metrics.
4. Logging and monitoring through Google Cloud Firestore.
5. Model inference through an HTTP endpoint.

The architecture relies on Google Cloud services and event-driven execution, avoiding the need to manage permanently running infrastructure for the individual processing tasks.

## Architecture

The system follows a serverless, event-driven architecture in which Google Cloud Storage acts as the communication layer between pipeline stages.

```text
                 User
                   |
                   v
          Streamlit Web Interface
               (Cloud Run)
                   |
                   v
             Google Cloud
                Storage
                   |
          +--------+--------+
          |                 |
          v                 v
    Preprocessing         Dataset
    Cloud Function       Upload
          |
          v
  Preprocessed Dataset
          |
          v
    Training Cloud
       Function
          |
          +--------------------+
          |                    |
          v                    v
    Trained Model         Training Metrics
          |                    |
          +---------+----------+
                    |
                    v
             Cloud Storage
                    |
                    v
              HTTP Inference
                 Function
                    |
                    v
             Streamlit Interface

        Processing and training logs
                    |
                    v
                 Firestore
```

Each processing stage is triggered by the creation of the corresponding files in Cloud Storage. Separate logging functions transfer execution status information to Firestore, allowing the application to retrieve and display the pipeline history.

## Main Components

### 1. Web Interface — Streamlit and Cloud Run

The Streamlit application provides the user-facing interface and is deployed on Google Cloud Run.

It allows users to:

- Upload one or more CSV files containing wine samples.
- Check the status of preprocessing and training operations.
- Retrieve pipeline execution logs.
- Visualize model evaluation metrics and confusion matrices.
- Inspect the training history.
- Submit individual wine samples for quality prediction.

### 2. Data Preprocessing — Cloud Functions

The preprocessing function is triggered when a new dataset is uploaded to the original dataset bucket.

The preprocessing stage includes:

- Data cleaning.
- Handling of missing values.
- Feature normalization.
- Preparation of the dataset for model training.

The resulting dataset is stored in a dedicated Cloud Storage bucket, triggering the next stage of the pipeline.

### 3. Model Training — Cloud Functions

The training function is automatically activated when a preprocessed dataset becomes available.

It trains a classification model, evaluates its performance, and stores the resulting artifacts, including the serialized model and evaluation metrics.

The project uses a **Balanced Random Forest Classifier**, implemented through the `imbalanced-learn` library.

This algorithm was selected to address the class imbalance in the Wine Quality dataset, where intermediate quality scores are considerably more frequent than the others.

The training pipeline includes the preprocessing steps and categorical feature encoding, allowing the complete model pipeline to be reused during inference.

### 4. Logging and Monitoring — Cloud Firestore

Dedicated Cloud Functions collect execution status information generated during preprocessing and training.

The information is transferred to Google Cloud Firestore, which acts as the persistence layer for pipeline logs and training history.

The Streamlit interface retrieves this information to provide visibility into the execution of the different pipeline stages.

### 5. Model Inference — HTTP Cloud Function

Once a model has been trained, an HTTP-triggered Cloud Function makes it available for inference.

The application sends the input features to the endpoint, which loads the trained model and returns the predicted wine quality.

This separates model training from prediction, allowing the inference service to use the latest available trained model without requiring the training process to run again.

## Dataset

The project uses the **Wine Quality dataset**, containing physicochemical measurements of red and white wine samples.

Each sample is described by 11 physicochemical features and a target variable representing wine quality.

The target values range from 3 to 8.

The input features include measurements such as:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

The dataset also includes a categorical feature identifying the wine type.

## Machine Learning Pipeline

The classification workflow is based on a scikit-learn pipeline integrating data preprocessing and the classification model.

The main steps are:

1. Data preparation and cleaning.
2. Feature preprocessing and normalization.
3. Encoding of the categorical `tipo_vino` feature.
4. Training of a Balanced Random Forest Classifier.
5. Model evaluation.
6. Serialization of the trained pipeline.

### Evaluation Metrics

Model performance is evaluated using:

- Balanced Accuracy
- Macro F1-score
- Weighted F1-score
- Classification Report
- Confusion Matrix

Using multiple evaluation metrics provides a more informative assessment than accuracy alone, particularly in the presence of imbalanced target classes.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | scikit-learn, imbalanced-learn |
| Data Processing | Pandas, NumPy |
| Machine Learning Models | Balanced Random Forest |
| Web Interface | Streamlit |
| Cloud Platform | Google Cloud Platform |
| Serverless Compute | Cloud Functions, Cloud Run |
| Object Storage | Google Cloud Storage |
| Database and Logging | Cloud Firestore |
| Visualization | Matplotlib, Seaborn |

