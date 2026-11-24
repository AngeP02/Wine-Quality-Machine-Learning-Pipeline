# Wine Quality ML Pipeline

Pipeline automatizzata per l'addestramento e l'inferenza di modelli di Machine Learning sul **Wine Quality Dataset**, costruita su architettura serverless con Google Cloud Platform.

Progetto universitario per il corso di **Sistemi Distribuiti e Cloud Computing**.

## Cos'è questo progetto

L'utente carica un dataset CSV di vini tramite un'interfaccia web, e da lì in poi tutto avviene in automatico. Il sistema fa preprocessing, addestra un modello di classificazione e lo rende disponibile per predire la qualità di nuovi vini, il tutto senza che l'utente debba toccare nulla nel mezzo.

L'architettura è interamente serverless, non si ha nessun server sempre acceso, tutto si attiva on-demand tramite eventi su Google Cloud Storage.

## Architettura

```
GUI Streamlit (Cloud Run)
        │
        ▼
Google Cloud Storage  ──trigger──▶  Cloud Function: Pre-processing
        │                                    │
        │                           Cloud Function: Logging Pre-processing
        │                                    │
        │                                    ▼
        │                              Firestore
        │
        └──trigger──▶  Cloud Function: Training
                                │
                       Cloud Function: Logging Training
                                │
                                ▼
                           Firestore
                                │
                       Cloud Function HTTP: Inferenza  ◀──  GUI
```

Il flusso è completamente event-driven, infatti, ogni fase si attiva quando la precedente deposita il proprio output su GCS, senza chiamate dirette tra componenti.

## Componenti principali

**GUI Streamlit (`Cloud_Run/`)**
Interfaccia web deployata su Cloud Run. Permette di:
- Caricare uno o più file CSV del dataset (con validazione automatica);
- Monitorare lo stato della pipeline tramite un pulsante di aggiornamento log;
- Visualizzare metriche, cronologia del training e matrice di confusione;
- Eseguire inferenza inserendo manualmente i valori delle feature.

**Cloud Functions (`Cloud_Functions/`)**

| Funzione | Trigger | Cosa fa |
|---|---|---|
| `preprocessing` | Nuovo file in `wine-dataset-originali` | Pulizia, imputazione valori mancanti, normalizzazione |
| `log_firestore_preprocessing` | Nuovo status JSON in `wine-dataset-status` | Trasferisce lo stato del preprocessing su Firestore |
| `training` | Nuovo file in `wine-dataset-preprocessati` | Addestra il modello e salva metriche + modello serializzato |
| `log_firestore_training` | Nuovo status JSON in `wine-dataset-status` | Trasferisce la cronologia del training su Firestore |
| `inferenza` | HTTP POST | Carica il modello e restituisce la predizione |

**Dataset (`Dati/`)**
Contiene i CSV del Wine Quality Dataset (vino bianco e rosso). Ogni riga rappresenta un campione di vino descritto da 11 feature chimico-fisiche più la variabile target `quality` (scala 3–8).

**Modello ML**

Viene usato un **Balanced Random Forest Classifier** dalla libreria `imbalanced-learn`, scelto per gestire lo sbilanciamento del dataset (le classi intermedie 5 e 6 sono molto più frequenti delle altre). Il modello è incluso in una pipeline scikit-learn che comprende anche preprocessing e OneHotEncoding della variabile categorica `tipo_vino`, così da essere completamente auto-contenuto al momento dell'inferenza.

**Metriche calcolate:**
- Balanced Accuracy
- F1 macro e F1 weighted
- Classification report completo + matrice di confusione
