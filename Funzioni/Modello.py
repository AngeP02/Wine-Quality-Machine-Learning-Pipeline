from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import streamlit as st

def mostra_confusion_matrix(y_test, y_pred):
    # Creazione matrice di confusione
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y_test.unique()))

    # Creazione della figura matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax)

    # Visualizzazione in Streamlit
    st.pyplot(fig)

def addestra_modello(df):
    # Separazione tra feature e target
    X = df.drop("quality", axis=1)
    y = df["quality"]

    # Suddivisione in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    class_weights = {
        3: 15.0,  # Aumenta molto il peso delle classi ignorate
        4: 10.0,
        5: 1.0,
        6: 1.0,
        7: 2.0,
        8: 5.0,
    }

    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights)

    # Inizializzazione modello Random Forest
    #clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"  # utile se il dataset è sbilanciato )

    # Addestramento
    clf.fit(X_train, y_train)

    # Valutazione
    y_pred = clf.predict(X_test)
    mostra_confusion_matrix(y_test, y_pred)
    return clf, X_test, y_test

from sklearn.model_selection import GridSearchCV


def addestra_modello_2(df):
    X = df.drop("quality", axis=1)
    y = df["quality"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Definizione del modello
    clf = RandomForestClassifier(random_state=42, class_weight="balanced")

    # Definizione della griglia di iperparametri da testare
    param_grid = {
        'n_estimators': [50, 100, 200],  # Numero di alberi nella foresta
        'max_depth': [5, 10, 20, None],  # Profondità massima dell'albero
        'min_samples_split': [2, 5, 10],  # Numero minimo di campioni richiesti per dividere un nodo
        'min_samples_leaf': [1, 2, 4]  # Numero minimo di campioni richiesti in una foglia
    }

    # Inizializzazione di GridSearchCV
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=5,  # Usa la validazione incrociata a 5 fold
        scoring='f1_weighted',  # La metrica da ottimizzare
        n_jobs=-1
    )

    # Addestramento con la ricerca degli iperparametri
    grid_search.fit(X_train, y_train)

    # Valutazione con il modello migliore
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test)

    return best_clf, X_test, y_test