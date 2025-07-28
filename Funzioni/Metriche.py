# logic/evaluation.py

from sklearn.metrics import classification_report
import pandas as pd

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return pd.DataFrame(report).transpose()
