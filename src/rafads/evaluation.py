"""
Módulo para evaluación de modelos y métricas.
"""

from sklearn.metrics import roc_auc_score, classification_report

def evaluate_model(model, X_test, y_test):
    """
    Evalúa un modelo con AUC y reporte de clasificación.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)
    return auc, report
