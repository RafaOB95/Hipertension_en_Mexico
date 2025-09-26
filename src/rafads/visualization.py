"""
Módulo para visualización de resultados.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

def plot_roc(model, X_test, y_test):
    """
    Dibuja la curva ROC de un modelo.
    """
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.show()


def plot_confusion_matrix(model, X_test, y_test):
    """
    Dibuja la matriz de confusión.
    """
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.show()
