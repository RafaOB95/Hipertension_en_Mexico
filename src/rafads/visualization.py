"""
Visualización: ROC y matriz de confusión (rápidas).
"""

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

def plot_roc(model, X_test, y_test):
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("Curva ROC")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, X_test, y_test):
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Matriz de Confusión")
    plt.tight_layout()
    plt.show()
