"""
Módulo para entrenamiento de modelos.
"""

from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Entrena un modelo Random Forest.
    Args:
        X_train: features de entrenamiento
        y_train: etiquetas de entrenamiento
    Returns:
        modelo entrenado
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model
