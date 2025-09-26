"""
Entrenamiento de modelos con scikit-learn (RandomForest, XGBoost opcional).
"""

from __future__ import annotations
from typing import Tuple, Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Si tienes xgboost instalado, descomenta:
# from xgboost import XGBClassifier

def make_rf_pipeline(preprocessor, **rf_params) -> Pipeline:
    """
    Construye un pipeline: preprocesador + RandomForestClassifier.
    rf_params ej.: n_estimators=300, max_depth=None, class_weight='balanced', random_state=42
    """
    rf_default = dict(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"  # si tu dataset está desbalanceado
    )
    rf_default.update(rf_params or {})
    model = RandomForestClassifier(**rf_default)
    pipe = Pipeline([("pre", preprocessor), ("clf", model)])
    return pipe

# Ejemplo con XGBoost (opcional):
# def make_xgb_pipeline(preprocessor, **xgb_params) -> Pipeline:
#     xgb_default = dict(
#         n_estimators=500,
#         max_depth=5,
#         learning_rate=0.05,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42,
#         n_jobs=-1,
#         eval_metric="logloss",
#     )
#     xgb_default.update(xgb_params or {})
#     model = XGBClassifier(**xgb_default)
#     pipe = Pipeline([("pre", preprocessor), ("clf", model)])
#     return pipe

def fit_model(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    """
    Entrena el pipeline completo.
    """
    pipeline.fit(X_train, y_train)
    return pipeline

def predict_proba(pipeline: Pipeline, X):
    """
    Probabilidades positivas (para métricas como AUC).
    """
    return pipeline.predict_proba(X)[:, 1]

def predict_label(pipeline: Pipeline, X):
    """
    Predicción binaria.
    """
    return pipeline.predict(X)
