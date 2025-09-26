"""
Carga, limpieza y preparación de datos para ML.
Ajusta las variables TARGET y listas de columnas si tu dataset lo requiere.
"""

from __future__ import annotations
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# === EDITA AQUÍ SEGÚN TU DATASET ===
TARGET = "hipertension"  # Cambia al nombre real de tu variable objetivo
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_csv(path: str, sep: str = ",") -> pd.DataFrame:
    """Carga un CSV a DataFrame."""
    return pd.read_csv(path, sep=sep)

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza mínima: elimina duplicados y estandariza nombres de columnas.
    Agrega aquí las reglas de tu notebook (outliers, rangos válidos, etc.).
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.drop_duplicates()
    return df

def infer_feature_types(df: pd.DataFrame, target: str = TARGET) -> Tuple[List[str], List[str]]:
    """
    Infiera columnas numéricas y categóricas automáticamente (excluye TARGET).
    Ajusta si necesitas listas fijas.
    """
    num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != target]
    cat_cols = [
        c for c in df.columns
        if c != target and (df[c].dtype.name in ["object", "category", "bool"])
    ]
    return num_cols, cat_cols

def build_preprocessor(
    num_cols: List[str], cat_cols: List[str],
    scale_numeric: bool = True
) -> ColumnTransformer:
    """
    ColumnTransformer con imputación y, opcionalmente, escalado en numéricas; OHE en categóricas.
    """
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(steps=num_steps)

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        n_jobs=None,
    )
    return pre

def prepare_xy(
    df: pd.DataFrame,
    target: str = TARGET,
    fixed_num_cols: Optional[List[str]] = None,
    fixed_cat_cols: Optional[List[str]] = None,
    scale_numeric: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer, List[str], List[str]]:
    """
    Separa X,y y construye el preprocesador.
    """
    if target not in df.columns:
        raise ValueError(f"TARGET '{target}' no está en las columnas: {df.columns.tolist()}")

    if fixed_num_cols is None or fixed_cat_cols is None:
        num_cols, cat_cols = infer_feature_types(df, target=target)
    else:
        num_cols, cat_cols = fixed_num_cols, fixed_cat_cols

    X = df.drop(columns=[target])
    y = df[target].copy()

    pre = build_preprocessor(num_cols, cat_cols, scale_numeric=scale_numeric)
    return X, y, pre, num_cols, cat_cols

def train_val_test_split(
    X: pd.DataFrame, y: pd.Series,
    test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
):
    """
    Split estratificado por la variable objetivo.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
