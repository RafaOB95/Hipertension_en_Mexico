"""
Módulo para carga, limpieza y preparación de datos.
"""

import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Carga datos desde un archivo CSV.
    Args:
        path (str): ruta al archivo CSV.
    Returns:
        pd.DataFrame: dataframe con los datos cargados.
    """
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza básica del dataset.
    Ajusta esta función con los pasos que ya usas en tus notebooks.
    """
    df = df.dropna()
    return df
