"""
Construcción de objetivos (targets) para horizontes H en series temporales.

Este módulo prepara pares (X, y) para modelos que predicen H pasos hacia delante.
En el caso de persistencia, X es el valor actual e y es el valor en t+H.
No se usan ventanas ni se generan ficheros; todo se realiza en memoria.

Funciones principales:
- make_xy_for_horizon: construye (X, y) alineados para una serie y un H dado.
- make_xy_for_horizon_splits: aplica lo anterior sobre train/val/test.
"""

from __future__ import annotations

from typing import Dict, Tuple
import pandas as pd


def make_xy_for_horizon(series: pd.Series, H: int) -> Tuple[pd.Series, pd.Series]:
    """
    Construye pares (X, y) para un horizonte H.

    Definiciones:
      - X(t) := series(t)                   (valor actual)
      - y(t) := series(t + H)               (valor H pasos adelante)

    Notas:
      - Se utiliza un desplazamiento negativo (shift(-H)) para alinear el objetivo en el índice de X.
      - Los últimos H puntos no tienen objetivo válido y se eliminan automáticamente.

    Args:
        series: Serie temporal con índice datetime y valores numéricos.
        H: Horizonte de predicción en número de pasos (ej. 1 = 10 min, 6 = 1 hora si freq=10 min).

    Returns:
        X_aligned: Serie con los valores actuales.
        y_aligned: Serie con los valores a H pasos, alineada con X_aligned.
    """
    # Objetivo a H pasos hacia delante; genera NaN en las últimas H posiciones.
    y = series.shift(-H)

    # Entrada para persistencia: el valor actual.
    X = series

    # Selección de posiciones válidas (donde y no es NaN).
    valid_mask = y.notna()

    X_aligned = X[valid_mask]
    y_aligned = y[valid_mask]

    # Garantiza nombres informativos en la salida.
    X_aligned.name = "X_current"
    y_aligned.name = f"y_t_plus_{H}"

    return X_aligned, y_aligned


def make_xy_for_horizon_splits(
    s_train: pd.Series,
    s_val: pd.Series,
    s_test: pd.Series,
    H: int,
) -> Dict[str, Tuple[pd.Series, pd.Series]]:
    """
    Aplica make_xy_for_horizon a cada split (train/val/test).

    Args:
        s_train: Serie del conjunto de entrenamiento.
        s_val:   Serie del conjunto de validación.
        s_test:  Serie del conjunto de prueba.
        H:       Horizonte de predicción en pasos.

    Returns:
        Un diccionario con tres entradas:
            {
              "train": (X_train, y_train),
              "val":   (X_val,   y_val),
              "test":  (X_test,  y_test),
            }
        Cada par (X_*, y_*) está alineado y sin NaN de cola por el desplazamiento.
    """
    Xtr, ytr = make_xy_for_horizon(s_train, H)
    Xva, yva = make_xy_for_horizon(s_val,   H)
    Xte, yte = make_xy_for_horizon(s_test,  H)

    return {
        "train": (Xtr, ytr),
        "val":   (Xva, yva),
        "test":  (Xte, yte),
    }
