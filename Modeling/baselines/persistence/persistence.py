"""
Modelo de persistencia para series temporales.

Definición:
    ŷ(t+H) = y(t)

Este módulo no entrena nada. Solo:
- Construye pares (X, y) para un horizonte H usando targets.make_xy_for_horizon.
- Aplica la regla de persistencia (y_pred = X).
- Calcula métricas por split (train/val/test).

Funciones principales:
- predict_persistence: devuelve la predicción de persistencia para una serie X.
- evaluate_persistence_split: evalúa métricas en un split concreto.
- evaluate_persistence_splits: aplica la evaluación a train/val/test y devuelve un dict con resultados.
"""

from __future__ import annotations

from typing import Dict
import pandas as pd

from Modeling.targets import make_xy_for_horizon_splits
from Modeling.metrics import mae, rmse, mape, wmape, smape #, nrmse_iqr, nrmae_median

def predict_persistence(X: pd.Series) -> pd.Series:
    """
    Regla de persistencia: la predicción futura es el valor actual.

    Entradas:
        X: Serie con valores actuales alineados temporalmente.

    Salida:
        Serie con las predicciones, misma longitud e índice que X.
    """
    # Copia superficial para dejar explícito que se devuelve un objeto 'nuevo' independiente.
    return X.copy()


def evaluate_persistence_split(
    X: pd.Series,
    y: pd.Series,
    median_ref: float | None = None,
    iqr_ref: float | None = None,
) -> Dict[str, float]:
    """
    Evalúa persistencia en un split y devuelve métricas absolutas y normalizadas.
    """
    y_pred = predict_persistence(X)

    out = {
        "n": int(len(y)),
        "MAE": mae(y, y_pred),
        "RMSE": rmse(y, y_pred),
        "MAPE": mape(y, y_pred),
        "wMAPE": wmape(y, y_pred),
        "sMAPE": smape(y, y_pred),
        "Y_MEAN": float(y.mean()), # media de tráfico
    #    "NRMSE_IQR": float("nan"),
    #    "NRMAE_MEDIAN": float("nan"),
    }
    # if (median_ref is not None) and (iqr_ref is not None):
    #     out["NRMSE_IQR"] = nrmse_iqr(y, y_pred, iqr_ref=iqr_ref)
    #     out["NRMAE_MEDIAN"] = nrmae_median(y, y_pred, median_ref=median_ref)
    return out


def evaluate_persistence_splits(
    s_train: pd.Series,
    s_val: pd.Series,
    s_test: pd.Series,
    H: int,
    # median_ref: float | None = None,
    # iqr_ref: float | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Aplica la evaluación de persistencia a los tres splits (train/val/test).

    Entradas:
        s_train: Serie del conjunto de entrenamiento.
        s_val:   Serie del conjunto de validación.
        s_test:  Serie del conjunto de prueba.
        H:       Horizonte de predicción en pasos (1=10min, 6=1h si freq=10min).

    Proceso:
        1) Construir (X, y) por split con 'make_xy_for_horizon'.
        2) Aplicar la regla de persistencia (ŷ = X).
        3) Calcular métricas por split.

    Salida:
        Diccionario con claves 'train', 'val', 'test', cada una con sus métricas:
            {
              "train": {"n":..., "MAE":..., "RMSE":..., "MAPE":...},
              "val":   {"n":..., "MAE":..., "RMSE":..., "MAPE":...},
              "test":  {"n":..., "MAE":..., "RMSE":..., "MAPE":...}
            }
    """

    # Construcción de (X, y) alineados por split para el horizonte H.
    pairs = make_xy_for_horizon_splits(s_train, s_val, s_test, H)
    return {
        "train": evaluate_persistence_split(*pairs["train"]),
        "val":   evaluate_persistence_split(*pairs["val"]),
        "test":  evaluate_persistence_split(*pairs["test"]),
    }