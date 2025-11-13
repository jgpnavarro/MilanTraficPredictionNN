"""
Métricas de evaluación para series temporales (errores punto a punto).

Este módulo define métricas comunes:
- MAE  (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- sMAPE (simetric Mean Absolute Percentage Error)
- wMAPE (weighted Mean Absolute Percentage Error)
- NRMSE_IQR (Normalized Root Mean Squared Error por dispersión robusta)
- NRMAE_Median (normalizado por mediana de la celda)

Las funciones aceptan arrays de NumPy o Series de pandas y devuelven
valores escalares (float). Se asume que y_true y y_pred están alineados
y tienen la misma longitud.
"""

from __future__ import annotations

from typing import Union
import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, pd.Series, list, tuple]


def _to_float_array(x: ArrayLike) -> np.ndarray:
    """
    Convierte la entrada a un array de NumPy de tipo float64.

    Esta función homogeniza la entrada para que las operaciones numéricas
    sean consistentes independientemente del tipo original (Series, list, etc.).
    """
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=np.float64, copy=False)
    return np.asarray(x, dtype=np.float64)


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Mean Absolute Error (MAE).

    Fórmula:
        MAE = mean( |y_true - y_pred| )

    Retorna:
        Error absoluto medio como float.
    """
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Root Mean Squared Error (RMSE).

    Fórmula:
        RMSE = sqrt( mean( (y_true - y_pred)^2 ) )

    Retorna:
        Raíz del error cuadrático medio como float.
    """
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-6) -> float:
    """
    Mean Absolute Percentage Error (MAPE) en porcentaje.

    Fórmula:
        MAPE = mean( |(y_true - y_pred) / max(|y_true|, eps)| ) * 100

    Notas:
        - Se usa 'eps' en el denominador para evitar divisiones por cero
          o valores extremadamente pequeños que inflen el resultado.
        - En series con valores cercanos a cero, MAPE puede ser inestable.
          Interpretar con cautela y acompañar de MAE/RMSE.

    Args:
        y_true: Valores reales.
        y_pred: Valores predichos.
        eps:    Pequeña constante para estabilizar el denominador.

    Retorna:
        Porcentaje de error absoluto medio como float.
    """
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    denom = np.maximum(np.abs(yt), eps)
    return float(np.mean(np.abs((yt - yp) / denom)) * 100.0)

# ... (deja lo que ya tienes: mae, rmse, mape, helpers) ...

def smape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-6) -> float:
    """
    Symmetric MAPE (%):
        sMAPE = mean( |y - yhat| / ((|y| + |yhat|)/2 + eps) ) * 100
    El término 'eps' evita divisiones por cero cuando ambos son muy pequeños.
    """
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    denom = np.maximum(denom, eps)
    return float(np.mean(np.abs(yt - yp) / denom) * 100.0)


def wmape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-6) -> float:
    """
    Weighted MAPE (%):
        wMAPE = (sum |y - yhat|) / (sum |y| + eps) * 100
    Más estable que MAPE clásico en valores pequeños.
    """
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    num = np.sum(np.abs(yt - yp))
    den = np.sum(np.abs(yt))
    return float((num / max(den, eps)) * 100.0)


# MEDIDAS CON POSIBILIDAD DE USO EN EL FUTURO

# def nrmse_iqr(y_true: ArrayLike, y_pred: ArrayLike, iqr_ref: float) -> float:
#     """
#     NRMSE normalizado por IQR (dispersión robusta):
#         NRMSE_IQR = RMSE / IQR_ref
#     'iqr_ref' normalmente se calcula en el TRAIN de la celda.
#     """
#     yt = _to_float_array(y_true)
#     yp = _to_float_array(y_pred)
#     rmse_val = float(np.sqrt(np.mean((yt - yp) ** 2)))
#     if iqr_ref <= 0:
#         # Evita división por cero. Si no hay dispersión en train, devolver NaN.
#         return float("nan")
#     return rmse_val / iqr_ref


# def nrmae_median(y_true: ArrayLike, y_pred: ArrayLike, median_ref: float, eps: float = 1e-6) -> float:
#     """
#     NRMAE normalizado por la mediana:
#         NRMAE_MEDIAN = MAE / max(median_ref, eps)
#     'median_ref' normalmente se calcula en el TRAIN de la celda.
#     """
#     yt = _to_float_array(y_true)
#     yp = _to_float_array(y_pred)
#     mae_val = float(np.mean(np.abs(yt - yp)))
#     denom = max(abs(median_ref), eps)
#     return mae_val / denom


# def ref_stats_for_normalization(series_train: ArrayLike) -> tuple[float, float]:
#     """
#     Calcula los factores de normalización en TRAIN:
#       - mediana (para NRMAE_MEDIAN)
#       - IQR = Q75 - Q25 (para NRMSE_IQR)
#     """
#     st = _to_float_array(series_train)
#     median = float(np.median(st))
#     q75 = float(np.percentile(st, 75))
#     q25 = float(np.percentile(st, 25))
#     iqr = q75 - q25
#     return median, iqr

