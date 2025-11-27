"""
Métricas de evaluación para series temporales (errores punto a punto).

Este módulo define métricas comunes:
- MAE  (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- sMAPE (simetric Mean Absolute Percentage Error)
- wMAPE (weighted Mean Absolute Percentage Error)

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


