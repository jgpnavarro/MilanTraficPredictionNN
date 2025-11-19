"""
Modelo de media móvil para series temporales.

Idea:
    Para una serie y(t), una ventana W y un horizonte H:

        - Se calcula la media móvil:
              m(t) = media de los últimos W valores hasta t
        - La predicción se define como:
              ŷ(t+H) = m(t)

    Es decir, la media de los últimos W puntos en t se usa como
    predicción para el valor futuro en t+H.

Este módulo no entrena nada. Solo:
- Calcula la media móvil sobre la serie completa.
- A partir de ella, construye predicciones alineadas para cada split.
- Evalúa métricas por split (train/val/test).

Funciones principales:
- moving_average_forecast: devuelve una serie de predicciones ŷ(t) para toda la serie.
- evaluate_moving_average_split: evalúa métricas en un split concreto.
- evaluate_moving_average_splits: aplica la evaluación a train/val/test y devuelve un dict.
"""

from __future__ import annotations
from typing import Dict
import pandas as pd
from Modeling.metrics import mae, rmse, mape, wmape, smape


def moving_average_forecast(
    series: pd.Series,
    horizon: int,
    window: int,
) -> pd.Series:
    """
    Calcula la predicción por media móvil para toda la serie.

    La idea es:
        1) Calcular la media móvil m(t) a partir de la serie original.
        2) Desplazar esa media 'horizon' pasos hacia adelante:
               y_pred(t) = m(t - horizon)

       De esta forma, en el índice t tendremos una predicción para el
       valor futuro que está 'horizon' pasos por delante del origen
       donde se calculó la media.

    Parámetros
    ----------
    series : pd.Series
        Serie temporal original, con índice de fechas y valores numéricos.
    horizon : int
        Horizonte de predicción en pasos (por ejemplo, 1 o 6).
    window : int
        Tamaño de la ventana de la media móvil (en número de puntos).

    Devuelve
    --------
    y_pred : pd.Series
        Serie con las predicciones de media móvil, alineadas con el
        tiempo del valor futuro. Tendrá NaN al principio (por falta
        de ventana) y también puede tener NaN en los últimos puntos
        si el horizonte es mayor que 0.
    """
    # Calcula la media móvil. min_periods=window exige tener la ventana completa
    # para obtener un valor de media; los primeros puntos quedan como NaN.
    rolling_mean = series.rolling(window=window, min_periods=window).mean()

    # Desplaza la media 'horizon' pasos hacia adelante.
    # Ejemplo:
    #   - rolling_mean en t = media hasta t
    #   - y_pred en t+horizon = rolling_mean en t
    # Para tener y_pred indexada por el instante futuro, aplicamos shift(horizon).
    y_pred = rolling_mean.shift(horizon)

    return y_pred


def evaluate_moving_average_split(
    full_series: pd.Series,
    split_series: pd.Series,
    cell_id: int,
    split_name: str,
    horizon: int,
    window: int,
) -> Dict[str, float]:
    """
    Evalúa la media móvil en un split concreto (train, val o test).

    Se calcula la media móvil sobre la serie completa de la celda
    (full_series) y, a partir de esas predicciones, se selecciona
    solo el intervalo temporal del split (split_series).

    Parámetros
    ----------
    full_series : pd.Series
        Serie completa de la celda (toda la ventana temporal).
    split_series : pd.Series
        Subserie correspondiente al split que se quiere evaluar
        (train, val o test). Se asume que es un subconjunto temporal
        de full_series.
    cell_id : int
        Identificador de la celda (solo informativo, no se usa en el cálculo).
    split_name : str
        Nombre del split ("train", "val", "test", ...). Se usa solo
        para trazas y para estructurar la salida a nivel superior.
    horizon : int
        Horizonte de predicción en pasos (por ejemplo, 1 o 6).
    window : int
        Tamaño de la ventana de la media móvil (en número de puntos).

    Devuelve
    --------
    out : dict
        Diccionario con las métricas y estadísticas del split:
            {
                "n":     número de puntos evaluados,
                "MAE":   ...,
                "RMSE":  ...,
                "MAPE":  ...,
                "wMAPE": ...,
                "sMAPE": ...,
                "Y_MEAN": media de y_true,
                "window": tamaño de la ventana usada
            }

        El campo "window" se incluye para poder identificar qué
        tamaño de ventana se usó en modelos que prueban varias W.
    """
    # 1) Predicción por media móvil sobre la serie completa.
    y_pred_full = moving_average_forecast(
        series=full_series,
        horizon=horizon,
        window=window,
    )

    # 2) Valores reales del split (y_true) y predicciones restringidas
    #    al mismo rango temporal.
    y_true = split_series.astype(float)
    # La indexación por y_true.index garantiza que comparamos en las
    # mismas fechas.
    y_pred = y_pred_full.loc[y_true.index].astype(float)

    # 3) Eliminamos filas donde la predicción sea NaN.
    #    Esto sucede al principio por falta de ventana y también puede
    #    ocurrir cerca del final según el horizonte.
    mask = ~y_pred.isna()
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # Si después de filtrar no queda ningún punto, devolvemos métricas vacías.
    if len(y_true) == 0:
        return {
            "n": 0,
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "MAPE": float("nan"),
            "wMAPE": float("nan"),
            "sMAPE": float("nan"),
            "Y_MEAN": float("nan"),
            "window": int(window),
        }

    # 4) Cálculo de métricas básicas.
    mae_val = mae(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    wmape_val = wmape(y_true, y_pred)
    smape_val = smape(y_true, y_pred)

    out: Dict[str, float] = {
        "n": int(len(y_true)),
        "MAE": float(mae_val),
        "RMSE": float(rmse_val),
        "MAPE": float(mape_val),
        "wMAPE": float(wmape_val),
        "sMAPE": float(smape_val),
        "Y_MEAN": float(y_true.mean()),
        # Campo adicional para saber qué ventana se usó.
        "window": int(window),
    }
    return out


def evaluate_moving_average_splits(
    full_series: pd.Series,
    s_train: pd.Series,
    s_val: pd.Series,
    s_test: pd.Series,
    cell_id: int,
    horizon: int,
    window: int,
) -> Dict[str, Dict[str, float]]:
    """
    Evalúa la media móvil para los tres splits principales: train, val y test.

    Esta función llama a evaluate_moving_average_split para cada split
    y devuelve un diccionario con la misma estructura que en el caso
    de persistencia, lo que facilita reutilizar las funciones de
    reporting ya existentes.

    Parámetros
    ----------
    full_series : pd.Series
        Serie completa de la celda.
    s_train, s_val, s_test : pd.Series
        Subseries correspondientes a cada split.
    cell_id : int
        Identificador de la celda.
    horizon : int
        Horizonte de predicción en pasos.
    window : int
        Tamaño de la ventana de la media móvil.

    Devuelve
    --------
    results : dict
        Diccionario con claves "train", "val", "test", donde cada valor
        es un dict con las métricas para ese split.
    """
    results: Dict[str, Dict[str, float]] = {}

    results["train"] = evaluate_moving_average_split(
        full_series=full_series,
        split_series=s_train,
        cell_id=cell_id,
        split_name="train",
        horizon=horizon,
        window=window,
    )

    results["val"] = evaluate_moving_average_split(
        full_series=full_series,
        split_series=s_val,
        cell_id=cell_id,
        split_name="val",
        horizon=horizon,
        window=window,
    )

    results["test"] = evaluate_moving_average_split(
        full_series=full_series,
        split_series=s_test,
        cell_id=cell_id,
        split_name="test",
        horizon=horizon,
        window=window,
    )

    return results
