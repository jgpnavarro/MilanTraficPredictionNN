"""
Construcción de objetivos (targets) para horizontes H en series temporales.

Este módulo prepara pares (X, y) para modelos que predicen H pasos hacia delante.
En el caso de persistencia, X es el valor actual e y es el valor en t+H.
No se usan ventanas ni se generan ficheros; todo se realiza en memoria.

Funciones principales:
- make_xy_for_horizon: construye (X, y) alineados para una serie y un H dado.
- make_xy_for_horizon_splits: aplica lo anterior sobre train/val/test.
- make_windowed_xy_for_horizon: construye (X, y) usando ventanas deslizantes para un horizonte H.
- make_windowed_xy_for_horizon_splits: aplica make_windowed_xy_for_horizon a los splits train/val/test.

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

# ============================================================
# Ventanas deslizantes para modelos tipo MLP / redes neuronales
# ============================================================

def make_windowed_xy_for_horizon(
    series: pd.Series,
    H: int,
    input_window: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Construye (X, y) usando ventanas deslizantes para un horizonte H.

    Para cada instante t donde sea posible:
        - Ventana de entrada (longitud = input_window):
              X_t = [y(t-input_window+1), ..., y(t-1), y(t)]
        - Objetivo:
              y_t = y(t+H)

    Solo se generan muestras cuando:
        - Hay suficientes puntos hacia atrás para la ventana.
        - Hay suficientes puntos hacia delante para el horizonte H.

    Parámetros
    ----------
    series : pd.Series
        Serie original, con índice temporal (DatetimeIndex) y valores numéricos.
    H : int
        Horizonte de predicción en número de pasos.
    input_window : int
        Longitud de la ventana de entrada (número de pasos hacia atrás).

    Devuelve
    --------
    X : pd.DataFrame
        Matriz de entradas de shape (n_muestras, input_window).
        Cada fila es una ventana y el índice es el instante futuro (t+H).
    y : pd.Series
        Vector de objetivos de longitud n_muestras, alineado con el índice de X.

    Notas
    -----
    - Si la serie es demasiado corta para construir al menos una ventana,
      se devuelve X e y vacíos.
    """
    # Asegurarse de que la serie está ordenada por tiempo
    series = series.sort_index()

    values = series.to_numpy(dtype=float)
    index = series.index
    n = len(series)

    # Número mínimo de puntos necesarios:
    # necesitamos 'input_window' hacia atrás y 'H' hacia adelante.
    if n < input_window + H:
        # No hay suficientes datos para construir ni una sola muestra
        empty_X = pd.DataFrame(
            data=[],
            index=pd.DatetimeIndex([], name=series.index.name),
            columns=[f"lag_{i}" for i in range(input_window, 0, -1)],
        )
        empty_y = pd.Series(
            data=[],
            index=pd.DatetimeIndex([], name=series.index.name),
            dtype=float,
        )
        return empty_X, empty_y

    X_rows = []
    y_list = []
    target_index = []

    # t es el índice del "último" punto de la ventana
    # La primera ventana posible termina en t = input_window - 1
    # La última ventana posible debe dejar sitio para H pasos de futuro: t+H <= n-1
    # => t <= n - H - 1
    start_t = input_window - 1
    end_t = n - H - 1

    for t in range(start_t, end_t + 1):
        # Ventana: desde t-input_window+1 hasta t (inclusive)
        window = values[t - input_window + 1 : t + 1]  # shape (input_window,)
        target = values[t + H]  # valor futuro

        X_rows.append(window)
        y_list.append(target)
        target_index.append(index[t + H])  # index del instante futuro

    # Crear DataFrame para X con columnas lag_... y index = instante futuro
    col_names = [f"lag_{i}" for i in range(input_window, 0, -1)]
    X = pd.DataFrame(X_rows, index=pd.DatetimeIndex(target_index), columns=col_names)

    # Crear Series para y con el mismo índice que X
    y = pd.Series(y_list, index=X.index, name=series.name)

    return X, y


def make_windowed_xy_for_horizon_splits(
    s_train: pd.Series,
    s_val: pd.Series,
    s_test: pd.Series,
    H: int,
    input_window: int,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Aplica make_windowed_xy_for_horizon a los splits train/val/test.

    Parámetros
    ----------
    s_train, s_val, s_test : pd.Series
        Series correspondientes a cada split temporal.
    H : int
        Horizonte de predicción en número de pasos.
    input_window : int
        Longitud de la ventana de entrada (número de pasos hacia atrás).

    Devuelve
    --------
    dict
        Diccionario con:
            {
              "train": (X_train, y_train),
              "val":   (X_val,   y_val),
              "test":  (X_test,  y_test),
            }
        Cada par (X_*, y_*) está alineado, sin NaN y con índice temporal
        correspondiente al instante futuro (t+H) de cada ventana.
    """
    Xtr, ytr = make_windowed_xy_for_horizon(s_train, H, input_window)
    Xva, yva = make_windowed_xy_for_horizon(s_val,   H, input_window)
    Xte, yte = make_windowed_xy_for_horizon(s_test,  H, input_window)

    return {
        "train": (Xtr, ytr),
        "val":   (Xva, yva),
        "test":  (Xte, yte),
    }
