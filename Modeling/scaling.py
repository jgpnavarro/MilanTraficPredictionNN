"""
Funciones de normalización por máximo para series y splits.

La idea es escalar los datos de cada celda dividiendo por el valor máximo
del split de entrenamiento (train). De esta forma:

    valor_escalado = valor_real / max_train

y, si queremos volver a unidades originales:

    valor_real = valor_escalado * max_train

De esta forma:
    - El modelo trabaja con valores de orden 1 (cómodo para RN).
    - No usamos información de val/test para decidir la escala.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def compute_train_max(s_train: pd.Series) -> float:
    """
    Calcula el valor máximo del split de entrenamiento.

    Si el máximo es 0 o NaN (serie vacía o constante en 0),
    devuelve 1.0 para evitar divisiones por cero.

    Parámetros
    ----------
    s_train : pd.Series
        Serie del split de entrenamiento.

    Devuelve
    --------
    max_train : float
        Valor máximo usado como factor de escala.
    """
    if s_train.empty:
        # No hay datos en train: devolver factor neutro
        return 1.0

    max_val = float(s_train.max())

    if not np.isfinite(max_val) or max_val <= 0.0:
        # Si el máximo no es válido o es 0/negativo, usamos 1.0
        # (en la práctica, significa que no escalamos).
        return 1.0

    return max_val


def scale_series_by_max(
    series: pd.Series,
    max_value: float,
) -> pd.Series:
    """
    Escala una serie dividiéndola por max_value.

    Parámetros
    ----------
    series : pd.Series
        Serie a escalar.
    max_value : float
        Factor de escala (por ejemplo, max_train).

    Devuelve
    --------
    scaled : pd.Series
        Serie escalada. Si max_value == 1.0, se devuelve un escalado neutro.
    """
    if max_value == 1.0:
        # Escalado neutro (útil si la serie es todo ceros o no hay datos)
        return series.astype(float)

    scaled_values = series.to_numpy(dtype=float) / max_value
    scaled = pd.Series(
        data=scaled_values,
        index=series.index,
        name=series.name,
    )
    return scaled


def scale_splits_by_train_max(
    s_train: pd.Series,
    s_val: pd.Series,
    s_test: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.Series, float]:
    """
    Escala los splits train/val/test dividiendo por el máximo de train.

    Parámetros
    ----------
    s_train, s_val, s_test : pd.Series
        Series originales (no escaladas) de cada split temporal.

    Devuelve
    --------
    s_train_scaled, s_val_scaled, s_test_scaled : pd.Series
        Splits escalados, con:
            valor_escalado = valor_real / max_train
    max_train : float
        Máximo de s_train usado como factor de escala.
        Útil para volver a unidades reales:
            valor_real = valor_escalado * max_train
    """
    max_train = compute_train_max(s_train)

    s_train_scaled = scale_series_by_max(s_train, max_train)
    s_val_scaled   = scale_series_by_max(s_val,   max_train)
    s_test_scaled  = scale_series_by_max(s_test,  max_train)

    return s_train_scaled, s_val_scaled, s_test_scaled, max_train
