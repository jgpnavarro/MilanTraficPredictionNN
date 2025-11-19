"""
Definición de un modelo MLP sencillo para predicción de series temporales,
incluyendo capas de Dropout para reducir sobreajuste.

El modelo trabaja con ventanas de tamaño fijo:
    X_t = [lag_..., ..., lag_1, features_temporales...]

y el objetivo es predecir un valor futuro y(t+H) para un horizonte H fijo.

Arquitectura:
    - Varias capas ocultas densas (Dense) con activación ReLU.
    - Entre capas densas se incluyen capas Dropout para reducir sobreajuste.
    - Una capa de salida con una única neurona (regresión).

Compilación:
    - loss = 'mse' (error cuadrático medio).
    - optimizer = 'adam'.
    - metrics = ['mae'] (error absoluto medio).
"""

from __future__ import annotations

from typing import Sequence

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


def build_mlp(
    input_dim: int,
    hidden_units: Sequence[int] = (64, 32),
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.2,
):
    """
    Construye y compila un modelo MLP para regresión sobre ventanas de serie temporal.

    Parámetros
    ----------
    input_dim : int
        Dimensión de la entrada (número total de columnas de X):
        incluye lags + cualquier feature adicional (hora, día, etc.).
    hidden_units : Sequence[int], opcional
        Número de neuronas en cada capa oculta. Por defecto (64, 32) crea
        dos capas ocultas: la primera con 64 neuronas y la segunda con 32.
    learning_rate : float, opcional
        Tasa de aprendizaje del optimizador Adam.
    dropout_rate : float, opcional
        Proporción de neuronas que se “apagan” aleatoriamente en cada capa
        oculta durante el entrenamiento (valor típico entre 0.1 y 0.5).

    Devuelve
    --------
    model : keras.Model
        Modelo Keras ya compilado, listo para entrenar con model.fit(...).

    Notas
    -----
    - La entrada se considera un vector plano de longitud input_dim.
    - Dropout solo actúa durante el entrenamiento; en predicción se usan
      todas las neuronas (pero con pesos ya “regularizados”).
    """
    model = Sequential()

    # Primera capa oculta con input_dim explícito.
    model.add(
        Dense(
            hidden_units[0],
            activation="relu",
            input_dim=input_dim,
            name="hidden_1",
        )
    )
    if dropout_rate > 0.0:
        # Dropout tras la primera capa oculta
        model.add(
            Dropout(
                dropout_rate,
                name="dropout_1",
            )
        )

    # Capas ocultas adicionales (si las hay), cada una seguida de Dropout
    for i, units in enumerate(hidden_units[1:], start=2):
        model.add(
            Dense(
                units,
                activation="relu",
                name=f"hidden_{i}",
            )
        )
        if dropout_rate > 0.0:
            model.add(
                Dropout(
                    dropout_rate,
                    name=f"dropout_{i}",
                )
            )

    # Capa de salida: una neurona para predecir un valor continuo (regresión).
    model.add(
        Dense(
            1,
            activation="linear",
            name="output",
        )
    )

    # Compilar el modelo con configuración típica para regresión
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mse",      # error cuadrático medio
        metrics=["mae"], # error absoluto medio
    )

    return model
