"""
Definición de un modelo GRU sencillo para predicción de series temporales.

Este modelo está pensado para trabajar con ventanas de tamaño fijo sobre una
serie temporal de una celda. Cada ventana es una secuencia de varios pasos
(timesteps) y en cada paso podemos tener una o varias "features" (features = 
características numéricas de ese instante de tiempo).

Ejemplos de features por instante:
    - Valor de tráfico normalizado (internet_total escalado).
    - Hora del día (0–23), derivada del timestamp.
    - Día de la semana (0=Lunes, ..., 6=Domingo), derivado del timestamp.
    - Hay descanso académico o es fiesta en Milán (0 no es, 1 si es)

Si usamos estas tres, cada paso de la secuencia tendrá 3 números:
    [trafico, hora_del_dia, dia_de_la_semana]

En general, la entrada a la GRU tiene forma:
    (timesteps, n_features)

y el objetivo del modelo es predecir un valor futuro y(t+H) para un horizonte H.
"""

from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.optimizers import Adam


def build_gru(
    input_timesteps: int,
    n_features: int = 1,
    gru_units: int = 32,
    learning_rate: float = 1e-3,
) -> Sequential:
    """
    Construye y compila un modelo GRU sencillo para regresión sobre series temporales.

    Parámetros
    ----------
    input_timesteps : int
        Número de pasos de la secuencia de entrada (longitud de la ventana W).
        En este proyecto se suele usar W = 12 (2 horas con datos cada 10 minutos).

    n_features : int, opcional (por defecto 1)
        Número de features (características) por cada paso de la secuencia.
        Algunos ejemplos:
            - n_features = 1:
                solo se usa el valor de tráfico normalizado.
            - n_features = 3:
                tráfico normalizado + hora del día + día de la semana.

        El modelo no sabe qué significan las features; simplemente recibe
        n_features números por cada paso temporal.

    gru_units : int, opcional (por defecto 32)
        Número de unidades (neuronas) de la capa GRU. Cuantas más unidades,
        más capacidad tiene el modelo para aprender patrones, pero también
        aumenta el riesgo de sobreajuste y el tiempo de entrenamiento.

    learning_rate : float, opcional (por defecto 1e-3)
        Tasa de aprendizaje del optimizador Adam.

    Devuelve
    --------
    model : keras.Model
        Modelo Keras ya compilado, listo para entrenar con model.fit(...).

    Notas
    -----
    - La arquitectura es intencionadamente simple:
        * Una sola capa GRU que "resume" la información de toda la ventana.
        * Una capa densa de salida con una única neurona (regresión).
    - Se usa 'mse' como función de pérdida y 'mae' como métrica principal,
      igual que en el modelo MLP, para que los resultados sean comparables.
    """
    model = Sequential(name="simple_gru")

    # Capa GRU principal.
    #
    # input_shape = (timesteps, n_features):
    #   - timesteps: longitud de la ventana (por ejemplo, 12 pasos).
    #   - n_features: cuántas características hay en cada paso
    #                 (por ejemplo, tráfico, hora del día, día de la semana).
    #
    # return_sequences = False:
    #   solo devolvemos el estado final de la GRU, que actúa como un resumen
    #   de toda la secuencia de entrada.
    model.add(
        GRU(
            units=gru_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=False,
            input_shape=(input_timesteps, n_features),
            name="gru",
        )
    )

    # Capa de salida:
    #   una única neurona lineal para predecir el valor futuro (regresión).
    model.add(
        Dense(
            units=1,
            activation="linear",
            name="output",
        )
    )

    # Compilar el modelo con una configuración estándar para regresión
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mse",      # error cuadrático medio
        metrics=["mae"], # error absoluto medio
    )

    return model
