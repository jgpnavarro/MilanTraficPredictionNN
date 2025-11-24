# MilanTrafficPredictionNN

Predicción de **tráfico de Internet móvil** en Milán (Telecom Italia) sobre una **malla espacial** con intervalos de **10 minutos**.

El repositorio incluye:

- Pipeline de **preprocesado** (desde los ficheros crudos de Telecom Italia).
- **Unificación** de datos por celda (`by_cell`) y tabla global (`all_cells`).
- Primera fase de **modelado clásico** (**baselines**: Persistencia y Media Móvil).
- Primera fase de **Redes Neuronales** (MLP por celda, con normalización y features temporales).
- Segunda red neuronal secuencial (GRU por celda, con features de calendario: día de la semana, hora, festivos, etc.).
- Métricas y reporting por celda, horizonte y split (train/val/test).
- Visualización de series reales vs predicción con sombreado de splits.
- Script de orquestación para ejecutar varios modelos a la vez y comparar sus predicciones y métricas (por celda).
- Módulo de **energía** para simular una política ACTIVO/AHORRO por celda usando las predicciones de los modelos.
- Pipeline de energía para calcular **ahorro energético teórico** y **riesgo** y generar gráficos de consumo acumulado.

---

## Estructura del proyecto

```text
MilanTrafficPredictionNN/
├─ Data/
│  ├─ raw/
│  └─ processed/
│     ├─ all_cells.csv
│     └─ by_cell/
│        ├─ cell_4259.csv
│        ├─ cell_4456.csv
│        └─ ...
├─ Mapa Squares Milán/
│  └─ SelectedSquares.kml
├─ Processing/
│  ├─ config.py
│  ├─ process_cells_internet.py
│  ├─ timeseries_dataset.py
│  └─ run_processing.py
├─ Modeling/
│  ├─ config.py
│  ├─ data_access.py
│  ├─ targets.py
│  ├─ metrics.py
│  ├─ reporting.py
│  ├─ scaling.py
│  ├─ features_calendar.py
│  ├─ baselines/
│  │  ├─ persistence/
│  │  │  ├─ persistence.py
│  │  │  ├─ run_persistence.py
│  │  │  ├─ run_plots.py
│  │  │  └─ output/
│  │  └─ moving_average/
│  │     ├─ moving_average.py
│  │     ├─ run_moving_average.py
│  │     ├─ run_plots.py
│  │     └─ output/
│  ├─ neural_networks/
│  │  ├─ mlp/
│  │  │  ├─ mlp.py
│  │  │  ├─ run_mlp.py
│  │  │  ├─ run_plots.py
│  │  │  └─ output/
│  │  └─ gru/
│  │     ├─ gru.py
│  │     ├─ run_gru.py
│  │     ├─ run_plots.py
│  │     └─ output/
│  └─ energy/
│     ├─ run_energy.py
|     ├─ run_plots.py 
│     └─ output/
├─ total_output/
│  ├─ metrics_by_cell_H6_W12_test.csv
│  └─ plots_cells/
│     └─ cell_<id>_combined.png
├─ run_orchestrator.py
├─ run_energy_pipeline.py
└─ Readme.md
````

---

## Datos (`Data/`)

### `Data/raw/`

Ficheros originales de Telecom Italia, uno por día:

* Formato tipo: `sms-call-internet-mi-YYYY-MM-DD.txt`
* Cobertura temporal: **2013-11-01 → 2014-01-01**
* Contienen tráfico por celda y tipo (SMS, llamadas, internet, etc.)

Estos ficheros son sólo la **fuente original**; todo el modelado se hace sobre los datos procesados.

### `Data/processed/`

Salidas del pipeline de preprocesado:

* `*_internet_total.csv`
  Un CSV **por día**, con la agregación `internet_total` cada 10 minutos para las celdas seleccionadas.

* `all_cells.csv`
  Tabla “larga” con todas las celdas y tiempos:

  ```text
  cell_id,datetime,internet_total
  4259,2013-11-01 00:00:00,261.68
  4259,2013-11-01 00:10:00,189.42
  ...
  ```

* `by_cell/`
  Un CSV **por celda** con su serie ya limpia y ordenada:

  ```text
  Data/processed/by_cell/cell_4259.csv
  Data/processed/by_cell/cell_4456.csv
  ...
  ```

  Cada fichero contiene una única serie de `internet_total` indexada por `datetime`.

> Todo el código de modelado (baselines y redes) trabaja **siempre** desde `Data/processed/`.
> `raw/` se usa sólo en el preprocesado.

---

## Mapa (`Mapa Squares Milán/`)

* `SelectedSquares.kml`
  Polígonos e **IDs de celda** (grid) de las zonas analizadas: Bocconi, Navigli, Duomo, etc.
  Es un **documento de referencia geográfica**, no es entrada del pipeline de datos.

---

## Preprocesado (`Processing/`)

### `Processing/config.py`

* Rutas de datos:

  * `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`

* Selección de celdas:

  * `CELL_IDS`: lista de IDs de celda a conservar en el análisis.

* Zona horaria:

  * `TIMEZONE = "Europe/Rome"`
    Usada para convertir los timestamps de epoch ms a `datetime` local.

### `Processing/process_cells_internet.py`

Lee los ficheros crudos de `raw/`, convierte tiempos y agrega tráfico de internet:

* Lee cada `sms-call-internet-mi-YYYY-MM-DD.txt`.
* Convierte el tiempo (epoch ms) a `datetime` (con `TIMEZONE`).
* Filtra sólo las celdas de `CELL_IDS`.
* Agrega `internet_total` por `(cell_id, datetime)` en frecuencia de **10 minutos**.

**Salida**: un fichero `*_internet_total.csv` por día en `Data/processed/`.

### `Processing/timeseries_dataset.py`

Unifica los diarios en estructuras listas para el modelado:

* Concatena todos los `*_internet_total.csv`.
* Normaliza nombres de columnas.
* Ordena por `cell_id, datetime`.

**Salidas:**

* `Data/processed/all_cells.csv`
* `Data/processed/by_cell/cell_<id>.csv`

### `Processing/__init__.py`

Marca la carpeta como paquete Python (no contiene lógica de negocio).

---

## Modelado general (`Modeling/`)

Utilidades comunes a baselines y redes neuronales.
Todo el modelado se hace **en memoria** a partir de `Data/processed/by_cell/`.

### `Modeling/config.py`

Parámetros y rutas de modelado:

* Horizontes de predicción (en pasos de 10 minutos):

  ```python
  H_LIST = [1, 6]  # 10 minutos y 60 minutos
  ```

* Split temporal (proporción):

  ```python
  SPLIT = (0.70, 0.15, 0.15)  # train / val / test
  ```

* Frecuencia informativa:

  ```python
  FREQ = "10T"  # 10 minutes
  ```

* Rutas a datos procesados (conceptualmente):

  ```python
  PROCESSED_DIR = "Data/processed/"
  BY_CELL_DIR   = "Data/processed/by_cell/"
  ```

* Parámetros específicos de salidas para baselines y redes (directorios de resultados y modelos):

  * `PERSISTENCE_OUTPUT_DIR`, `MOVING_AVG_OUTPUT_DIR`
  * `MLP_OUTPUT_DIR`, `MLP_MODELS_DIR`
  * `GRU_OUTPUT_DIR`, `GRU_MODELS_DIR`
  * Parámetros de redes: `NN_INPUT_WINDOW`, `NN_HORIZON`, `NN_EPOCHS_MAX`, `NN_BATCH_SIZE`, `NN_EARLY_STOPPING_PATIENCE`, etc.

### `Modeling/data_access.py`

Acceso a las series por celda.

Funciones clave:

* `iter_cells_by_file()`
  Itera sobre todos los ficheros de `by_cell/` y devuelve tuplas:

  ```python
  for cell_id, series in iter_cells_by_file():
      # cell_id: int
      # series: pd.Series con DatetimeIndex y valores de internet_total
  ```

* `split_series(series)`
  Divide la serie completa de una celda en tres splits temporales:

  ```python
  s_train, s_val, s_test = split_series(series)
  ```

  donde `SPLIT = (0.70, 0.15, 0.15)` se respeta por índice temporal (no mezcla el orden).

### `Modeling/targets.py`

Construcción de objetivos (pares `(X, y)`) para un horizonte `H`.

Funciones principales:

* `make_xy_for_horizon(series, H)`
  Construye los datos para un horizonte fijo:

  * `X(t) = y(t)` (en la versión simple, X es el valor actual).
  * `y(t) = y(t+H)` → usando `series.shift(-H)`.
  * Se recortan las últimas `H` muestras para alinear tamaños.

* `make_xy_for_horizon_splits(s_train, s_val, s_test, H)`
  Aplica la misma lógica a cada split y devuelve:

  ```python
  {
      "train": (X_train, y_train),
      "val":   (X_val, y_val),
      "test":  (X_test, y_test),
  }
  ```

* `make_windowed_xy_for_horizon_splits(s_train, s_val, s_test, H, input_window)`
  Versión para redes neuronales: crea **ventanas deslizantes** de tamaño `input_window`:

  * Para cada instante futuro `t+H`, la entrada es:

    ```text
    [y(t - input_window + 1), ..., y(t - 1), y(t)]
    ```

  * La salida es `y(t+H)`.

  Devuelve un diccionario de splits donde:

  * `X_*` es un `DataFrame` con columnas `lag_k` (lags).
  * `y_*` es una `Series` con los valores futuros.

### `Modeling/metrics.py`

Métricas de evaluación entre `y_true` y `y_pred`:

* **MAE** (`mae`) → error absoluto medio.
* **RMSE** (`rmse`) → raíz del error cuadrático medio.
* **MAPE** (`mape`) → error porcentual medio (sensible a valores muy pequeños).
* **wMAPE** (`wmape`) → versión ponderada, más estable con valores pequeños.
* **sMAPE** (`smape`) → porcentaje simétrico, menos sesgo en extremos.

Todas las funciones devuelven un `float`.

### `Modeling/reporting.py`

Agrega resultados y facilita la impresión y guardado.

Funciones:

* `rows_from_eval_dict(cell_id, horizon, eval_dict)`
  Recibe un diccionario de métricas por split:

  ```python
  eval_dict = {
      "train": {"n": ..., "MAE": ..., ..., "Y_MEAN": ...},
      "val":   {...},
      "test":  {...},
  }
  ```

  y devuelve una lista de filas (diccionarios) con columnas:

  ```text
  cell_id, horizon, split, n,
  MAE, RMSE, MAPE, wMAPE, sMAPE, Y_MEAN
  ```

* `aggregate_results(all_rows)`
  Construye un `DataFrame` a partir de todas las filas acumuladas, ordenado por `horizon, cell_id, split`.

* `print_summary(df)`
  Imprime un resumen por `horizon/split` con:

  * media de MAE, RMSE, MAPE, wMAPE, sMAPE,
  * suma de `n` (`n_total`),
  * media de `Y_MEAN`.

* `print_per_cell(df, cells=None, columns=None)`
  Imprime un resumen por celda, separando horizontes y splits.
  Permite filtrar por `cells` (lista de `cell_id`) y por columnas.

* `save_results(df, filename, output_dir)`
  Guarda un CSV global en `output_dir/filename`.

* `save_per_cell_csv(df, filename_prefix, output_dir)`
  Guarda un CSV por celda:

  ```text
  <output_dir>/<filename_prefix><cell_id>.csv
  ```

### `Modeling/scaling.py`

Funciones de **normalización por máximo** por celda:

* `compute_train_max(s_train)`
  Calcula el máximo de la serie de entrenamiento (`s_train`).
  Si el máximo no es válido o es ≤ 0, devuelve 1.0 (para evitar divisiones raras).

* `scale_series_by_max(series, max_value)`
  Devuelve `series / max_value` como `pd.Series` de floats.

* `scale_splits_by_train_max(s_train, s_val, s_test)`

  Aplica la normalización:

  ```text
  valor_escalado = valor_real / max_train
  ```

  usando sólo `s_train` para obtener `max_train`, y devuelve:

  ```python
  s_train_scaled, s_val_scaled, s_test_scaled, max_train
  ```

  Para deshacer la normalización:

  ```text
  valor_real = valor_escalado * max_train
  ```

Esta normalización se utiliza especialmente en las redes neuronales.

### `Modeling/features_calendar.py`

Funciones para añadir **features de calendario** a un `DataFrame` que contenga:

* `datetime`
* `internet_total`

Funciones:

* `add_public_holiday_feature(df)`
  Añade `is_public_holiday` (0/1) usando la librería `holidays` y los festivos oficiales de Italia.

* `add_special_break_feature(df, periods_path, column_name="is_special_break")`
  Marca rangos de fechas especiales definidos en un CSV (por ejemplo, periodo de Navidad) añadiendo una columna binaria.

* `add_calendar_features(df, special_periods_path=None)`
  Añade:

  * `day_of_week` (0=lunes … 6=domingo),
  * `hour_of_day` (0–23),
  * `is_public_holiday` (0/1),
  * opcionalmente `is_special_break` (0/1) si se pasa un CSV con periodos especiales (por ejemplo `Data/calendar/special_periods.csv`).

---

## Baselines (`Modeling/baselines/`)

Contiene modelos clásicos sin entrenamiento (o muy sencillo) para servir como referencia.

### Persistencia (`Modeling/baselines/persistence/`)

#### `persistence.py`

Modelo de persistencia:

* Regla:

  ```text
  ŷ(t+H) = y(t)
  ```

* Funciones:

  * `predict_persistence(X)`
    Devuelve `X` como predicción (identidad).

  * `evaluate_persistence_split(X, y)`
    Calcula métricas para un split:

    * `n` (número de puntos),
    * `MAE`, `RMSE`, `MAPE`, `wMAPE`, `sMAPE`,
    * `Y_MEAN` (media de `y_true` en ese split).

  * `evaluate_persistence_splits(s_train, s_val, s_test, H)`
    Para un horizonte `H`, construye `(X, y)` por split usando `targets`, aplica la regla de persistencia y devuelve un diccionario de métricas por split.

#### `run_persistence.py`

Orquestador de la evaluación de persistencia:

1. Recorre las celdas con `iter_cells_by_file()`.

2. Divide cada serie en `s_train, s_val, s_test` (`split_series`).

3. Para cada `H` en `H_LIST`:

   * Construye `(X, y)` con `make_xy_for_horizon_splits`.
   * Evalúa persistencia con `evaluate_persistence_splits`.
   * Convierte resultados a filas (`rows_from_eval_dict`) y las acumula.

4. Agrega todas las filas en un `DataFrame` (`aggregate_results`).

5. Imprime:

   * resumen global (`print_summary`),
   * detalle por celda (`print_per_cell`).

6. Guarda resultados en `Modeling/baselines/persistence/output/`:

   * `persistence_results.csv` (global),
   * `persistence_cell_<id>.csv` (uno por celda).

**Ejecución** (desde la raíz del repo):

```bash
python -m Modeling.baselines.persistence.run_persistence
```

#### `viz.py` y `run_plots.py`

Visualización de persistencia:

* `viz.py`
  Funciones de ayuda para:

  * pintar la serie real (línea sólida),
  * pintar la predicción de persistencia (línea discontinua),
  * controlar tamaño de figura y estilo básico.

* `run_plots.py`

  * Reproduce el pipeline de datos (split, targets, predicciones).

  * Genera una **figura por celda y horizonte** que contiene:

    * serie real completa,
    * predicción de persistencia,
    * sombreado de zonas train / val / test.

  * Guarda las figuras en:

    ```text
    Modeling/baselines/persistence/output/plots_all/<cell_id>/H{H}_ALL.png
    ```

**Ejecución**:

```bash
python -m Modeling.baselines.persistence.run_plots
```

---

### Media Móvil (`Modeling/baselines/moving_average/`)

#### `moving_average.py`

Baseline de media móvil (rolling mean):

* Idea:

  ```text
  ŷ(t+H) = media de [y(t-L+1), ..., y(t)]
  ```

  donde `L` es el tamaño de la ventana (número de pasos de 10 min).

* Se implementa usando ventanas deslizantes sobre la serie.

* Funciones principales:

  * `moving_average_forecast(series, window_size, H)`
    Calcula la predicción de media móvil para un horizonte `H` y una ventana `window_size` sobre una serie (un split).

  * `evaluate_moving_average_splits(s_train, s_val, s_test, H, window_size)`
    Aplica la media móvil por split y devuelve métricas:

    * `n`, `MAE`, `RMSE`, `MAPE`, `wMAPE`, `sMAPE`, `Y_MEAN`.

#### `run_moving_average.py`

Orquestador del baseline de media móvil:

1. Recorre las celdas con `iter_cells_by_file()`.

2. Divide en `s_train, s_val, s_test`.

3. Para cada horizonte `H` (normalmente `H=1` y `H=6`) y uno o varios tamaños de ventana (ej. 6 y 12 pasos):

   * Aplica `evaluate_moving_average_splits`.
   * Convierte los resultados en filas con `rows_from_eval_dict`.
   * Añade columnas adicionales como:

     * `model = "moving_average"`,
     * `window = window_size`.

4. Agrega todas las filas en un `DataFrame`.

5. Imprime resumen global y detalle por celda.

6. Guarda en `Modeling/baselines/moving_average/output/`:

   * `moving_avg_results.csv` (todos los resultados),
   * `moving_avg_cell_<id>.csv` (uno por celda).

**Ejecución**:

```bash
python -m Modeling.baselines.moving_average.run_moving_average
```

#### `run_plots.py` (media móvil)

Visualización similar a persistencia, pero para la media móvil:

* Reproduce el pipeline de datos.

* Aplica la media móvil con la ventana/horizonte configurados.

* Genera **una figura por celda** que contiene:

  * serie real completa,
  * predicción de media móvil,
  * zonas sombreadas para train / val / test.

* Guarda las figuras en:

  ```text
  Modeling/baselines/moving_average/output/plots_all/cell_<id>_moving_avg.png
  ```

**Ejecución**:

```bash
python -m Modeling.baselines.moving_average.run_plots
```

---

## Redes Neuronales (`Modeling/neural_networks/`)

Se implementan dos modelos neuronales por celda:

* Un **MLP** (red densa) con lags y features temporales.
* Una **GRU** (red recurrente) con features de calendario.

### Arquitectura MLP (`Modeling/neural_networks/mlp/mlp.py`)

Define la función `build_mlp`:

```python
def build_mlp(
    input_dim: int,
    hidden_units=(64, 32),
    learning_rate=1e-3,
    dropout_rate=0.2,
)
```

* `input_dim`
  Número total de características de entrada (lags + features temporales).

* `hidden_units`
  Número de neuronas por capa oculta (por defecto dos capas: 64 y 32 neuronas).

* `dropout_rate`
  Proporción de neuronas que se apagan aleatoriamente en entrenamiento para reducir sobreajuste (por defecto 0.2).

Arquitectura:

* Capa oculta 1: `Dense(64, activation="relu")` + `Dropout(0.2)`
* Capa oculta 2: `Dense(32, activation="relu")` + `Dropout(0.2)`
* Capa de salida: `Dense(1, activation="linear")` (regresión)

Compilación:

* `loss="mse"` (error cuadrático medio).
* `optimizer=Adam(learning_rate)`.
* `metrics=["mae"]`.

### Entrenamiento y evaluación MLP (`Modeling/neural_networks/mlp/run_mlp.py`)

Orquestador para entrenar y evaluar un MLP **por celda**:

1. Itera celdas con `iter_cells_by_file()`.

2. Divide la serie de cada celda en `s_train, s_val, s_test` (`split_series`).

3. Normaliza cada split dividiendo por el máximo de `s_train` (`scale_splits_by_train_max`).

4. Construye ventanas con:

   ```python
   make_windowed_xy_for_horizon_splits(
       s_train_s, s_val_s, s_test_s,
       H=NN_HORIZON,           # 6 pasos (1 hora)
       input_window=NN_INPUT_WINDOW,  # 12 pasos (2 horas)
   )
   ```

   Esto genera `X_*` con lags (`lag_12` ... `lag_1`) y `y_*` con el valor futuro escalado.

5. Añade **features temporales** basadas en el índice de `X_*` (que representa el instante futuro `t+H`):

   * `hour_sin`, `hour_cos` → codificación cíclica de la hora del día.
   * `dow_sin`, `dow_cos` → codificación cíclica del día de la semana.
   * `is_weekend` → 1 si sábado/domingo, 0 en caso contrario.

6. Convierte `X_*` y `y_*` a `numpy.float32`.

7. Construye el modelo:

   ```python
   input_dim = X_train.shape[1]
   model = build_mlp(input_dim=input_dim)
   ```

8. Entrena con:

   * `epochs = NN_EPOCHS_MAX`
   * `batch_size = NN_BATCH_SIZE`
   * `EarlyStopping` (monitorizando `val_loss` y restaurando los mejores pesos).

9. Guarda el modelo entrenado en:

   ```text
   Modeling/neural_networks/mlp/output/models/mlp_cell_<id>.keras
   ```

10. Predice en `train/val/test` (en escala normalizada) y deshace la normalización multiplicando por `max_train` (factor de escala por celda).

11. Calcula métricas por split (`MAE`, `RMSE`, `MAPE`, `wMAPE`, `sMAPE`, `Y_MEAN`, `n`).

12. Convierte los resultados a filas con `rows_from_eval_dict`, añadiendo columnas extra:

    * `model = "mlp"`,
    * `window = NN_INPUT_WINDOW` (tamaño de la ventana de lags).

13. Agrega resultados en un DataFrame y:

    * imprime resumen global (`print_summary`),
    * imprime detalle por celda (`print_per_cell`),
    * guarda en `Modeling/neural_networks/mlp/output/`:

      * `mlp_results.csv`,
      * `mlp_cell_<id>.csv`.

**Ejecución**:

```bash
python -m Modeling.neural_networks.mlp.run_mlp
```

> Se entrena un modelo MLP **independiente** por celda.

### Visualización MLP (`Modeling/neural_networks/mlp/run_plots.py`)

Visualiza el comportamiento del MLP por celda:

1. Para cada celda, reproduce el pipeline de datos (split, normalización, ventanas, features temporales).

2. Carga el modelo ya entrenado `mlp_cell_<id>.keras`.

3. Predice en `train/val/test`, deshace la normalización y reconstruye una serie de predicción (`y_pred`) alineada con la serie real (`y_real`).

4. Calcula los rangos temporales de `train/val/test` para sombreados.

5. Genera una **figura por celda** con:

   * serie real (línea sólida),
   * predicción MLP (línea discontinua),
   * bandas sombreadas para train / val / test.

6. Guarda la figura en:

   ```text
   Modeling/neural_networks/mlp/output/plots_all/cell_<id>_mlp.png
   ```

**Ejecución**:

```bash
python -m Modeling.neural_networks.mlp.run_plots
```

---

### Arquitectura GRU (`Modeling/neural_networks/gru/gru.py`)

Define la función `build_gru`:

```python
def build_gru(
    input_timesteps: int,
    n_features: int = 1,
    gru_units: int = 32,
    learning_rate: float = 1e-3,
) -> Sequential:
    ...
```

* `input_timesteps`
  Longitud de la ventana temporal de entrada (por ejemplo, 12 pasos de 10 minutos).

* `n_features`
  Número de características por instante de tiempo. En este proyecto se usan, por ejemplo:

  * `traffic_scaled` (tráfico normalizado),
  * `hour_of_day` (0–23),
  * `day_of_week` (0–6),
  * `is_public_holiday` (0/1, festivo oficial en Italia).

* `gru_units`
  Número de neuronas de la capa GRU (por defecto 32). Más unidades = más capacidad, pero también más riesgo de sobreajuste.

Arquitectura:

* Una sola capa `GRU` que resume toda la ventana temporal.
* Una capa de salida densa `Dense(1, activation="linear")` para predecir el valor futuro (regresión).

Compilación:

* `loss="mse"` (error cuadrático medio).
* `optimizer=Adam(learning_rate)`.
* `metrics=["mae"]`.

### Entrenamiento y evaluación GRU (`Modeling/neural_networks/gru/run_gru.py`)

Orquestador para entrenar y evaluar una GRU **por celda**:

1. Itera celdas con `iter_cells_by_file()`.

2. Divide la serie de cada celda en `s_train, s_val, s_test` (`split_series`).

3. Normaliza cada split dividiendo por el máximo de `s_train` (`scale_splits_by_train_max`).

4. Construye un `DataFrame` con:

   * `datetime`,
   * `internet_total`,
   * features de calendario (`day_of_week`, `hour_of_day`, `is_public_holiday` y, opcionalmente, `is_special_break` leída desde un CSV).

5. Genera ventanas deslizantes para H=6 y ventana `NN_INPUT_WINDOW=12` (2 horas):

   * Entrada: secuencia de tamaño `(12, n_features)` con features como
     `[traffic_scaled, hour_of_day, day_of_week, is_public_holiday]`.
   * Salida: `traffic_scaled` en `t+H`.

6. Convierte las ventanas a `numpy.float32` y construye el modelo:

   ```python
   model = build_gru(
       input_timesteps=NN_INPUT_WINDOW,
       n_features=n_features,
   )
   ```

7. Entrena con:

   * `epochs = NN_EPOCHS_MAX`,
   * `batch_size = NN_BATCH_SIZE`,
   * `EarlyStopping` monitorizando `val_loss` y restaurando los mejores pesos.

8. Guarda el modelo entrenado en:

   ```text
   Modeling/neural_networks/gru/output/models/gru_cell_<id>.keras
   ```

9. Predice en `train/val/test` (en escala normalizada) y deshace la normalización multiplicando por el factor `max_train` de cada celda.

10. Calcula métricas por split (`MAE`, `RMSE`, `MAPE`, `wMAPE`, `sMAPE`, `Y_MEAN`, `n`) y añade columnas:

    * `model = "gru"`,
    * `window = NN_INPUT_WINDOW`,
    * `n_features` (número de features por timestep).

11. Agrega resultados y los guarda en:

    ```text
    Modeling/neural_networks/gru/output/
      ├─ gru_results.csv
      ├─ gru_cell_<id>.csv
      └─ models/
           ├─ gru_cell_4259.keras
           └─ ...
    ```

**Ejecución**:

```bash
python -m Modeling.neural_networks.gru.run_gru
```

### Visualización GRU (`Modeling/neural_networks/gru/run_plots.py`)

Visualiza el comportamiento de la GRU por celda, de forma similar al MLP:

1. Para cada celda, reproduce el pipeline:

   * split en `train/val/test`,
   * normalización por máximo,
   * añadido de features de calendario,
   * construcción de ventanas `(NN_INPUT_WINDOW, n_features)`.

2. Carga el modelo `gru_cell_<id>.keras`.

3. Predice en `train/val/test`, deshace la normalización y reconstruye una serie `y_pred` alineada con la serie real `y_real`.

4. Calcula rangos temporales de `train/val/test` para sombrear en la figura.

5. Genera una **figura por celda** con:

   * serie real (línea sólida),
   * predicción GRU (línea discontinua),
   * bandas sombreadas para train / val / test.

6. Guarda la figura en:

   ```text
   Modeling/neural_networks/gru/output/plots_all/cell_<id>_gru.png
   ```

**Ejecución**:

```bash
python -m Modeling.neural_networks.gru.run_plots
```

---

## Orquestador de modelos (`run_orchestrator.py` + `total_output/`)

El script `run_orchestrator.py` (en la raíz del proyecto) permite:

* Ejecutar uno o varios modelos:

  * `persistence`
  * `moving_average`
  * `mlp`
  * `gru`
  * `all` (todos)

* Generar una figura por celda con:

  * serie real,
  * predicción de cada modelo seleccionado,
  * fondo sombreado train / val / test.

* Crear una tabla con MAPE, wMAPE y sMAPE en test por celda y modelo
  (para H=6 y, en el caso de media móvil, ventana W=12).

Ejemplos de uso:

```bash
# Ejecutar todos los modelos y generar salidas combinadas
python run_orchestrator.py --models all

# Solo MLP y GRU
python run_orchestrator.py --models mlp,gru

# Solo baselines
python run_orchestrator.py --models persistence,moving_average
```

Salidas en la raíz:

```text
total_output/
  ├─ metrics_by_cell_H6_W12_test.csv   # MAPE, wMAPE, sMAPE por celda y modelo (split test)
  └─ plots_cells/
       ├─ cell_4259_combined.png       # Real + modelos seleccionados
       ├─ cell_4456_combined.png
       └─ ...
```

---

## Módulo de energía (`Modeling/energy/`)

Este módulo usa las **predicciones de los modelos** para simular una política sencilla de ahorro de energía por celda.

### `Modeling/energy/run_energy.py`

Este script calcula, para cada celda y cada modelo (`persistence`, `moving_average`, `mlp`, `gru`):

1. **Niveles de carga L/M/H**

   * Usa los datos reales de train+val para calcular dos percentiles de tráfico por celda:

     * `P20` → umbral de baja demanda,
     * `P80` → umbral de alta demanda.
   * En test, convierte el tráfico real y el tráfico **predicho a 1 hora vista** en niveles:

     * `L` (Low) si tráfico < `P20`,
     * `M` (Medium) si `P20 ≤ tráfico ≤ P80`,
     * `H` (High) si tráfico > `P80`.

2. **Política ACTIVO / AHORRO**

   * En cada instante del test, mira el nivel **predicho** (`L`, `M`, `H`).
   * Regla:

     * si la predicción es `M` o `H` → estado = `ACTIVE`,
     * si la predicción es `L` durante al menos `K` intervalos seguidos (por ejemplo 3 → 30 minutos) → estado = `SAVING` (modo ahorro),
     * si vuelve a `M` o `H`, la celda pasa otra vez a `ACTIVE`.

   El resultado es una serie `state_pred(t)` con valores `ACTIVE` o `SAVING` para cada intervalo de test.

3. **Modelo simple de consumo energético**

   * Se considera un consumo relativo por intervalo:

     * `ACTIVE` → 1.0 (100 %),
     * `SAVING` → 0.6 (60 %).
   * Se comparan dos escenarios:

     * **Base**: celda siempre `ACTIVE` → consumo total = `n_test * 1.0`.
     * **Con política**: se usa `state_pred(t)` y se suma 1.0 o 0.6 según corresponda.

   De ahí se obtiene el **ahorro energético relativo**:

   ```text
   Ahorro (%) = (E_base - E_policy) / E_base * 100
   ```

4. **Riesgo de QoS**

   * Se cuentan los intervalos en los que la celda está en `SAVING` (`N_ahorro`).
   * De esos, se miran los casos donde el nivel real no es `L` (es decir, es `M` o `H`) → `N_conflictos`.
   * Se define un **riesgo**:

     ```text
     Riesgo (%) = N_conflictos / N_ahorro * 100
     ```

     (si `N_ahorro` = 0, el riesgo no se define y se deja como NaN).

El script recorre todas las celdas y todos los modelos elegidos y guarda el resultado en:

```text
Modeling/energy/output/energy_results_all_models.csv
```

Cada fila contiene, entre otras, las columnas:

* `cell_id`, `model`, `H`, `window`,
* `P20`, `P80`, `K_min_low`, `P_active`, `P_saving`,
* `n_test`, `E_base`, `E_policy`,
* `saving_percent`, `time_saving_percent`,
* `N_ahorro`, `N_conflictos`, `risk_percent`.

Ejemplo de ejecución (todos los modelos):

```bash
python -m Modeling.energy.run_energy --models all
```

También se puede limitar la lista de modelos, por ejemplo:

```bash
python -m Modeling.energy.run_energy --models mlp,gru
```

### `Modeling/energy/run_plots.py`

Este script visualiza el **consumo acumulado** en el tramo de test para cada celda y para varios modelos a la vez.

Para cada celda:

1. Reproduce la lógica de `run_energy.py` para obtener, por modelo:

   * tráfico real y predicho en test,
   * niveles L/M/H,
   * estados `ACTIVE` / `SAVING`,
   * consumo por intervalo (1.0 o 0.6).

2. Calcula:

   * **Consumo acumulado base**: suma de 1.0 en cada intervalo (recta ascendente).
   * **Consumo acumulado con política** para cada modelo:

     * se suma 1.0 si está en `ACTIVE`,
     * se suma 0.6 si está en `SAVING`.

3. Genera una figura por celda con:

   * una línea base (`Base (siempre ACTIVO)`),
   * una línea de consumo acumulado por modelo (por ejemplo Persistencia, Media móvil, MLP, GRU).

Las figuras se guardan en:

```text
Modeling/energy/output/plots_cumulative/cell_<id>_energy.png
```

**Ejecución**:

```bash
python -m Modeling.energy.run_plots
```

---

## Pipeline de energía (`run_energy_pipeline.py`)

En la raíz del proyecto hay un script pequeño que ejecuta **sólo la parte energética**, sin volver a entrenar modelos:

* Llama a `Modeling.energy.run_energy` con todos los modelos.
* Llama a `Modeling.energy.run_plots` para generar las gráficas de consumo acumulado.

Se asume que los modelos (baselines, MLP, GRU) ya se han ejecutado antes y que existen los ficheros necesarios.

**Ejecución** desde la raíz:

```bash
python run_energy_pipeline.py
```

Esto actualiza:

* `Modeling/energy/output/energy_results_all_models.csv`
* `Modeling/energy/output/plots_cumulative/cell_<id>_energy.png`

---

## Requisitos

* Python **3.10+**

* Librerías de datos y gráficos:

  ```bash
  pip install pandas numpy matplotlib
  ```

* TensorFlow + Keras (para las Redes Neuronales) y festivos:

  ```bash
  pip install tensorflow keras holidays
  ```

> Recomendado usar un entorno virtual:
>
> ```bash
> python -m venv .venv
> .venv\Scripts\activate   # Windows
> pip install -r requirements.txt  # si se define
> ```

---

## Comandos típicos

> Ejecutar siempre desde la **raíz** del repositorio.

### 1. Preprocesar datos crudos → diarios `*_internet_total.csv`

```bash
python -m Processing.run_processing
```

### 2. Unificar en `all_cells` y `by_cell`

```bash
python -m Processing.timeseries_dataset
```

Salidas:

* `Data/processed/all_cells.csv`
* `Data/processed/by_cell/cell_<id>.csv`

### 3. Evaluar baseline de Persistencia

```bash
python -m Modeling.baselines.persistence.run_persistence
```

Salidas en:

```text
Modeling/baselines/persistence/output/
  ├─ persistence_results.csv
  ├─ persistence_cell_4259.csv
  ├─ persistence_cell_4456.csv
  └─ ...
```

Plots:

```bash
python -m Modeling.baselines.persistence.run_plots
```

### 4. Evaluar baseline de Media Móvil

```bash
python -m Modeling.baselines.moving_average.run_moving_average
```

Salidas en:

```text
Modeling/baselines/moving_average/output/
  ├─ moving_avg_results.csv
  ├─ moving_avg_cell_4259.csv
  └─ ...
```

Plots:

```bash
python -m Modeling.baselines.moving_average.run_plots
```

### 5. Entrenar y evaluar MLP por celda (H=6, ventana de lags=12)

```bash
python -m Modeling.neural_networks.mlp.run_mlp
```

Salidas en:

```text
Modeling/neural_networks/mlp/output/
  ├─ mlp_results.csv
  ├─ mlp_cell_4259.csv
  ├─ ...
  └─ models/
       ├─ mlp_cell_4259.keras
       └─ ...
```

Plots:

```bash
python -m Modeling.neural_networks.mlp.run_plots
```

Figuras en:

```text
Modeling/neural_networks/mlp/output/plots_all/
  ├─ cell_4259_mlp.png
  ├─ cell_4456_mlp.png
  └─ ...
```

### 6. Entrenar y evaluar GRU por celda (H=6, ventana=12)

```bash
python -m Modeling.neural_networks.gru.run_gru
```

Salidas en:

```text
Modeling/neural_networks/gru/output/
  ├─ gru_results.csv
  ├─ gru_cell_4259.csv
  ├─ ...
  └─ models/
       ├─ gru_cell_4259.keras
       └─ ...
```

Plots:

```bash
python -m Modeling.neural_networks.gru.run_plots
```

Figuras en:

```text
Modeling/neural_networks/gru/output/plots_all/
  ├─ cell_4259_gru.png
  ├─ cell_4456_gru.png
  └─ ...
```

### 7. Orquestador: ejecutar varios modelos y generar gráficas combinadas

```bash
# Todos los modelos
python run_orchestrator.py --models all

# Solo MLP y GRU
python run_orchestrator.py --models mlp,gru

# Solo baselines
python run_orchestrator.py --models persistence,moving_average
```

Salidas en:

```text
total_output/
  ├─ metrics_by_cell_H6_W12_test.csv
  └─ plots_cells/
       ├─ cell_4259_combined.png
       └─ ...
```

### 8. Pipeline de energía (ahorro y riesgo)

```bash
python run_energy_pipeline.py
```

Esto ejecuta:

* `Modeling.energy.run_energy` (cálculo de ahorro y riesgo con todos los modelos).
* `Modeling.energy.run_plots` (gráficas de consumo acumulado).

---

## Notas y posibles extensiones actuales

* Los módulos **generales** (`data_access`, `targets`, `metrics`, `reporting`, `scaling`, `features_calendar`) están diseñados para ser reutilizables tanto por baselines como por redes neuronales.
* La normalización, el tamaño de ventana y el horizonte se ajustan fácilmente vía `Modeling/config.py`.
* La parte energética permite pasar de “MAPE y MAE” a indicadores más interpretables para una operadora: **ahorro potencial** y **riesgo de degradar QoS**.

Extensiones posibles:

* Añadir más features temporales (otros festivos, eventos deportivos, etc.).
* Probar arquitecturas específicas de series (más capas GRU/LSTM, CNN).
* Hacer búsqueda de hiperparámetros para MLP/GRU por celda.
* Comparar explícitamente el comportamiento de los modelos por subperiodos (antes/durante vacaciones, días laborables vs fines de semana).
* Usar otras políticas de energía (más de dos niveles de consumo, histéresis distinta, etc.).

## Próximos pasos

* Analizar en detalle las diferencias entre modelos (baselines, MLP, GRU) en términos de **error de predicción** en test.
* Interpretar los resultados de energía por celda: cuánto se podría ahorrar y con qué riesgo, especialmente en zonas como Bocconi y Navigli frente a zonas más tranquilas.
* Discutir las limitaciones del modelo de energía (muy simplificado) y posibles mejoras para acercarse más a la realidad de una red móvil.

