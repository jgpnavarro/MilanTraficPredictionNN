# MilanTrafficPredictionNN

Predicción de **tráfico de Internet móvil** en Milán (Telecom Italia) sobre una **malla espacial** con intervalos de **10 minutos**.

El repositorio incluye:

- Pipeline de **preprocesado** (desde los ficheros crudos de Telecom Italia).
- **Unificación** de datos por celda (`by_cell`) y tabla global (`all_cells`).
- Primera fase de **modelado clásico** (**baselines**: Persistencia y Media Móvil).
- Primera fase de **Redes Neuronales** (MLP por celda, con normalización y features temporales).
- Métricas y reporting por celda, horizonte y split (train/val/test).
- Visualización de series reales vs predicción con sombreado de splits.
'''
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
│  ├─ __init__.py
│  ├─ config.py
│  ├─ process_cells_internet.py
│  ├─ timeseries_dataset.py
|  └─ run_processing.py
├─ Modeling/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data_access.py
│  ├─ targets.py
│  ├─ metrics.py
│  ├─ reporting.py
│  ├─ scaling.py
│  ├─ baselines/
│  │  ├─ __init__.py
│  │  ├─ persistence/
│  │  │  ├─ __init__.py
│  │  │  ├─ persistence.py
│  │  │  ├─ run_persistence.py
│  │  │  ├─ viz.py
│  │  │  ├─ run_plots.py
│  │  │  └─ output/
│  │  └─ moving_average/
│  │     ├─ __init__.py
│  │     ├─ moving_average.py
│  │     ├─ run_moving_average.py
│  │     ├─ run_plots.py
│  │     └─ output/
│  └─ neural_networks/
│     ├─ __init__.py
│     └─ mlp/
│        ├─ __init__.py
│        ├─ mlp.py
│        ├─ run_mlp.py
│        ├─ run_plots.py
│        └─ output/
└─ Readme.md


```
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

```

## Mapa (`Mapa Squares Milán/`)

* `SelectedSquares.kml`
  Polígonos e **IDs de celda** (grid) de las zonas analizadas: Bocconi, Navigli, Duomo, etc.
  Es un **documento de referencia geográfica**, no es entrada del pipeline de datos.

```

## Preprocesado (`Processing/`)

### `Processing/config.py`

* Rutas de datos:

  * `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`

* Selección de celdas:

  * `CELL_IDS`: lista de IDs de celda a conservar en el análisis.

* Zona horaria:

  * `TIMEZONE = "Europe/Rome"`
    usada para convertir los timestamps de epoch ms a `datetime` local.

### `Processing/process_cells_internet.py`

Lee los ficheros crudos de `raw/`, convierte tiempos y agrega tráfico de internet:

* Lee cada `sms-call-internet-mi-YYYY-MM-DD.txt`.
* Convierte el tiempo (epoch ms) a `datetime` (con `TIMEZONE`).
* Filtra sólo las celdas de `CELL_IDS`.
* Agrega `internet_total` por `(cell_id, datetime)` en frecuencia de **10 minutos**.

**Salida**: un fichero `*_internet_total.csv` por día en `Data/processed/`.

### `Processing/timeseries_dataset.py`

Ejecuta lo de arriba

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

```

## Modelado general (`Modeling/`)

Utilidades comunes a baselines y redes neuronales.
Todo el modelado se hace **en memoria** a partir de `Data/processed/by_cell/`.

### `Modeling/config.py`

Parámetros y rutas de modelado:

* Horizontes de predicción (en pasos de 10 minutos):

  ```python
  H_LIST = [1, 6]       # 10 minutos y 60 minutos
  ```

* Split temporal (proporción):

  ```python
  SPLIT = (0.70, 0.15, 0.15)   # train / val / test
  ```

* Frecuencia informativa:

  ```python
  FREQ = "10T"   # 10 minutes
  ```

* Rutas a datos procesados:

  ```python
  PROCESSED_DIR = Data/processed/
  BY_CELL_DIR   = Data/processed/by_cell/
  ```

* Parámetros específicos de salidas para baselines y redes (directorios de resultados y modelos).

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

donde `SPLIT=(0.70,0.15,0.15)` se respeta por índice temporal (no mezcla el orden).

### `Modeling/targets.py`

Construcción de objetivos (pares `(X, y)`) para un horizonte `H`.

Funciones principales:

* `make_xy_for_horizon(series, H)`

  Construye los datos para un horizonte fijo:

  * `X(t) = y(t)` (en la versión simple, X es un array/serie con el valor actual).
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

  y devuelve una lista de filas (dicts) con columnas:

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
  Permite filtrar por `cells` (lista de `cell_id`) y columnas.

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
  Si el máximo no es válido o es menor o igual que 0, devuelve 1.0.

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

```

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
    Para un horizonte `H`, construye `(X,y)` por split usando `targets`, aplica la regla de persistencia y devuelve un diccionario de métricas por split.

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

```

### Media Móvil (`Modeling/baselines/moving_average/`)

#### `moving_average.py`

Baseline de media móvil (rolling mean):

* Idea:

  ```text
  ŷ(t+H) = media de [y(t-L+1), ..., y(t)]
  ```

  donde `L` es el tamaño de la ventana (número de pasos de 10 min).

* Se implementa de forma eficiente usando ventanas deslizantes sobre la serie.

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

```

## Redes Neuronales (`Modeling/neural_networks/`)

Primera red neuronal implementada: un **MLP por celda**, para horizonte fijo `H=6` (1 hora) usando ventanas de `NN_INPUT_WINDOW=12` (2 horas) y features temporales sencillas.

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
  Núm. de neuronas por capa oculta (por defecto dos capas: 64 y 32 neuronas).

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

### Entrenamiento y evaluación (`Modeling/neural_networks/mlp/run_mlp.py`)

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

   Estas columnas se añaden en la función `add_time_features(X)`.

6. Convierte `X_*` y `y_*` a `numpy.float32`.

7. Construye el modelo:

   ```python
   input_dim = X_train.shape[1]
   model = build_mlp(input_dim=input_dim)
   ```

8. Entrena con:

   * `epochs = MLP_EPOCHS_MAX`
   * `batch_size = MLP_BATCH_SIZE`
   * `EarlyStopping` (monitorizando `val_loss` y restaurando los mejores pesos).

9. Guarda el modelo entrenado en:

   ```text
   Modeling/neural_networks/mlp/output/models/mlp_cell_<id>.keras
   ```

   (ruta concreta en `MLP_MODELS_DIR` dentro de `config.py`).

10. Predice en `train/val/test` (en escala normalizada) y deshace la normalización multiplicando por `max_train` (factor de escala por celda).

11. Calcula métricas por split (`MAE`, `RMSE`, `MAPE`, `wMAPE`, `sMAPE`, `Y_MEAN`, `n`) con `metrics.py`.

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

> Se entrena un modelo MLP **independiente** por celda (no hay modelo común a todas).

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

```

## Requisitos

* Python **3.10+**

* Librerías de datos y gráficos:

  ```bash
  pip install pandas numpy matplotlib
  ```

* TensorFlow + Keras (para las Redes Neuronales):

  ```bash
  pip install tensorflow keras
  ```

> Recomendado usar un entorno virtual:
>
> ```bash
> python -m venv .venv
> .venv\Scripts\activate   # Windows
> pip install -r requirements.txt  # si se define
> ```

```

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

```

## Notas y posibles extensiones

* Los módulos **generales** (`data_access`, `targets`, `metrics`, `reporting`, `scaling`) están diseñados para ser reutilizables tanto por baselines como por redes neuronales.
* La normalización y el tamaño de ventana se pueden ajustar fácilmente vía `Modeling/config.py`.
* Extensiones posibles:

  * añadir más features temporales (festivos, vacaciones, etc.),
  * probar arquitecturas específicas de series (LSTM/GRU, CNN),
  * hacer búsqueda de hiperparámetros para el MLP,
  * comparar explícitamente MLP vs modelo lineal con las mismas features.

