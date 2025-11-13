---

# MilanTrafficPredictionNN

Predicción de **tráfico de Internet móvil** en Milán (Telecom Italia) sobre una **malla espacial** con intervalos de **10 minutos**.
El repositorio incluye el pipeline de **preprocesado**, la **unificación** por celda y una primera fase de **modelado** (baseline de **persistencia**), con métricas y reporting por celda.

---

## Estructura del proyecto

```
MilanTraficPredictionNN/
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
│  └─ timeseries_dataset.py
├─ Modeling/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data_access.py
│  ├─ targets.py
│  ├─ metrics.py
│  ├─ reporting.py
│  └─ persistence/
│     ├─ __init__.py
│     ├─ persistence.py         ← implementación del modelo (persistencia)
│     ├─ run_persistence.py     ← orquestador del experimento
│     └─ output/                ← **único** directorio de salida de persistencia
├─ run_pipeline.py
└─ Readme.md
```

---

## Datos (`Data/`)

* **`raw/`**: ficheros originales por día (`sms-call-internet-mi-YYYY-MM-DD.txt`). Cobertura: **2013-11-01 → 2014-01-01**.
* **`processed/`** (salida del preprocesado):

  * `*_internet_total.csv` → **un CSV por día** con `internet_total` cada 10 minutos.
  * `all_cells.csv` → **tabla larga** con todas las celdas: `cell_id, datetime, internet_total`.
  * `by_cell/` → **un CSV por celda** (`cell_<id>.csv`) con su serie completa ordenada por tiempo.

> El modelado trabaja **siempre** desde `Data/processed/`. Los `raw/` solo son origen.

---

## Mapa (`Mapa Squares Milán/`)

* **`SelectedSquares.kml`**: polígonos e **IDs de celda** (grid) de las zonas analizadas (Bocconi, Navigli, Duomo, etc.). Documento de referencia geográfica (no es entrada del pipeline).

---

## Preprocesado (`Processing/`)

* **`config.py`**

  * Rutas (`DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`).
  * `CELL_IDS` (lista de celdas a conservar).
  * `TIMEZONE="Europe/Rome"` (conversión correcta desde epoch ms).

* **`process_cells_internet.py`**
  Lee `raw/`, convierte tiempos a `datetime` local, filtra por `CELL_IDS` y **agrega `internet_total`** por `(cell_id, datetime)` en pasos de 10’.
  **Salida:** `Data/processed/*_internet_total.csv`.

* **`timeseries_dataset.py`**
  Concatena todos los diarios, normaliza y **ordena**.
  **Salidas:**

  * `Data/processed/all_cells.csv`
  * `Data/processed/by_cell/cell_<id>.csv`

* **`__init__.py`**
  Marca la carpeta como paquete Python.

---

## Modelado (`Modeling/`)

Utilidades comunes y modelos. Todo se realiza **en memoria** a partir de los CSV de `Data/processed/`.

### Configuración (`Modeling/config.py`)

* `H_LIST = [1, 6]` → horizontes (10’ y 60’).
* `SPLIT = (0.70, 0.15, 0.15)` → cortes temporales train/val/test.
* `FREQ = "10T"` → frecuencia esperada (solo informativa aquí).
* `PROCESSED_DIR` → ruta a `Data/processed/`.
* (No se guardan salidas aquí; persistencia guarda en su subcarpeta).

### Acceso a datos (`Modeling/data_access.py`)

* **Objetivo:** leer series por celda ya procesadas y dividir por tiempo.
* **Funciones clave:**

  * `iter_cells_by_file()` → genera `(cell_id: int, serie: pd.Series)` leyendo `Data/processed/by_cell/cell_*.csv`.
    La serie es `internet_total` indexada por `datetime`.
  * `split_series(serie)` → devuelve `s_train, s_val, s_test` (70/15/15) **por posición temporal**.

### Construcción de objetivos (`Modeling/targets.py`)

* **Objetivo:** preparar pares supervisados `(X, y)` para un horizonte `H`.
* **Funciones:**

  * `make_xy_for_horizon(series, H)` →
    `X(t)=series(t)`, `y(t)=series(t+H)` (via `shift(-H)`), recortando los últimos `H` puntos.
  * `make_xy_for_horizon_splits(s_train, s_val, s_test, H)` → aplica lo anterior a cada split y retorna un diccionario:
    `{"train": (X_tr,y_tr), "val": (X_va,y_va), "test": (X_te,y_te)}`.

### Métricas (`Modeling/metrics.py`)

* **Objetivo:** calcular errores punto a punto entre `y_true` y `y_pred`.
* **Disponibles:**

  * **MAE** (error absoluto medio) y **RMSE** (raíz del error cuadrático medio) → **absolutos**, misma unidad que `internet_total`.
  * **MAPE** (porcentaje punto a punto; sensible a valores pequeños).
  * **wMAPE** (porcentaje ponderado, más estable con reales pequeños).
  * **sMAPE** (porcentaje simétrico, menos sesgo en extremos).

### Reporting (`Modeling/reporting.py`)

* **Objetivo:** agregar resultados por **celda/horizonte/split** y exportar.
* **Funciones clave:**

  * `rows_from_eval_dict(cell_id, horizon, eval_dict)` → transforma el dict de métricas por split en **filas homogéneas**.
    Columnas estándar:
    `cell_id, horizon, split, n, MAE, RMSE, MAPE, wMAPE, sMAPE, Y_MEAN`
  * `aggregate_results(all_rows)` → DataFrame con todas las filas.
  * `print_summary(df)` → resumen por `horizon/split` (medias) + `n_total`.
  * `print_per_cell(df, cells=None, columns=None)` → detalle por celda (opcional filtro de celdas/columnas).

> **Importante**: las funciones de guardado **no** se usan para persistencia, porque persistencia guarda **solo** en su subcarpeta `Modeling/persistence/output/` (ver abajo).

---

## Baseline: Persistencia (`Modeling/persistence/`)

### Implementación (`Modeling/persistence/persistence.py`)

* **Regla del modelo:** `ŷ(t+H) = y(t)` (sin entrenamiento, sin hiperparámetros).
* **Funciones:**

  * `predict_persistence(X)` → devuelve `X` como predicción.
  * `evaluate_persistence_split(X, y)` → calcula métricas en un split:
    `n, MAE, RMSE, MAPE, wMAPE, sMAPE, Y_MEAN` (donde `Y_MEAN` es la **media de `y_true` evaluado**).
  * `evaluate_persistence_splits(s_train, s_val, s_test, H)` → aplica a los tres splits para un `H`.

### Orquestador (`Modeling/persistence/run_persistence.py`)

* **Flujo:**

  1. Itera celdas con `data_access.iter_cells_by_file()`.
  2. Divide cada serie con `data_access.split_series()`.
  3. Para cada `H` en `H_LIST`:

     * Construye `(X, y)` con `targets`.
     * Predice con persistencia (`ŷ = X`).
     * Calcula métricas por split.
     * Convierte a filas (reporting) y acumula.
  4. Agrega a un DataFrame y:

     * imprime **resumen global** por `horizon/split`,
     * imprime **detalle por celda**.
  5. **Guarda los CSV en** `Modeling/persistence/output/`

     * `persistence_results.csv` → todas las celdas, horizontes y splits.
     * `persistence_cell_<cell_id>.csv` → un CSV por celda.
* **Ejecución:**

  ```bash
  python -m Modeling.persistence.run_persistence
  ```

  > Ejecutar **siempre** desde la **raíz** del repo.

### Salidas (`Modeling/persistence/output/`)

* **`persistence_results.csv`** (global):

  ```
  cell_id,horizon,split,n,MAE,RMSE,MAPE,wMAPE,sMAPE,Y_MEAN
  4259,1,train,6248,35.5383,55.3115,10.9738,10.7086,10.7675, ...
  4259,1,val,  1338,14.9333,21.4796, 7.2451, 6.8321, 7.1878, ...
  4259,1,test, 1339, 8.1264,10.5815, 9.6856, 9.4915, 9.6040, ...
  4259,6,train,6243,62.6277,91.9753,19.2594,18.8664,18.9206, ...
  ...
  ```
* **`persistence_cell_<id>.csv`** (por celda): todas sus filas (horizonte × split).

---

## Requisitos

* Python **3.10+**
* Librerías: `pandas`, `numpy`
* (Opcional) `matplotlib`/`notebook` para exploración

> Recomendado entorno virtual (Windows):
>
> ```bash
> python -m venv .venv
> .venv\Scripts\activate
> pip install pandas numpy
> ```

---

## Comandos típicos

> Ejecutar siempre desde la **raíz** del repo.

1. **Preprocesar** (genera diarios `*_internet_total.csv`):

```bash
python run_pipeline.py
```

2. **Unificar** y crear “una serie por celda”:

```bash
python -m Processing.timeseries_dataset
```

Salidas:

* `Data/processed/all_cells.csv`
* `Data/processed/by_cell/cell_<id>.csv`

3. **Evaluar Persistencia** (H=1 y H=6):

```bash
python -m Modeling.persistence.run_persistence
```

Salidas en:

```
Modeling/persistence/output/
  ├─ persistence_results.csv
  ├─ persistence_cell_4259.csv
  ├─ persistence_cell_4456.csv
  └─ ...
```

---

## Roadmap

* **Media móvil** (`Modeling/moving_average/`): elegir ventana K en **val**, comparar en **test** vs persistencia.
* **Redes neuronales** (MLP/GRU) con **ventanas W=12** y **H∈{1,6}**, reutilizando `data_access`, `targets`, `metrics` y `reporting`.
* Informe por **celda** con wMAPE/sMAPE + MAE/RMSE y análisis por **hora del día**.

---
