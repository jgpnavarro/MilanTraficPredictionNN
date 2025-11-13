# MilanTrafficPredictionNN

Predicción de **tráfico de Internet móvil** en Milán (Telecom Italia) sobre una **malla espacial** con intervalos de **10 minutos**.
El repositorio incluye el pipeline de **preprocesado**, la **unificación** de series por celda y utilidades para preparar los datos antes del modelado.

---

## Estructura del proyecto (¿qué hay en cada sitio?)

```
MilanTraficPredictionNN/
├─ Data/
│  ├─ raw/
│  └─ processed/
├─ Mapa Squares Milán/
├─ Processing/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ process_cells_internet.py
│  └─ timeseries_dataset.py
├─ run_pipeline.py
└─ Readme.md
```

### `Data/`

* **`raw/`**: ficheros **originales** por día (`sms-call-internet-mi-YYYY-MM-DD.txt`).
  Todos comparten el mismo formato y cubren del **2013-11-01** al **2014-01-01**.
* **`processed/`**: resultados del preprocesado:

  * `*_internet_total.csv` → **un CSV por día** ya filtrado a las celdas de interés y con la métrica **`internet_total`** cada 10 minutos.
  * `all_cells.csv` → **todo junto** (todas las fechas y celdas) en una **tabla larga**: `cell_id, datetime, internet_total`.
  * `by_cell/` → **un CSV por celda** (`cell_<id>.csv`) con su serie temporal completa (todos los días seguidos).

> Idea rápida: trabajas **a partir de `processed/`**. Los `raw/` solo sirven como origen.

### `Mapa Squares Milán/`

* **`SelectedSquares.kml`**: polígonos y **IDs de celda** (grid) de las zonas analizadas (Bocconi, Navigli, Duomo, Parco Forlanini, San Bovio, Cava Manara).
  No es un dato “operativo” del pipeline, pero documenta claramente qué celdas se usan.

### `Processing/`

* **`__init__.py`**: marca esta carpeta como **paquete Python** (permite usar `python -m Processing...`).
* **`config.py`**: **único punto de configuración**:

  * rutas (`Data/`, `raw/`, `processed/`),
  * **lista de `CELL_IDS`** a filtrar,
  * zona horaria (`TIMEZONE="Europe/Rome"`) para convertir correctamente desde epoch ms.
* **`process_cells_internet.py`**: **preprocesado diario**.
  Lee cada `.txt` de `raw/`, convierte el tiempo a `datetime` local, filtra por `CELL_IDS` y **agrega `internet_total`** por `(cell_id, datetime)` en pasos de 10'.
  **Salida**: `*_internet_total.csv` en `processed/`.
* **`timeseries_dataset.py`**: **unificación de series**.
  Lee todos los `*_internet_total.csv`, normaliza columnas, concatena y **ordena**.
  **Salidas**:

  * `processed/all_cells.csv` (tabla larga con todo),
  * `processed/by_cell/cell_<id>.csv` (una serie por celda).

### Raíz del repo

* **`run_pipeline.py`**: orquesta el **preprocesado inicial** (del `raw/` a `processed/*_internet_total.csv`).
  Se usa una vez (o cuando cambien los `raw/`/config).
* **`Readme.md`**: este documento.

---

## Requisitos

* Python **3.10+**
* Librerías: `pandas`
  *(recomendado usar entorno virtual: `python -m venv .venv && .venv\Scripts\activate && pip install pandas` en Windows)*

---

## Configuración rápida (`Processing/config.py`)

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "Data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Celdas/zonas de interés (IDs de la malla)
CELL_IDS = [4259, 4456, 4703, 5060, 5085, 5200]

# Zona horaria del dataset para convertir epoch ms a hora local
TIMEZONE = "Europe/Rome"
```

---

## Comandos habituales

> Ejecutar **siempre desde la raíz del repo**.

### 1) Preprocesar los brutos (genera `*_internet_total.csv`)

```bash
python run_pipeline.py
```

### 2) Unificar todo y crear “una serie por celda”

```bash
python -m Processing.timeseries_dataset
```

Resultados:

* `Data/processed/all_cells.csv`
* `Data/processed/by_cell/cell_<id>.csv`

---

## Formato de salida (lo esencial)

**`all_cells.csv`** (tabla larga):

```
cell_id,datetime,internet_total
4259,2013-11-01 00:00:00,261.68
4259,2013-11-01 00:10:00,189.42
...
4260,2013-11-01 00:00:00,210.35
...
```

**`by_cell/cell_4259.csv`** (una serie por celda):

```
cell_id,datetime,internet_total
4259,2013-11-01 00:00:00,261.68
4259,2013-11-01 00:10:00,189.42
...
```

> **`datetime`** está en hora local **Europe/Rome** tras convertir desde epoch ms.

---

## Uso programático (cargar en Python)

```python
from Processing.timeseries_dataset import load_unified, get_cell_timeseries

all_df = load_unified()                  # tabla larga con todas las celdas y fechas
df_4259 = get_cell_timeseries(all_df, 4259)  # serie de una celda (índice datetime)

print(all_df.head())
print(df_4259.head())
```

---

## Siguientes pasos (resumen)

1. Reindexar a malla exacta de 10' y decidir cómo tratar huecos (NaN / 0 / ffill).
2. Construir dataset supervisado por celda (ventana **W=12** → 2h; horizonte **H=6** → 1h).
3. Split temporal: **70% train · 15% val · 15% test**.
4. Baselines (persistencia, media móvil) y modelos (MLP/GRU).
5. Métricas (MAE, RMSE, MAPE) y, después, simulación del caso de uso energético (ACTIVE/SAVING).
