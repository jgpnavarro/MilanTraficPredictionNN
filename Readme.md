# MilanTrafficPredictionNN

Predicción de **tráfico de Internet móvil** en Milán (dataset Telecom Italia) sobre una **malla espacial** con resolución temporal de **10 minutos**.
Este repositorio contiene el pipeline de preprocesado, la unificación de series temporales por celda y utilidades para preparar los datos de cara al modelado.

---

## Estructura del proyecto

```
MilanTraficPredictionNN/
├─ Data/
│  ├─ raw/                    # Ficheros originales día a día (.txt)
│  └─ processed/              # Salidas del preprocesado
│     ├─ *_internet_total.csv # Un CSV por día (internet_total por celda y 10')
│     ├─ all_cells.csv        # TODAS las celdas y fechas unificadas (tabla larga)
│     └─ by_cell/
│        ├─ cell_4259.csv     # Una serie por celda (todas las fechas seguidas)
│        └─ ...               # Resto de celdas de interés
│
├─ Mapa Squares Milán/
│  └─ SelectedSquares.kml     # Polígonos/IDs de celdas de interés
│
├─ Processing/
│  ├─ __init__.py
│  ├─ config.py               # Rutas, celdas de interés, zona horaria (Europe/Rome)
│  ├─ process_cells_internet.py  # Filtrado + agregación a internet_total (10')
│  └─ timeseries_dataset.py      # Unificación y guardado (global y por celda)
│
├─ run_pipeline.py            # Orquestación del preprocesado
└─ Readme.md
```

---

## Requisitos

* Python **3.10+**
* Paquetes:

  * `pandas`

Sugerencia: entorno virtual con `venv` y `pip install pandas`.

---

## Configuración

Editar `Processing/config.py`:

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "Data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Celdas/zonas de interés (IDs de la malla)
CELL_IDS = [4259, 4260, 4359, 4360, 4455, 4456, 4703, 5060, 5085, 5200]

# Zona horaria del dataset (para convertir desde epoch ms a local)
TIMEZONE = "Europe/Rome"
```

---

## Uso

> Ejecutar siempre desde la **raíz** del repo.

### 1) Preprocesado inicial (filtra celdas y agrega internet_total por día)

Genera `*_internet_total.csv` en `Data/processed/`.

```bash
python run_pipeline.py
```

### 2) Unificación (todo junto y por celda)

Crea `Data/processed/all_cells.csv` y un CSV por celda en `Data/processed/by_cell/`.

```bash
python -m Processing.timeseries_dataset
```

---

## Formato de los datos unificados

**`Data/processed/all_cells.csv`** (tabla larga):

| cell_id | datetime            | internet_total |
| ------: | ------------------- | -------------- |
|    4259 | 2013-11-01 00:00:00 | 261.68         |
|    4259 | 2013-11-01 00:10:00 | 189.42         |
|    4259 | 2013-11-01 00:20:00 | 231.90         |
|    4260 | 2013-11-01 00:00:00 | 210.35         |
|    4260 | 2013-11-01 00:10:00 | 198.77         |

**`Data/processed/by_cell/cell_4259.csv`** (una serie por celda):

| cell_id | datetime            | internet_total |
| ------: | ------------------- | -------------- |
|    4259 | 2013-11-01 00:00:00 | 261.68         |
|    4259 | 2013-11-01 00:10:00 | 189.42         |
|    4259 | 2013-11-01 00:20:00 | 231.90         |
|       … | …                   | …              |

> Las marcas temporales están en **hora local (Europe/Rome)** tras convertir desde epoch ms.

---

## Carga programática (ejemplo)

```python
from Processing.timeseries_dataset import load_unified, get_cell_timeseries

all_df = load_unified()               # tabla larga con todas las celdas y fechas
df_4259 = get_cell_timeseries(all_df, 4259)  # serie de una celda (índice datetime)

print(all_df.head())
print(df_4259.head())
```

---

## Roadmap (siguientes pasos)

1. Reindexado a malla exacta de 10 minutos y gestión de huecos (NaN/0/ffill).
2. Construcción del dataset supervisado por celda:

   * Ventana entrada **W=12** (2 h), horizonte **H=6** (1 h).
3. Partición temporal: **70% train · 15% val · 15% test**.
4. Baselines: **persistencia** y **media móvil**.
5. Modelos: **MLP** y **GRU** con normalización y early stopping.
6. Evaluación: **MAE**, **RMSE**, **MAPE** por celda.
7. Caso de uso energético: niveles **L/M/H** y simulación **ACTIVE/SAVING** (% ahorro y riesgo).

---

## Notas

* `cell_id` se refiere a **celdas de malla**, no a estaciones base físicas.
* `SelectedSquares.kml` documenta los polígonos/IDs usados (Bocconi, Navigli, Duomo, Parco Forlanini, San Bovio, Cava Manara).
* El preprocesado y la unificación están pensados para ser **idempotentes** y reproducibles.
