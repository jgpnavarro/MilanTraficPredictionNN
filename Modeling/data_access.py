"""
Acceso de datos minimalista para modelado.

Este módulo consume directamente los ficheros generados en Processing:
Data/processed/by_cell/cell_*.csv

Objetivo:
- Leer cada celda como una Serie (pd.Series) indexada por datetime.
- Iterar por celda de forma ordenada.
- Aplicar un split temporal 70/15/15 por posición.

No realiza reindexados ni validaciones pesadas. Asume que los datos están limpios.
"""

from __future__ import annotations

from typing import Generator, Tuple, List
from pathlib import Path
import re

import numpy as np
import pandas as pd

from Modeling.config import PROCESSED_DIR, SPLIT


# -----------------------------------------------------------------------------
# Localizar y ordenar ficheros por celda
# -----------------------------------------------------------------------------

def list_cell_files() -> List[Path]:
    """
    Devuelve la lista de ficheros 'cell_*.csv' en PROCESSED_DIR/by_cell,
    ordenada por el identificador numérico de la celda.
    """
    by_cell_dir = Path(PROCESSED_DIR) / "by_cell"
    files = sorted(by_cell_dir.glob("cell_*.csv"), key=_sort_key_by_cell_id)
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos 'cell_*.csv' en {by_cell_dir}.")
    return files


def _sort_key_by_cell_id(path: Path) -> int:
    """
    Extrae el identificador numérico del nombre de fichero 'cell_<id>.csv'
    para poder ordenar de forma natural por 'cell_id'.
    """
    m = re.search(r"cell_(\d+)\.csv$", path.name)
    return int(m.group(1)) if m else 0


# -----------------------------------------------------------------------------
# Cargar una celda como Serie temporal
# -----------------------------------------------------------------------------

def load_cell_series(fp: Path) -> pd.Series:
    """
    Carga un fichero 'cell_<id>.csv' y devuelve una Serie con índice datetime.

    Requisitos del CSV:
      - columnas: 'cell_id', 'datetime', 'internet_total'
      - 'datetime' convertible a Timestamp
    """
    df = pd.read_csv(fp, dtype={"cell_id": "Int64"})
    # Conversión a Timestamp y orden por si no viene estrictamente ordenado.
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime", kind="stable").reset_index(drop=True)

    s = df.set_index("datetime")["internet_total"].astype(float)
    s.name = "internet_total"
    return s


# -----------------------------------------------------------------------------
# Iterador de celdas
# -----------------------------------------------------------------------------

def iter_cells_by_file() -> Generator[Tuple[int, pd.Series], None, None]:
    """
    Itera sobre los ficheros 'cell_*.csv' y produce pares (cell_id, serie).
    """
    for fp in list_cell_files():
        cid = _sort_key_by_cell_id(fp)
        s = load_cell_series(fp)
        yield cid, s


# -----------------------------------------------------------------------------
# Split temporal 70/15/15
# -----------------------------------------------------------------------------

def temporal_split_indices(n: int, split: Tuple[float, float, float] = SPLIT) -> Tuple[int, int]:
    """
    Calcula los índices de corte (i_train, i_val) para un split temporal dado.

    Retorna:
        (i_train, i_val) para rebanados:
            - train: [0 : i_train)
            - val:   [i_train : i_val)
            - test:  [i_val : n)
    """
    p_train, p_val, _ = split
    i_train = int(n * p_train)
    i_val = int(n * (p_train + p_val))
    # Acotar por si la serie fuese muy corta.
    i_train = max(0, min(i_train, n))
    i_val = max(i_train, min(i_val, n))
    return i_train, i_val


def split_series(s: pd.Series, split: Tuple[float, float, float] = SPLIT) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Divide una serie temporal en train, val y test por posición temporal.
    """
    n = len(s)
    i_train, i_val = temporal_split_indices(n, split=split)
    s_train = s.iloc[:i_train]
    s_val = s.iloc[i_train:i_val]
    s_test = s.iloc[i_val:]
    return s_train, s_val, s_test
