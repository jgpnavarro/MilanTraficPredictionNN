# Processing/timeseries_dataset.py
from pathlib import Path
from typing import List

import pandas as pd

from .config import PROCESSED_DIR, CELL_IDS, TIMEZONE


def list_daily_csv() -> List[Path]:
    """Localiza los ficheros diarios *_internet_total.csv en PROCESSED_DIR, ordenados por nombre."""
    files = sorted(PROCESSED_DIR.glob("*_internet_total.csv"))
    if not files:
        raise FileNotFoundError(
            f"No se han encontrado archivos *_internet_total.csv en {PROCESSED_DIR}"
        )
    return files


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza el esquema mínimo requerido:
      - 'square_id' -> 'cell_id' (si ya existe 'cell_id', se respeta)
      - 'time_interval' (ms desde epoch) -> 'datetime' (Timestamp naive)
      - mantiene 'internet_total'
    Devuelve únicamente: ['cell_id', 'datetime', 'internet_total'].
    """
    if "cell_id" not in df.columns:
        if "square_id" in df.columns:
            df = df.rename(columns={"square_id": "cell_id"})
        else:
            raise KeyError("No se encuentra 'cell_id' ni 'square_id'.")

    if "datetime" not in df.columns:
        if "time_interval" in df.columns:
            # 1) Epoch ms -> UTC tz-aware
            dt_utc = pd.to_datetime(df["time_interval"], unit="ms", utc=True)
            # 2) Convertir a hora local del dataset (Milán)
            dt_local = dt_utc.dt.tz_convert(TIMEZONE)
            # 3) Quitar tz (naïve) manteniendo la hora local correcta
            df["datetime"] = dt_local.dt.tz_localize(None)
        else:
            raise KeyError("No se encuentra 'datetime' ni 'time_interval'.")


    if "internet_total" not in df.columns:
        raise KeyError("No se encuentra la columna 'internet_total'.")

    return df[["cell_id", "datetime", "internet_total"]]


def load_unified() -> pd.DataFrame:
    """
    Lee todos los CSV diarios, normaliza columnas, concatena y ordena.
    Filtra a CELL_IDS (si se ha definido en config.py).
    """
    frames: List[pd.DataFrame] = []
    for fp in list_daily_csv():
        df = pd.read_csv(fp)
        df = normalize_columns(df)
        if CELL_IDS:
            df = df[df["cell_id"].isin(CELL_IDS)]
        frames.append(df)

    unified = pd.concat(frames, ignore_index=True)
    unified = unified.sort_values(["cell_id", "datetime"]).reset_index(drop=True)
    return unified


def save_outputs(unified: pd.DataFrame) -> None:
    """
    Guarda:
      - Data/processed/all_cells.csv
      - Data/processed/by_cell/cell_<id>.csv (un archivo por celda)
    """
    out_all = PROCESSED_DIR / "all_cells.csv"
    unified.to_csv(out_all, index=False)

    out_dir = PROCESSED_DIR / "by_cell"
    out_dir.mkdir(parents=True, exist_ok=True)

    for cid, df_c in unified.groupby("cell_id", sort=True):
        df_c = df_c.sort_values("datetime")
        df_c.to_csv(out_dir / f"cell_{int(cid)}.csv", index=False)


def build_and_save() -> None:
    """
    Orquestación simple:
      1) Unifica todos los CSV diarios de PROCESSED_DIR.
      2) Guarda el agregado y los archivos por celda en PROCESSED_DIR.
    """
    unified = load_unified()
    save_outputs(unified)


if __name__ == "__main__":
    # Permite ejecutar el módulo directamente:
    #   python -m Processing.timeseries_dataset_basic
    # o  python Processing/timeseries_dataset_basic.py
    print("== Building unified time series and saving outputs ==")
    print(f"Reading from: {PROCESSED_DIR}")
    build_and_save()
    print("== Done. Outputs written to Data/processed/ ==")
