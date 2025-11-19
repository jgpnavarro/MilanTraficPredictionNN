"""
Agregación y salida de resultados de evaluación.

Este módulo reúne resultados por celda, horizonte y split en una tabla
única (DataFrame), imprime resúmenes por grupo y, opcionalmente, guarda
los resultados en disco en el directorio que se le indique.

Métricas incluidas:
- Absolutas: MAE, RMSE
- Porcentuales: MAPE, wMAPE, sMAPE
- Estadístico descriptivo adicional: Y_MEAN (media del valor real evaluado)
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Optional
from pathlib import Path

import pandas as pd

from Baselines.config import SAVE_RESULTS


# Columnas estándar del DataFrame de resultados
BASE_COLS: List[str] = [
    "cell_id", "horizon", "split", "n",
    "MAE", "RMSE", "MAPE", "wMAPE", "sMAPE",
    "Y_MEAN",
]


def rows_from_eval_dict(
    cell_id: int,
    horizon: int,
    eval_dict: Dict[str, Dict[str, float]],
) -> List[Dict]:
    """
    Convierte el diccionario de métricas por split en una lista de filas homogéneas.

    Parámetros
    ----------
    cell_id : int
        Identificador de la celda.
    horizon : int
        Horizonte de predicción (en pasos de la serie).
    eval_dict : dict
        Diccionario con claves "train", "val", "test" (u otras) y, como valor,
        otro dict con las métricas calculadas.

    Devuelve
    --------
    rows : list[dict]
        Lista de filas preparadas para construir un DataFrame.
    """
    rows: List[Dict] = []
    for split_name, metrics in eval_dict.items():
        rows.append({
            "cell_id": int(cell_id),
            "horizon": int(horizon),
            "split": split_name,
            "n": int(metrics.get("n", 0)),
            "MAE": float(metrics.get("MAE", float("nan"))),
            "RMSE": float(metrics.get("RMSE", float("nan"))),
            "MAPE": float(metrics.get("MAPE", float("nan"))),
            "wMAPE": float(metrics.get("wMAPE", float("nan"))),
            "sMAPE": float(metrics.get("sMAPE", float("nan"))),
            "Y_MEAN": float(metrics.get("Y_MEAN", float("nan"))),
        })
    return rows


def aggregate_results(all_rows: List[Dict]) -> pd.DataFrame:
    """
    Construye un DataFrame con todas las filas recogidas durante la evaluación.

    Si no hay filas, devuelve un DataFrame vacío con las columnas estándar.
    """
    if not all_rows:
        return pd.DataFrame(columns=BASE_COLS)

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["horizon", "cell_id", "split"], kind="stable").reset_index(drop=True)
    return df


def print_summary(df: pd.DataFrame, include_y_mean: bool = True) -> None:
    """
    Imprime un resumen compacto por horizonte y split con medias y totales.

    Parámetros
    ----------
    df : DataFrame
        Tabla con resultados por celda/horizonte/split.
    include_y_mean : bool
        Si True, muestra también la media de Y_MEAN por grupo.
    """
    if df.empty:
        print("No hay resultados para resumir.")
        return

    agg_dict = {
        "MAE": "mean",
        "RMSE": "mean",
        "MAPE": "mean",
        "wMAPE": "mean",
        "sMAPE": "mean",
        "n": "sum",
    }
    if include_y_mean:
        agg_dict["Y_MEAN"] = "mean"  # media simple entre celdas

    summary = (
        df.groupby(["horizon", "split"], as_index=False)
          .agg(agg_dict)
          .rename(columns={
              "MAE": "MAE_mean",
              "RMSE": "RMSE_mean",
              "MAPE": "MAPE_mean",
              "wMAPE": "wMAPE_mean",
              "sMAPE": "sMAPE_mean",
              "Y_MEAN": "Y_MEAN_mean",
              "n": "n_total",
          })
          .sort_values(["horizon", "split"], kind="stable")
    )

    print("\n== Resumen por horizonte y split ==")
    print(summary.to_string(index=False))


def print_per_cell(
    df: pd.DataFrame,
    cells: Optional[Sequence[int]] = None,
    columns: Optional[Sequence[str]] = None,
) -> None:
    """
    Imprime métricas por celda, separadas por horizonte y split.

    Parámetros
    ----------
    df : DataFrame
        Tabla con resultados.
    cells : secuencia de int, opcional
        Lista de cell_id a mostrar. Si es None, se muestran todas las celdas.
    columns : secuencia de str, opcional
        Columnas a mostrar en cada tabla por celda.
    """
    if df.empty:
        print("No hay resultados para mostrar por celda.")
        return

    if cells is not None:
        df = df[df["cell_id"].isin(cells)]

    if df.empty:
        print("No hay filas para los cell_id indicados.")
        return

    if columns is None:
        columns = ["horizon", "split", "n", "Y_MEAN", "MAE", "RMSE", "MAPE", "wMAPE", "sMAPE"]

    # Orden estable por celda, horizonte y split
    df = df.sort_values(["cell_id", "horizon", "split"], kind="stable")

    for cid, g in df.groupby("cell_id"):
        print(f"\n== Celda {cid} ==")
        g_sel = g[list(columns)].sort_values(["horizon", "split"], kind="stable")
        print(g_sel.to_string(index=False))


def save_per_cell_csv(
    df: pd.DataFrame,
    output_dir: Path,
    filename_prefix: str = "results_cell_",
) -> None:
    """
    Guarda un CSV por celda con todas las filas (horizonte/split) de esa celda.

    Parámetros
    ----------
    df : DataFrame
        Resultados completos (todas las celdas).
    output_dir : Path
        Directorio donde se guardarán los ficheros.
    filename_prefix : str
        Prefijo del nombre de fichero (por ejemplo, 'persistence_cell_').
    """
    if df.empty:
        print("No hay resultados para guardar por celda.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for cid, g in df.groupby("cell_id"):
        out_fp = output_dir / f"{filename_prefix}{int(cid)}.csv"
        (
            g.sort_values(["horizon", "split"], kind="stable")
             .to_csv(out_fp, index=False)
        )

    print(f"Guardados {df['cell_id'].nunique()} ficheros por celda en: {output_dir}")


def save_results(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "results.csv",
) -> Path:
    """
    Guarda un único CSV con todos los resultados.

    Parámetros
    ----------
    df : DataFrame
        Resultados completos.
    output_dir : Path
        Directorio donde se guardará el fichero.
    filename : str
        Nombre del fichero (por defecto 'results.csv').

    Devuelve
    --------
    out_path : Path
        Ruta completa del fichero (aunque SAVE_RESULTS sea False).
    """
    out_path = output_dir / filename

    if SAVE_RESULTS:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nResultados guardados en: {out_path}")
    else:
        print("\nSAVE_RESULTS=False -> no se guardó archivo en disco.")
        print(f"Ruta sugerida: {out_path}")

    return out_path
