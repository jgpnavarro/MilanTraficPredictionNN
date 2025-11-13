"""
Agregación y salida de resultados de evaluación.

Este módulo reúne resultados por celda, horizonte y split en una tabla
única (DataFrame), imprime resúmenes por grupo y, opcionalmente, guarda
los resultados en disco dentro de Modeling/outputs.

Métricas incluidas:
- Absolutas: MAE, RMSE
- Porcentuales: MAPE, wMAPE, sMAPE
- Estadístico descriptivo adicional: Y_MEAN (media del valor real evaluado)
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Optional
from pathlib import Path
import pandas as pd

from Modeling.config import SAVE_RESULTS, PERSISTENCE_OUTPUT_DIR


# Columnas estándar del DataFrame de resultados
BASE_COLS: List[str] = [
    "cell_id", "horizon", "split", "n",
    "MAE", "RMSE", "MAPE", "wMAPE", "sMAPE",
    "Y_MEAN",
]


def rows_from_eval_dict(
    cell_id: int,
    horizon: int,
    eval_dict: Dict[str, Dict[str, float]]
) -> List[Dict]:
    """
    Convierte el diccionario de métricas por split en una lista de filas homogéneas.
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
    """
    if not all_rows:
        return pd.DataFrame(columns=BASE_COLS)
    df = pd.DataFrame(all_rows)
    df = df.sort_values(["horizon", "cell_id", "split"], kind="stable").reset_index(drop=True)
    return df


def print_summary(df: pd.DataFrame, include_y_mean: bool = True) -> None:
    """
    Imprime un resumen compacto por horizonte y split con medias y totales.
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
    columns: Optional[Sequence[str]] = None
) -> None:
    """
    Imprime métricas por celda, separadas por horizonte y split.
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

    for cid, g in df.sort_values(["cell_id", "horizon", "split"]).groupby("cell_id"):
        print(f"\n== Celda {cid} ==")
        g = g[list(columns)].sort_values(["horizon", "split"], kind="stable")
        print(g.to_string(index=False))


def save_per_cell_csv(
    df: pd.DataFrame,
    filename_prefix: str = "persistence_cell_",
    output_dir: Path = PERSISTENCE_OUTPUT_DIR,   # <-- nuevo
) -> None:
    if df.empty:
        print("No hay resultados para guardar por celda.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for cid, g in df.groupby("cell_id"):
        out_fp = output_dir / f"{filename_prefix}{int(cid)}.csv"
        g.sort_values(["horizon","split"], kind="stable").to_csv(out_fp, index=False)
    print(f"Guardados {df['cell_id'].nunique()} ficheros por celda en: {output_dir}")


def save_results(
    df: pd.DataFrame,
    filename: str = "persistence_results.csv",
    output_dir: Path = PERSISTENCE_OUTPUT_DIR,   # <-- nuevo
) -> Path:
    out_path = output_dir / filename
    if SAVE_RESULTS:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nResultados guardados en: {out_path}")
    else:
        print("\nSAVE_RESULTS=False -> no se guardó archivo en disco.")
        print(f"Ruta sugerida: {out_path}")
    return out_path

