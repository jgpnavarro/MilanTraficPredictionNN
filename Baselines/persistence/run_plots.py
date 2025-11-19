"""
Generador de gráficas 'ALL-IN-ONE' para el modelo de persistencia.

Para cada celda se crean DOS figuras:

1) Figura "grande" (toda la serie):
    - Serie real completa.
    - Todas las curvas de predicción por persistencia (cada H en H_LIST).
    - Zonas de train, val y test sombreadas.
    -> .../plots_all/cell_<id>_persistence_all.png

2) Figura "semana" (zoom en la última semana de datos):
    - Igual que la anterior, pero solo para los últimos 7 días.
    -> .../plots_week/cell_<id>_persistence_week.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from Baselines.config import H_LIST, PERSISTENCE_OUTPUT_DIR
from Baselines.data_access import iter_cells_by_file, split_series


def _aligned_pred_for_horizon(series: pd.Series, H: int) -> pd.Series:
    """
    Construye la serie de predicciones de persistencia para la serie completa.

    Regla de persistencia:
        ŷ(t+H) = y(t)

    Implementación:
        - X = serie original (valor actual).
        - y = serie desplazada H pasos hacia arriba (futuro real).
        - Se seleccionan los índices donde y no es NaN.
        - La predicción es X en esos índices (valor actual).
    """
    X = series
    y = series.shift(-H)

    mask = y.notna()
    y_pred = X[mask].astype(float)
    y_pred.index = y.index[mask]

    return y_pred


def _get_split_ranges(
    s_train: pd.Series,
    s_val: pd.Series,
    s_test: pd.Series,
) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Obtiene los rangos temporales (inicio, fin) de cada split.

    Devuelve un diccionario:
        {
            "train": (t0_train, t1_train),
            "val":   (t0_val,   t1_val),
            "test":  (t0_test,  t1_test),
        }
    """
    ranges: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}

    if len(s_train) > 0:
        ranges["train"] = (s_train.index[0], s_train.index[-1])
    if len(s_val) > 0:
        ranges["val"] = (s_val.index[0], s_val.index[-1])
    if len(s_test) > 0:
        ranges["test"] = (s_test.index[0], s_test.index[-1])

    return ranges


def _plot_splits_background(ax, split_ranges: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]) -> None:
    """
    Pinta el fondo sombreado para train, val y test.
    """
    colors = {
        "train": "green",
        "val": "orange",
        "test": "red",
    }

    for name, (t0, t1) in split_ranges.items():
        color = colors.get(name, "gray")
        ax.axvspan(t0, t1, alpha=0.05, color=color, label=f"{name} (fondo)")


def plot_persistence_for_cell_full(
    cell_id: int,
    series: pd.Series,
    out_dir: Path,
) -> None:
    """
    Genera y guarda la figura de persistencia para una celda
    usando TODA la serie temporal.
    """
    s_train, s_val, s_test = split_series(series)
    split_ranges = _get_split_ranges(s_train, s_val, s_test)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Serie real
    ax.plot(series.index, series.values, label="Real", linewidth=1.0)

    # Curvas de persistencia (todas las H)
    for H in H_LIST:
        y_pred = _aligned_pred_for_horizon(series, H)

        if len(y_pred) == 0:
            continue

        label = f"Persistencia H={H}"
        ax.plot(
            y_pred.index,
            y_pred.values,
            linestyle="--",
            linewidth=0.8,
            label=label,
        )

    _plot_splits_background(ax, split_ranges)

    ax.set_title(f"Celda {cell_id} – Persistencia (todos los horizontes)")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("internet_total")
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    ax.grid(True)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cell_{cell_id}_persistence_all.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[persistence_plots] Guardada figura completa para celda {cell_id} en: {out_path}")


def plot_persistence_for_cell_week(
    cell_id: int,
    series: pd.Series,
    out_dir: Path,
) -> None:
    """
    Genera y guarda una figura de persistencia solo para la última semana.

    Incluye:
        - Serie real de la última semana.
        - Predicciones de persistencia (todas las H) recortadas a esa semana.
        - Fondo de train/val/test recortado a esa semana.
    """
    s_train, s_val, s_test = split_series(series)
    full_split_ranges = _get_split_ranges(s_train, s_val, s_test)

    end_time = series.index.max()
    start_time = end_time - pd.Timedelta(days=7)

    series_week = series.loc[start_time:end_time]
    if len(series_week) == 0:
        print(f"[persistence_plots] Celda {cell_id}: sin datos en la última semana, se omite figura semanal.")
        return

    # Adaptar rangos de splits a la semana
    split_ranges_week: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for name, (t0, t1) in full_split_ranges.items():
        t_start = max(t0, start_time)
        t_end = min(t1, end_time)
        if t_start <= t_end:
            split_ranges_week[name] = (t_start, t_end)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Serie real (semana)
    ax.plot(series_week.index, series_week.values, label="Real", linewidth=1.0)

    # Curvas de persistencia (semana)
    for H in H_LIST:
        y_pred_full = _aligned_pred_for_horizon(series, H)
        y_pred_week = y_pred_full.loc[start_time:end_time]
        if len(y_pred_week) == 0:
            continue

        label = f"Persistencia H={H}"
        ax.plot(
            y_pred_week.index,
            y_pred_week.values,
            linestyle="--",
            linewidth=0.8,
            label=label,
        )

    _plot_splits_background(ax, split_ranges_week)

    ax.set_title(f"Celda {cell_id} – Persistencia (última semana)")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("internet_total")
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    ax.grid(True)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cell_{cell_id}_persistence_week.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[persistence_plots] Guardada figura semanal para celda {cell_id} en: {out_path}")


def main() -> None:
    """
    Recorre todas las celdas y genera:
        - Una figura con toda la serie.
        - Una figura con solo la última semana.
    """
    plots_dir_all = PERSISTENCE_OUTPUT_DIR / "plots_all"
    plots_dir_week = PERSISTENCE_OUTPUT_DIR / "plots_week"

    for cell_id, series in iter_cells_by_file():
        print(f"[persistence_plots] Procesando celda {cell_id}...")

        plot_persistence_for_cell_full(
            cell_id=cell_id,
            series=series,
            out_dir=plots_dir_all,
        )

        plot_persistence_for_cell_week(
            cell_id=cell_id,
            series=series,
            out_dir=plots_dir_week,
        )


if __name__ == "__main__":
    main()
