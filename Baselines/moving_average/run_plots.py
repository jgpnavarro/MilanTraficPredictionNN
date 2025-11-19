"""
Generación de gráficos para el modelo de media móvil.

Para cada celda se crean DOS figuras:

1) Figura "grande" (toda la serie):
    - Serie real completa.
    - Todas las series predichas por media móvil para:
        * Cada horizonte H en H_LIST.
        * Cada ventana W en MOVING_AVG_WINDOWS.
    - Zonas de train, val y test sombreadas.
    -> se guarda en: .../plots_all/cell_<id>_moving_avg_all.png

2) Figura "semana" (zoom en la última semana de datos):
    - Igual que la anterior, pero solo para los últimos 7 días.
    -> se guarda en: .../plots_week/cell_<id>_moving_avg_week.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from Baselines.config import (
    H_LIST,
    MOVING_AVG_WINDOWS,
    MOVING_AVG_OUTPUT_DIR,
)
from Baselines.data_access import iter_cells_by_file, split_series
from Baselines.moving_average.moving_average import moving_average_forecast


def _get_split_ranges(s_train: pd.Series,
                      s_val: pd.Series,
                      s_test: pd.Series) -> dict:
    """
    Obtiene los rangos temporales (inicio, fin) de cada split.

    Devuelve un diccionario:
        {
            "train": (t0_train, t1_train),
            "val":   (t0_val,   t1_val),
            "test":  (t0_test,  t1_test),
        }
    """
    ranges = {}

    if len(s_train) > 0:
        ranges["train"] = (s_train.index[0], s_train.index[-1])
    if len(s_val) > 0:
        ranges["val"] = (s_val.index[0], s_val.index[-1])
    if len(s_test) > 0:
        ranges["test"] = (s_test.index[0], s_test.index[-1])

    return ranges


def _plot_splits_background(ax, split_ranges: dict) -> None:
    """
    Pinta el fondo sombreado para train, val y test.

    Cada split se representa como una franja vertical con un color suave.
    """
    colors = {
        "train": "green",
        "val": "orange",
        "test": "red",
    }

    for name, (t0, t1) in split_ranges.items():
        color = colors.get(name, "gray")
        ax.axvspan(t0, t1, alpha=0.05, color=color, label=f"{name} (fondo)")


def plot_moving_average_for_cell_full(
    cell_id: int,
    series: pd.Series,
    out_dir: Path,
) -> None:
    """
    Genera y guarda la figura de media móvil para una celda
    usando TODA la serie temporal.
    """
    # 1) Dividir en train / val / test
    s_train, s_val, s_test = split_series(series)
    split_ranges = _get_split_ranges(s_train, s_val, s_test)

    # 2) Crear figura y eje
    fig, ax = plt.subplots(figsize=(12, 6))

    # 3) Serie real completa
    ax.plot(series.index, series.values, label="Real", linewidth=1.0)

    # 4) Predicciones por media móvil para todas las combinaciones (H, W)
    for H in H_LIST:
        for window in MOVING_AVG_WINDOWS:
            y_pred_full = moving_average_forecast(
                series=series,
                horizon=H,
                window=window,
            )

            mask = ~y_pred_full.isna()
            y_pred = y_pred_full[mask]

            if len(y_pred) == 0:
                continue

            label = f"MA H={H}, W={window}"
            ax.plot(
                y_pred.index,
                y_pred.values,
                linestyle="--",
                linewidth=0.8,
                label=label,
            )

    # 5) Fondo para train / val / test
    _plot_splits_background(ax, split_ranges)

    # 6) Ajustes estéticos básicos
    ax.set_title(f"Celda {cell_id} – Media móvil (todas ventanas y horizontes)")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("internet_total")
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    ax.grid(True)
    fig.tight_layout()

    # 7) Guardar figura
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cell_{cell_id}_moving_avg_all.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[moving_average_plots] Guardada figura completa para celda {cell_id} en: {out_path}")


def plot_moving_average_for_cell_week(
    cell_id: int,
    series: pd.Series,
    out_dir: Path,
) -> None:
    """
    Genera y guarda una figura con SOLO la última semana de datos.

    La figura contiene:
        - Serie real de la última semana.
        - Predicciones de media móvil (todas las combinaciones H, W)
          recortadas a esa semana.
        - Fondo con las zonas de train, val y test recortadas a esa semana.
    """
    # 1) Dividir en train / val / test (para saber rangos originales)
    s_train, s_val, s_test = split_series(series)
    full_split_ranges = _get_split_ranges(s_train, s_val, s_test)

    # 2) Calcular ventana temporal de la última semana
    end_time = series.index.max()
    start_time = end_time - pd.Timedelta(days=7)

    # Subserie de la semana
    series_week = series.loc[start_time:end_time]

    if len(series_week) == 0:
        print(f"[moving_average_plots] Celda {cell_id}: sin datos en la última semana, se omite figura semanal.")
        return

    # 3) Adaptar rangos de splits a la ventana semanal
    split_ranges_week = {}
    for name, (t0, t1) in full_split_ranges.items():
        # Intersección del rango del split con [start_time, end_time]
        t_start = max(t0, start_time)
        t_end = min(t1, end_time)
        if t_start <= t_end:
            split_ranges_week[name] = (t_start, t_end)

    # 4) Crear figura y eje
    fig, ax = plt.subplots(figsize=(12, 6))

    # 5) Serie real (semana)
    ax.plot(series_week.index, series_week.values, label="Real", linewidth=1.0)

    # 6) Predicciones de media móvil (semana)
    for H in H_LIST:
        for window in MOVING_AVG_WINDOWS:
            y_pred_full = moving_average_forecast(
                series=series,
                horizon=H,
                window=window,
            )

            # Recortar a la semana y eliminar NaN
            y_pred_week = y_pred_full.loc[start_time:end_time]
            mask = ~y_pred_week.isna()
            y_pred_week = y_pred_week[mask]

            if len(y_pred_week) == 0:
                continue

            label = f"MA H={H}, W={window}"
            ax.plot(
                y_pred_week.index,
                y_pred_week.values,
                linestyle="--",
                linewidth=0.8,
                label=label,
            )

    # 7) Fondo de splits adaptado a la semana
    _plot_splits_background(ax, split_ranges_week)

    # 8) Ajustes estéticos
    ax.set_title(f"Celda {cell_id} – Media móvil (última semana)")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("internet_total")
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    ax.grid(True)
    fig.tight_layout()

    # 9) Guardar figura
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cell_{cell_id}_moving_avg_week.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[moving_average_plots] Guardada figura semanal para celda {cell_id} en: {out_path}")


def main() -> None:
    """
    Recorre todas las celdas y genera:
        - Una figura con toda la serie.
        - Una figura con solo la última semana.
    """
    plots_dir_all = MOVING_AVG_OUTPUT_DIR / "plots_all"
    plots_dir_week = MOVING_AVG_OUTPUT_DIR / "plots_week"

    for cell_id, series in iter_cells_by_file():
        print(f"[moving_average_plots] Procesando celda {cell_id}...")

        plot_moving_average_for_cell_full(
            cell_id=cell_id,
            series=series,
            out_dir=plots_dir_all,
        )

        plot_moving_average_for_cell_week(
            cell_id=cell_id,
            series=series,
            out_dir=plots_dir_week,
        )


if __name__ == "__main__":
    main()
