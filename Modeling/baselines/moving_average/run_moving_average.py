"""
Script de evaluación para el modelo de media móvil.

Este módulo:
    - Recorre todas las celdas disponibles.
    - Divide sus series en train/val/test.
    - Evalúa la media móvil para varios horizontes y tamaños de ventana.
    - Agrega los resultados y los guarda en CSV.

Depende de:
    - Modeling.config           (H_LIST, MOVING_AVG_WINDOWS, MOVING_AVG_OUTPUT_DIR)
    - Modeling.data_access      (iter_cells_by_file, split_series)
    - Modeling.moving_average   (evaluate_moving_average_splits)
    - Modeling.reporting        (rows_from_eval_dict, aggregate_results,
                                 print_summary, print_per_cell,
                                 save_results, save_per_cell_csv)
"""

from __future__ import annotations

from typing import List, Dict

import pandas as pd

from Modeling.config import (
    H_LIST,
    MOVING_AVG_WINDOWS,
    MOVING_AVG_OUTPUT_DIR,
)
from Modeling.data_access import iter_cells_by_file, split_series
from Modeling.baselines.moving_average.moving_average import evaluate_moving_average_splits
from Modeling.reporting import (
    rows_from_eval_dict,
    aggregate_results,
    print_summary,
    print_per_cell,
    save_results,
    save_per_cell_csv,
)


def run_moving_average_for_cell(
    cell_id: int,
    series: pd.Series,
) -> List[Dict]:
    """
    Evalúa la media móvil para una celda concreta.

    Para la celda indicada:
        - Divide la serie en train/val/test.
        - Para cada horizonte H y cada ventana W, calcula métricas
          en train, val y test.
        - Convierte esos resultados en filas (dicts) compatibles
          con las funciones de reporting.

    Parámetros
    ----------
    cell_id : int
        Identificador de la celda.
    series : pd.Series
        Serie temporal completa de la celda (índice = datetime).

    Devuelve
    --------
    rows_cell : list[dict]
        Lista de filas con resultados para esta celda.
        Cada fila corresponde a una combinación (horizon, split)
        y contiene las métricas calculadas.
    """
    # 1) Dividir la serie en train / val / test
    s_train, s_val, s_test = split_series(series)

    rows_cell: List[Dict] = []

    # 2) Recorrer todos los horizontes y tamaños de ventana
    for H in H_LIST:
        for window in MOVING_AVG_WINDOWS:
            # Evaluar media móvil para este H y esta ventana
            eval_dict = evaluate_moving_average_splits(
                full_series=series,
                s_train=s_train,
                s_val=s_val,
                s_test=s_test,
                cell_id=cell_id,
                horizon=H,
                window=window,
            )

            # Convertir el dict de resultados por split en filas homogéneas
            rows = rows_from_eval_dict(
                cell_id=cell_id,
                horizon=H,
                eval_dict=eval_dict,
            )

            # Añadir información de la ventana a cada fila
            for r in rows:
                r["window"] = int(window)

            rows_cell.extend(rows)

    return rows_cell


def main() -> None:
    """
    Punto de entrada principal del script.

    Recorre todas las celdas, acumula resultados de media móvil
    y genera:
        - Un resumen por horizonte y split.
        - Un CSV global con todos los resultados.
        - Un CSV por celda con las filas correspondientes.
    """
    all_rows: List[Dict] = []

    # 1) Recorrer todas las celdas disponibles
    for cell_id, series in iter_cells_by_file():
        print(f"[moving_average] Procesando celda {cell_id}...")

        rows_cell = run_moving_average_for_cell(cell_id, series)
        all_rows.extend(rows_cell)

    # 2) Construir DataFrame con todos los resultados
    df_results = aggregate_results(all_rows)

    # 3) Mostrar un resumen global por horizonte y split
    print_summary(df_results, include_y_mean=True)

    # (Opcional) Mostrar métricas por celda
    # Si se quiere limitar a algunas celdas, se puede pasar una lista:
    #   print_per_cell(df_results, cells=[4259, 4456])
    print_per_cell(df_results)

    # 4) Guardar resultados en CSV (global y por celda)
    save_results(
        df_results,
        output_dir=MOVING_AVG_OUTPUT_DIR,
        filename="moving_avg_results.csv",
    )

    save_per_cell_csv(
        df_results,
        output_dir=MOVING_AVG_OUTPUT_DIR,
        filename_prefix="moving_avg_cell_",
    )


if __name__ == "__main__":
    main()
