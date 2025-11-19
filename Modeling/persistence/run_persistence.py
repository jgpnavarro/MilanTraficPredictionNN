# Modeling/persistence/run_persistence.py

from __future__ import annotations

from typing import List, Dict

from Modeling.config import H_LIST, PERSISTENCE_OUTPUT_DIR
from Modeling.data_access import iter_cells_by_file, split_series
from Modeling.persistence.persistence import evaluate_persistence_splits
from Modeling.reporting import (
    rows_from_eval_dict,
    aggregate_results,
    print_summary,
    print_per_cell,
    save_results,
    save_per_cell_csv,
)


def main() -> None:
    all_rows: List[Dict] = []

    # 1) recorrer celdas
    for cell_id, series in iter_cells_by_file():
        # 2) split temporal
        s_train, s_val, s_test = split_series(series)

        # 3) evaluar persistencia para cada horizonte
        for H in H_LIST:
            eval_dict = evaluate_persistence_splits(s_train, s_val, s_test, H=H)
            all_rows.extend(rows_from_eval_dict(cell_id, H, eval_dict))

    # 4) agregar y reportar
    df_results = aggregate_results(all_rows)
    print_summary(df_results, include_y_mean=True)
    print_per_cell(df_results)  # opcional: detalle por celda

    # 5) guardar SOLO en Modeling/persistence/output/
    save_results(
        df_results,
        filename="persistence_results.csv",
        output_dir=PERSISTENCE_OUTPUT_DIR,
    )
    save_per_cell_csv(
        df_results,
        filename_prefix="persistence_cell_",
        output_dir=PERSISTENCE_OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
