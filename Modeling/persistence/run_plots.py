"""
Generador de gráficas 'ALL-IN-ONE' para persistencia:
una imagen por celda y horizonte con toda la serie y splits sombreados.
"""

from __future__ import annotations

from typing import Dict, Tuple
from Modeling.config import H_LIST, PERSISTENCE_OUTPUT_DIR
from Modeling.data_access import iter_cells_by_file, split_series
import pandas as pd

from Modeling.persistence.viz import (
    plot_overlay_full_series_with_splits,
)


def _aligned_true_and_pred(series: pd.Series, H: int) -> Tuple[pd.Series, pd.Series]:
    """
    Construye y_true e y_pred alineados para la serie completa y horizonte H.
    """
    X = series
    y = series.shift(-H)
    mask = y.notna()
    y_true = y[mask].astype(float)
    y_pred = X[mask].astype(float)
    y_pred.index = y_true.index  # alineación temporal
    return y_true, y_pred


def main() -> None:
    plots_root = PERSISTENCE_OUTPUT_DIR / "plots_all"

    for cell_id, series in iter_cells_by_file():
        # Calcular rangos temporales exactos de cada split para sombrear
        s_train, s_val, s_test = split_series(series)
        split_ranges = {
            "train": (s_train.index.min(), s_train.index.max()),
            "val":   (s_val.index.min(),   s_val.index.max()),
            "test":  (s_test.index.min(),  s_test.index.max()),
        }

        for H in H_LIST:
            y_true, y_pred = _aligned_true_and_pred(series, H)

            if len(y_true) == 0:
                continue

            start = y_true.index.min()
            end = y_true.index.max()
            title = f"Celda {cell_id} — H={H} ( {start} → {end} ) — Splits sombreados"

            out_dir = plots_root / f"{int(cell_id)}"
            out_path = out_dir / f"H{H}_ALL_overlay.png"

            plot_overlay_full_series_with_splits(
                y_true=y_true,
                y_pred=y_pred,
                split_ranges=split_ranges,
                title=title,
                out_path=out_path,
            )
            print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
