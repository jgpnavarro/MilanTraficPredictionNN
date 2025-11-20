"""
Visualización de resultados del MLP por celda.

Para cada celda:
  - Carga la serie original.
  - Reproduce el pipeline de datos del entrenamiento:
      * split en train/val/test
      * normalización por máximo usando train
      * construcción de ventanas (H = NN_HORIZON, ventana = NN_INPUT_WINDOW)
  - Carga el modelo MLP entrenado desde disco (mlp_cell_<id>.keras).
  - Predice en train/val/test, deshace la normalización y organiza
    las predicciones como una serie temporal.
  - Dibuja en una sola figura:
      * Serie real (toda la serie)
      * Predicción MLP (solo donde hay valor)
      * Regiones sombreadas para los splits train / val / test
  - Guarda una imagen por celda en:
        Modeling/neural_networks/mlp/output/plots_all/cell_<id>_mlp.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import load_model

from Modeling.config import (
    NN_HORIZON,
    NN_INPUT_WINDOW,
    MLP_OUTPUT_DIR,
    MLP_MODELS_DIR,
)
from Modeling.data_access import iter_cells_by_file, split_series
from Modeling.scaling import scale_splits_by_train_max
from Modeling.targets import make_windowed_xy_for_horizon_splits


# Directorio donde guardaremos las figuras
PLOTS_DIR = MLP_OUTPUT_DIR / "plots_all"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _build_pred_series_for_cell(
    series: pd.Series,
    cell_id: int,
) -> Optional[Tuple[pd.DataFrame, Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]]]:
    """
    Calcula la serie de predicciones del MLP para una celda y los rangos de splits.

    Este método:
      - Hace split temporal en train/val/test.
      - Escala por máximo usando solo train.
      - Construye ventanas sobre las series escaladas.
      - Carga el modelo MLP entrenado para la celda.
      - Predice en train/val/test y deshace la escala.
      - Devuelve:
          * Un DataFrame con dos columnas:
              'y_real' : serie real completa (no escalada)
              'y_pred' : predicción MLP (NaN donde no hay ventana)
          * Un diccionario con los rangos temporales de cada split:
              {'train': (t0_train, t1_train), 'val': (...), 'test': (...)}
    Si no existe el modelo para esa celda o no hay datos suficientes,
    devuelve None.
    """
    model_path = MLP_MODELS_DIR / f"mlp_cell_{cell_id}.keras"
    if not model_path.exists():
        print(f"[mlp_plots] Modelo no encontrado para la celda {cell_id}: {model_path}")
        return None

    # Serie real completa, ordenada por tiempo
    series = series.sort_index()
    full_index = series.index
    y_real = series.astype(float)

    # 1) Split temporal
    s_train, s_val, s_test = split_series(series)

    # Construir rangos de tiempo por split para el sombreados
    split_ranges: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for name, s in (("train", s_train), ("val", s_val), ("test", s_test)):
        if not s.empty:
            split_ranges[name] = (s.index[0], s.index[-1])

    # 2) Escalado por máximo (usando solo train)
    s_train_s, s_val_s, s_test_s, scale_factor = scale_splits_by_train_max(
        s_train, s_val, s_test
    )

    # 3) Ventanas sobre las series escaladas
    splits_xy = make_windowed_xy_for_horizon_splits(
        s_train_s,
        s_val_s,
        s_test_s,
        H=NN_HORIZON,
        input_window=NN_INPUT_WINDOW,
    )

    X_train, y_train = splits_xy["train"]
    X_val, y_val = splits_xy["val"]
    X_test, y_test = splits_xy["test"]

    if len(X_train) == 0:
        print(f"[mlp_plots] Celda {cell_id}: no hay suficientes datos para ventanas.")
        return None

    # 4) Convertir a numpy para el modelo
    X_train_np = X_train.to_numpy(dtype="float32")
    X_val_np = X_val.to_numpy(dtype="float32")
    X_test_np = X_test.to_numpy(dtype="float32")

    # 5) Cargar el modelo entrenado
    model = load_model(model_path)

    # 6) Predicciones en escala normalizada
    y_pred_train_s = model.predict(X_train_np, verbose=0).flatten()
    y_pred_val_s = model.predict(X_val_np, verbose=0).flatten()
    y_pred_test_s = model.predict(X_test_np, verbose=0).flatten()

    # 7) Desescalar predicciones y objetivos
    y_train_true = y_train.to_numpy(dtype="float32") * scale_factor
    y_val_true = y_val.to_numpy(dtype="float32") * scale_factor
    y_test_true = y_test.to_numpy(dtype="float32") * scale_factor

    y_train_pred = y_pred_train_s * scale_factor
    y_val_pred = y_pred_val_s * scale_factor
    y_test_pred = y_pred_test_s * scale_factor

    # 8) Construir una serie completa de predicciones en el índice original
    #    Inicialmente todo NaN; luego rellenamos solo donde hay ventana válida.
    y_pred_full = pd.Series(
        data=np.nan,
        index=full_index,
        name="y_pred",
        dtype=float,
    )

    # Cada split tiene un índice temporal (el instante futuro t+H)
    # que ya está alineado con la serie original, así que podemos asignar.
    y_pred_train_series = pd.Series(
        data=y_train_pred,
        index=y_train.index,
        name="y_pred_train",
    )
    y_pred_val_series = pd.Series(
        data=y_val_pred,
        index=y_val.index,
        name="y_pred_val",
    )
    y_pred_test_series = pd.Series(
        data=y_test_pred,
        index=y_test.index,
        name="y_pred_test",
    )

    # Usamos update para combinar las predicciones en la serie completa
    y_pred_full.update(y_pred_train_series)
    y_pred_full.update(y_pred_val_series)
    y_pred_full.update(y_pred_test_series)

    # Devolvemos un DataFrame con real y predicho, y los rangos de splits
    df = pd.DataFrame(
        {
            "y_real": y_real,
            "y_pred": y_pred_full,
        },
        index=full_index,
    )

    return df, split_ranges


def plot_mlp_for_cell(cell_id: int, series: pd.Series) -> None:
    """
    Genera y guarda una figura para una celda con:

        - Serie real completa.
        - Predicción MLP (solo donde hay predicción disponible).
        - Regiones sombreadas para train / val / test.

    Guarda la figura como:
        Modeling/neural_networks/mlp/output/plots_all/cell_<id>_mlp.png
    """
    result = _build_pred_series_for_cell(series, cell_id)
    if result is None:
        return

    df, split_ranges = result

    fig, ax = plt.subplots(figsize=(14, 5))

    # Serie real
    ax.plot(
        df.index,
        df["y_real"],
        label="Real",
        linewidth=1.0,
    )

    # Predicción MLP (línea discontinua)
    ax.plot(
        df.index,
        df["y_pred"],
        label="Predicción MLP",
        linewidth=1.0,
        linestyle="--",
    )

    # Sombreado de los splits (train / val / test)
    colors = {
        "train": "green",
        "val": "orange",
        "test": "red",
    }

    for name, (t0, t1) in split_ranges.items():
        color = colors.get(name, "gray")
        ax.axvspan(t0, t1, alpha=0.05, color=color, label=f"{name} (fondo)")

    ax.set_title(f"Celda {cell_id} - MLP (H={NN_HORIZON}, ventana={NN_INPUT_WINDOW})")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("internet_total")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", linewidth=0.5)

    fig.tight_layout()

    out_path = PLOTS_DIR / f"cell_{cell_id}_mlp.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[mlp_plots] Figura guardada para celda {cell_id}: {out_path}")


def run_plots_for_all_cells() -> None:
    """
    Recorre todas las celdas y genera una figura por celda.
    """
    for cell_id, series in iter_cells_by_file():
        print(f"[mlp_plots] Generando figura para celda {cell_id}...")
        plot_mlp_for_cell(cell_id, series)


if __name__ == "__main__":
    run_plots_for_all_cells()
