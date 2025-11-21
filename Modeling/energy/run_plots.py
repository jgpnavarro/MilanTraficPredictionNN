"""
Visualización del consumo acumulado (base vs política) por celda y modelo.

Para cada celda:
  - Usamos la misma lógica que en run_energy.py:
      * umbrales P20 / P80 sobre train+val (tráfico real),
      * reconstrucción de y_real / y_pred por modelo,
      * tramo de test,
      * política ACTIVO / AHORRO basada en niveles L/M/H predichos.
  - Construimos:
      * consumo acumulado base (siempre ACTIVO) → recta ascendente,
      * consumo acumulado con política para cada modelo.
  - Dibujamos una figura con:
      * 1 línea base,
      * 4 líneas (persistencia, media móvil, MLP, GRU).

Las figuras se guardan en:
    Modeling/energy/output/plots_cumulative/cell_<id>_energy.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Modeling.data_access import iter_cells_by_file, split_series

# Importamos constantes y funciones auxiliares de run_energy
from Modeling.energy.run_energy import (
    ENERGY_OUTPUT_DIR,
    H_FIXED,
    W_FIXED,
    P_ACTIVE,
    P_SAVING,
    K_MIN_LOW,
    _compute_thresholds_train_val,
    _build_levels_and_states_for_test,
    _build_pred_series_for_model,
)

# Modelos que queremos visualizar
MODELS_FOR_PLOT = ["persistence", "moving_average", "mlp", "gru"]

# Directorio donde guardaremos las figuras
PLOTS_DIR = ENERGY_OUTPUT_DIR / "plots_cumulative"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Nombres “bonitos” para la leyenda
MODEL_LABELS = {
    "persistence": "Persistencia",
    "moving_average": "Media móvil",
    "mlp": "MLP",
    "gru": "GRU",
}


def _build_common_test_dfs_for_cell(
    cell_id: int,
    series: pd.Series,
    models: List[str],
) -> Dict[str, pd.DataFrame] | None:
    """
    Para una celda:

      1) Hace el split en train/val/test.
      2) Calcula P20 y P80 en train+val.
      3) Para cada modelo:
         - reconstruye y_real / y_pred (serie completa),
         - extrae el tramo de test,
         - elimina filas con NaN en y_real o y_pred.
      4) Calcula la intersección de timestamps de test entre todos los modelos
         que tengan datos válidos.
      5) Devuelve:
         - un diccionario model_name -> df_test_restringido
           (con el mismo índice temporal para todos),
         - los umbrales p20, p80.

    Si no hay intersección no vacía entre los modelos, devuelve None.
    """
    # 1) Split temporal
    s_train, s_val, s_test = split_series(series)

    if s_test.empty:
        print(f"[energy_plots] Celda {cell_id}: split de test vacío.")
        return None

    # 2) Umbrales P20 y P80 usando train+val
    p20, p80 = _compute_thresholds_train_val(s_train, s_val)

    # 3) Reconstruir df_test por modelo
    model_to_df_test: Dict[str, pd.DataFrame] = {}

    for model_name in models:
        df_pred_full = _build_pred_series_for_model(model_name, cell_id, series)
        if df_pred_full is None:
            print(
                f"[energy_plots] Celda {cell_id}, modelo {model_name}: "
                "no se pudo reconstruir y_pred."
            )
            continue

        # Tomamos sólo el tramo de test y eliminamos NaNs
        df_test = df_pred_full.loc[s_test.index].copy()
        df_test = df_test.dropna(subset=["y_real", "y_pred"])

        if df_test.empty:
            print(
                f"[energy_plots] Celda {cell_id}, modelo {model_name}: "
                "sin datos válidos en test tras filtrar NaNs."
            )
            continue

        model_to_df_test[model_name] = df_test

    if not model_to_df_test:
        print(f"[energy_plots] Celda {cell_id}: ningún modelo tiene datos válidos en test.")
        return None

    # 4) Intersección de índices de test entre modelos
    common_index = None
    for df in model_to_df_test.values():
        idx = df.index
        common_index = idx if common_index is None else common_index.intersection(idx)

    if common_index is None or len(common_index) == 0:
        print(f"[energy_plots] Celda {cell_id}: no hay índice común entre modelos.")
        return None

    common_index = common_index.sort_values()

    # Restringimos todos los df_test al índice común
    for model_name in list(model_to_df_test.keys()):
        df = model_to_df_test[model_name]
        df = df.loc[common_index].sort_index()
        model_to_df_test[model_name] = df

    # Guardamos también los umbrales
    model_to_df_test["_thresholds"] = pd.DataFrame({"P20": [p20], "P80": [p80]})
    return model_to_df_test


def _compute_cumulative_energy_for_cell(
    model_to_df_test: Dict[str, pd.DataFrame],
) -> Dict[str, np.ndarray]:
    """
    A partir del diccionario model_name -> df_test (con índice temporal común)
    y los umbrales P20/P80 guardados en "_thresholds", calcula:

      - "time_index"  : np.array de timestamps (ordenados).
      - "cum_base"    : consumo acumulado base (siempre ACTIVO).
      - "cum_<modelo>": consumo acumulado con política para cada modelo.

    Devuelve un diccionario con estos arrays.
    """
    # Recuperamos P20/P80
    thresholds_df = model_to_df_test["_thresholds"]
    p20 = float(thresholds_df["P20"].iloc[0])
    p80 = float(thresholds_df["P80"].iloc[0])

    # Eliminamos la entrada especial para no interferir
    model_to_df_test = {
        k: v for k, v in model_to_df_test.items() if k != "_thresholds"
    }

    # Todos los df_test comparten índice
    any_df = next(iter(model_to_df_test.values()))
    time_index = any_df.index
    n = len(time_index)

    # Consumo base acumulado: recta ascendente
    cum_base = np.cumsum(np.full(shape=n, fill_value=P_ACTIVE, dtype=float))

    results: Dict[str, np.ndarray] = {
        "time_index": time_index.to_numpy(),
        "cum_base": cum_base,
    }

    # Para cada modelo, calculamos estados y consumo acumulado
    for model_name, df_test in model_to_df_test.items():
        # Niveles y estados
        level_real, level_pred, state_pred = _build_levels_and_states_for_test(
            df_test,
            p20=p20,
            p80=p80,
            k_min_low=K_MIN_LOW,
        )

        # Potencia en cada intervalo
        power = np.where(state_pred == "ACTIVE", P_ACTIVE, P_SAVING)
        cum_policy = np.cumsum(power)

        results[f"cum_{model_name}"] = cum_policy

    return results


def plot_energy_for_cell(
    cell_id: int,
    series: pd.Series,
    models: List[str],
) -> None:
    """
    Genera y guarda una figura de consumo acumulado para una celda.

    En la figura:
      - Línea base: consumo acumulado "siempre activo".
      - Una línea por modelo: consumo acumulado con política de ahorro.
    """
    common_dfs = _build_common_test_dfs_for_cell(cell_id, series, models)
    if common_dfs is None:
        return

    energy_dict = _compute_cumulative_energy_for_cell(common_dfs)

    time_index = energy_dict["time_index"]
    cum_base = energy_dict["cum_base"]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Línea base (siempre activo)
    ax.plot(
        time_index,
        cum_base,
        label="Base (siempre ACTIVO)",
        linewidth=2.0,
    )

    # Líneas por modelo
    for model_name in models:
        key = f"cum_{model_name}"
        if key not in energy_dict:
            continue

        label = MODEL_LABELS.get(model_name, model_name)
        ax.plot(
            time_index,
            energy_dict[key],
            label=f"{label}",
            linewidth=1.2,
            linestyle="--",
        )

    ax.set_title(
        f"Celda {cell_id} - Consumo acumulado en test "
        f"(H={H_FIXED}, W={W_FIXED}, K={K_MIN_LOW})"
    )
    ax.set_xlabel("Tiempo (test)")
    ax.set_ylabel("Consumo acumulado (unidades relativas)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", linewidth=0.5)

    fig.tight_layout()

    out_path = PLOTS_DIR / f"cell_{cell_id}_energy.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[energy_plots] Figura guardada para celda {cell_id}: {out_path}")


def run_plots_for_all_cells() -> None:
    """
    Recorre todas las celdas y genera una figura por celda.
    """
    for cell_id, series in iter_cells_by_file():
        print(f"[energy_plots] Generando figura para celda {cell_id}...")
        plot_energy_for_cell(cell_id, series, MODELS_FOR_PLOT)


if __name__ == "__main__":
    run_plots_for_all_cells()
