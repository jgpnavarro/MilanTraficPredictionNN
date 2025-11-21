"""
Cálculo de ahorro energético teórico usando las predicciones de varios modelos.

Para cada celda y cada modelo seleccionado (persistence, moving_average, mlp, gru):
  - Se calculan P20 y P80 sobre train+val (tráfico real).
  - Se reconstruye la serie de tráfico real y predicho a 1h vista (H=NN_HORIZON).
  - En el tramo de test:
      * Se convierten tráfico real y predicho en niveles L/M/H.
      * Se aplica una política ACTIVO/AHORRO basada en niveles predichos:
            - L repetido K veces seguidas => AHORRO
            - M o H => ACTIVO
      * Se calcula el consumo en un escenario base (siempre ACTIVO)
        y en un escenario con política.
      * Se deriva el ahorro (%) y el riesgo (%) de haber estado en
        AHORRO cuando el nivel real era M o H.

Resultados por celda y modelo en:
    Modeling/energy/output/energy_results_all_models.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from Modeling.data_access import iter_cells_by_file, split_series
from Modeling.config import NN_HORIZON, NN_INPUT_WINDOW, PERCENTILE_LOW, PERCENTILE_HIGH, K_MIN_LOW, P_ACTIVE, P_SAVING, ENERGY_OUTPUT_DIR

# Reutilizamos las funciones de run_plots para reconstruir y_real/y_pred en MLP y GRU
from Modeling.neural_networks.mlp.run_plots import (
    _build_pred_series_for_cell as build_mlp_pred_series_for_cell,
)
from Modeling.neural_networks.gru.run_plots import (
    _build_pred_series_for_cell as build_gru_pred_series_for_cell,
)


# Fijamos H y W a lo configurado (por claridad)
H_FIXED = NN_HORIZON       # debería ser 6 (1 hora)
W_FIXED = NN_INPUT_WINDOW  # debería ser 12 (2 horas)

# -------------------------------------------------------------------
# Argumentos y selección de modelos
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parseo de argumentos de línea de comandos.

    Ejemplos:
        python -m Modeling.energy.run_energy --models all
        python -m Modeling.energy.run_energy --models mlp,gru
        python -m Modeling.energy.run_energy --models persistence,moving_average
    """
    parser = argparse.ArgumentParser(
        description="Evaluación de ahorro energético con varios modelos."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help=(
            "Modelos a usar: all, "
            "persistence, moving_average, mlp, gru "
            "o una lista separada por comas (por ejemplo: mlp,gru)."
        ),
    )
    return parser.parse_args()


def normalize_model_list(models_arg: str) -> List[str]:
    """
    A partir del string de entrada (por ejemplo 'mlp,gru' o 'all'),
    devuelve una lista de nombres de modelo válidos.
    """
    allowed = {"persistence", "moving_average", "mlp", "gru"}

    models_arg = models_arg.strip().lower()
    if models_arg == "all":
        return sorted(allowed)

    requested = [m.strip() for m in models_arg.split(",") if m.strip()]
    models = [m for m in requested if m in allowed]

    if not models:
        print("[energy] No se ha especificado ningún modelo válido. "
              "Se usarán todos por defecto.")
        return sorted(allowed)

    return sorted(set(models))


# -------------------------------------------------------------------
# Umbrales y niveles L/M/H
# -------------------------------------------------------------------

def _compute_thresholds_train_val(
    s_train: pd.Series,
    s_val: pd.Series,
) -> Tuple[float, float]:
    """
    Calcula los umbrales P20 y P80 usando los valores reales de train+val.

    P20 se usará como frontera de "baja demanda".
    P80 se usará como frontera de "alta demanda".
    """
    s_train_val = pd.concat([s_train, s_val]).dropna()

    if s_train_val.empty:
        # En un caso extremo, devolvemos umbrales triviales
        return 0.0, 0.0

    p20 = float(s_train_val.quantile(PERCENTILE_LOW / 100.0))
    p80 = float(s_train_val.quantile(PERCENTILE_HIGH / 100.0))
    return p20, p80


def _map_to_level(value: float, p20: float, p80: float) -> str:
    """
    Mapea un valor de tráfico a un nivel L/M/H usando los umbrales (p20, p80).

    - L (Low)    : value < p20
    - M (Medium) : p20 <= value <= p80
    - H (High)   : value > p80
    """
    if value < p20:
        return "L"
    if value > p80:
        return "H"
    return "M"


def _build_levels_and_states_for_test(
    df_test: pd.DataFrame,
    p20: float,
    p80: float,
    k_min_low: int,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    A partir de un DataFrame df_test con columnas:
        - y_real : tráfico real (test)
        - y_pred : tráfico predicho por el modelo (test)

    y de los umbrales p20, p80, y del parámetro K (k_min_low),
    calcula:

        level_real(t): L/M/H según y_real(t)
        level_pred(t): L/M/H según y_pred(t)
        state_pred(t): 'ACTIVE' o 'SAVING' según la política:

            - Si predicción es M o H → estado = ACTIVE
            - Si predicción es L durante al menos K pasos seguidos → estado = SAVORING
            - Si en algún instante predicción vuelve a M o H → estado = ACTIVE

    Devuelve:
        level_real, level_pred, state_pred  (todas Series con mismo índice que df_test)
    """
    # Nos aseguramos de trabajar sobre una copia ordenada por tiempo
    df_test = df_test.sort_index().copy()

    # Convertimos a niveles (L/M/H)
    level_real = df_test["y_real"].apply(lambda v: _map_to_level(v, p20, p80))
    level_pred = df_test["y_pred"].apply(lambda v: _map_to_level(v, p20, p80))

    # Ahora aplicamos la lógica de estados con memoria K
    states = []
    current_state = "ACTIVE"
    low_run = 0  # cuántos pasos seguidos llevamos con nivel predicho L

    for lvl in level_pred:
        if lvl == "L":
            low_run += 1
        else:
            low_run = 0
            current_state = "ACTIVE"  # si pasa a M/H, volvemos a ACTIVO

        if current_state == "ACTIVE" and low_run >= k_min_low:
            current_state = "SAVING"

        states.append(current_state)

    state_pred = pd.Series(states, index=df_test.index, name="state_pred")

    return level_real, level_pred, state_pred


# -------------------------------------------------------------------
# Energía y riesgo
# -------------------------------------------------------------------

def _compute_energy_and_risk_for_cell(
    level_real: pd.Series,
    state_pred: pd.Series,
) -> Dict[str, float]:
    """
    Dado:
        - level_real(t): niveles reales L/M/H en test.
        - state_pred(t): estado de la celda según la política (ACTIVE/SAVING).

    Calcula:
        - consumo base (siempre ACTIVO),
        - consumo con política,
        - ahorro (%),
        - porcentaje de tiempo en ahorro,
        - riesgo (%): veces que se estuvo en ahorro cuando el nivel real no era bajo.

    Devuelve un diccionario con estos valores.
    """
    # Aseguramos alineamiento por índice
    level_real = level_real.astype(str)
    state_pred = state_pred.astype(str)
    idx_common = level_real.index.intersection(state_pred.index)

    if len(idx_common) == 0:
        return {
            "n_test": 0,
            "E_base": float("nan"),
            "E_policy": float("nan"),
            "saving_percent": float("nan"),
            "time_saving_percent": float("nan"),
            "N_ahorro": 0,
            "N_conflictos": 0,
            "risk_percent": float("nan"),
        }

    level_real = level_real.loc[idx_common]
    state_pred = state_pred.loc[idx_common]

    n_test = len(idx_common)

    # Escenario base: siempre ACTIVO
    E_base = n_test * P_ACTIVE

    # Escenario con política
    power = np.where(state_pred == "ACTIVE", P_ACTIVE, P_SAVING)
    E_policy = float(power.sum())

    # Ahorro relativo (%)
    saving_percent = 100.0 * (E_base - E_policy) / E_base if E_base > 0 else float("nan")

    # Porcentaje de tiempo en modo ahorro
    N_ahorro = int(np.sum(state_pred == "SAVING"))
    time_saving_percent = 100.0 * N_ahorro / n_test if n_test > 0 else float("nan")

    # Riesgo: intervalos en ahorro donde el nivel real NO es bajo (M o H)
    conflictos_mask = (state_pred == "SAVING") & (level_real != "L")
    N_conflictos = int(np.sum(conflictos_mask))

    if N_ahorro > 0:
        risk_percent = 100.0 * N_conflictos / N_ahorro
    else:
        risk_percent = float("nan")

    return {
        "n_test": n_test,
        "E_base": E_base,
        "E_policy": E_policy,
        "saving_percent": saving_percent,
        "time_saving_percent": time_saving_percent,
        "N_ahorro": N_ahorro,
        "N_conflictos": N_conflictos,
        "risk_percent": risk_percent,
    }


# -------------------------------------------------------------------
# Construcción de y_real / y_pred según modelo
# -------------------------------------------------------------------

def _build_pred_series_for_model(
    model_name: str,
    cell_id: int,
    series: pd.Series,
) -> pd.DataFrame | None:
    """
    Construye un DataFrame con columnas:

        - y_real : tráfico real (serie completa)
        - y_pred : tráfico predicho por el modelo (alineado en el tiempo)

    para el modelo indicado:

        - 'persistence'     : ŷ(t) = y(t - H_FIXED)
        - 'moving_average'  : ŷ(t) = media de ventana W_FIXED en t-H_FIXED
        - 'mlp'             : se usa run_plots.mlp._build_pred_series_for_cell
        - 'gru'             : se usa run_plots.gru._build_pred_series_for_cell

    Devuelve None si no se puede construir (por ejemplo, modelo no entrenado).
    """
    model_name = model_name.lower()
    series = series.sort_index()
    full_index = series.index
    y_real = series.astype(float)

    if model_name == "persistence":
        # Predicción por persistencia simple (shift hacia adelante en el tiempo)
        y_pred = series.shift(H_FIXED).astype(float)
        df = pd.DataFrame(
            {
                "y_real": y_real,
                "y_pred": y_pred,
            },
            index=full_index,
        )
        return df

    if model_name == "moving_average":
        # Predicción por media móvil con ventana W_FIXED y horizonte H_FIXED
        rolling_mean = series.rolling(window=W_FIXED, min_periods=W_FIXED).mean()
        y_pred = rolling_mean.shift(H_FIXED).astype(float)
        df = pd.DataFrame(
            {
                "y_real": y_real,
                "y_pred": y_pred,
            },
            index=full_index,
        )
        return df

    if model_name == "mlp":
        try:
            result = build_mlp_pred_series_for_cell(series, cell_id)
        except Exception as e:
            print(f"[energy] Error al reconstruir predicción MLP para celda {cell_id}: {e}")
            return None

        if result is None:
            return None

        df_mlp, _ = result
        # Aseguramos que tiene las columnas esperadas
        if not {"y_real", "y_pred"}.issubset(df_mlp.columns):
            print(f"[energy] MLP: df_mlp para celda {cell_id} no tiene columnas y_real/y_pred.")
            return None

        return df_mlp[["y_real", "y_pred"]]

    if model_name == "gru":
        try:
            result = build_gru_pred_series_for_cell(series, cell_id)
        except Exception as e:
            print(f"[energy] Error al reconstruir predicción GRU para celda {cell_id}: {e}")
            return None

        if result is None:
            return None

        df_gru, _ = result
        if not {"y_real", "y_pred"}.issubset(df_gru.columns):
            print(f"[energy] GRU: df_gru para celda {cell_id} no tiene columnas y_real/y_pred.")
            return None

        return df_gru[["y_real", "y_pred"]]

    print(f"[energy] Modelo desconocido: {model_name}")
    return None


# -------------------------------------------------------------------
# Flujo principal
# -------------------------------------------------------------------

def run_energy_for_models(models: List[str]) -> None:
    """
    Ejecuta la evaluación de ahorro energético para todos los modelos
    indicados en la lista `models`.

    Para cada celda:
      - Calcula P20 y P80 con train+val (tráfico real).
      - Para cada modelo:
          * Construye y_real / y_pred (serie completa).
          * Se queda con el tramo de test.
          * Convierte a niveles L/M/H (real y predicho).
          * Aplica la política de ACTIVO/AHORRO (K=K_MIN_LOW).
          * Calcula consumo base, consumo con política, ahorro y riesgo.
          * Añade una fila al DataFrame de resultados.

    Resultados en:
        Modeling/energy/output/energy_results_all_models.csv
    """
    rows: List[Dict] = []

    for cell_id, series in iter_cells_by_file():
        print(f"[energy] Procesando celda {cell_id}...")

        # 1) Split temporal
        s_train, s_val, s_test = split_series(series)

        if s_test.empty:
            print(f"[energy] Celda {cell_id}: split de test vacío, se omite.")
            continue

        # 2) Umbrales P20 y P80 usando train+val
        p20, p80 = _compute_thresholds_train_val(s_train, s_val)

        for model_name in models:
            print(f"    [energy] Modelo {model_name}...")

            # 3) Reconstruir serie real y predicha para este modelo
            df_pred_full = _build_pred_series_for_model(model_name, cell_id, series)
            if df_pred_full is None:
                print(f"    [energy] Celda {cell_id}, modelo {model_name}: "
                      "no se pudo construir y_pred, se omite.")
                continue

            # 4) Quedarnos solo con el tramo de test y eliminar NaNs en la predicción
            df_test = df_pred_full.loc[s_test.index].copy()
            df_test = df_test.dropna(subset=["y_real", "y_pred"])

            if df_test.empty:
                print(f"    [energy] Celda {cell_id}, modelo {model_name}: "
                      "sin datos válidos en test tras filtrar NaNs.")
                continue

            # 5) Niveles y estados en test
            level_real, level_pred, state_pred = _build_levels_and_states_for_test(
                df_test,
                p20=p20,
                p80=p80,
                k_min_low=K_MIN_LOW,
            )

            # 6) Energía y riesgo
            energy_risk = _compute_energy_and_risk_for_cell(level_real, state_pred)

            # 7) Construir fila de resultados para esta celda y modelo
            row = {
                "cell_id": cell_id,
                "model": model_name,
                "H": H_FIXED,
                "window": W_FIXED,
                "P20": p20,
                "P80": p80,
                "K_min_low": K_MIN_LOW,
                "P_active": P_ACTIVE,
                "P_saving": P_SAVING,
            }
            row.update(energy_risk)
            rows.append(row)

    if not rows:
        print("[energy] No se han generado resultados para ninguna celda.")
        return

    df_results = pd.DataFrame(rows)

    out_path = ENERGY_OUTPUT_DIR / "energy_results_all_models.csv"
    df_results.to_csv(out_path, index=False)

    print(f"[energy] Resultados de energía guardados en: {out_path}")


def main() -> None:
    """
    Entrypoint del módulo:

      1) Lee lista de modelos desde --models.
      2) Normaliza la lista (all -> todos).
      3) Ejecuta run_energy_for_models(models).
    """
    args = parse_args()
    models = normalize_model_list(args.models)

    print(f"[energy] Modelos seleccionados: {', '.join(models)}")
    print(f"[energy] H_FIXED={H_FIXED}, W_FIXED={W_FIXED}")
    run_energy_for_models(models)


if __name__ == "__main__":
    main()
