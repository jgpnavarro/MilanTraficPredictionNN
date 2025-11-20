"""
Orquestador para ejecutar varios modelos y comparar sus predicciones.

Este script permite:
  - Elegir qué modelos queremos ejecutar:
        * persistence
        * moving_average
        * mlp
        * gru
        * all  (para ejecutar todos)
  - Lanzar los scripts de evaluación/entrenamiento de cada modelo (ya existentes).
  - Para cada celda:
        * reconstruir las predicciones de cada modelo para H=6, W=12 (cuando aplica),
          sobre la serie completa.
        * dibujar una figura con:
              - Serie real.
              - Predicción de cada modelo seleccionado.
              - Fondo sombreado para train / val / test.
  - Construir una tabla con las métricas de test (MAPE, wMAPE, sMAPE) por celda
    y por modelo, para H=6 (y W=12 para la media móvil).
  - Guardar todo en una carpeta nueva en la raíz del proyecto: ./total_output
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importamos las rutas de salida de cada modelo desde la configuración
from Modeling.config import (
    PERSISTENCE_OUTPUT_DIR,
    MOVING_AVG_OUTPUT_DIR,
    MLP_OUTPUT_DIR,
    GRU_OUTPUT_DIR,
    NN_HORIZON,
    NN_INPUT_WINDOW,
)

# Funciones para lanzar cada modelo
from Modeling.baselines.persistence.run_persistence import main as run_persistence_main
from Modeling.baselines.moving_average.run_moving_average import main as run_moving_average_main
from Modeling.neural_networks.mlp.run_mlp import run_mlp_for_all_cells
from Modeling.neural_networks.gru.run_gru import run_gru_for_all_cells

# Funciones de acceso a datos
from Modeling.data_access import iter_cells_by_file, split_series

# Para reconstruir predicciones de MLP y GRU reutilizamos las funciones
# internas de sus módulos run_plots (ya definidas en el proyecto).
from Modeling.neural_networks.mlp.run_plots import (
    _build_pred_series_for_cell as build_mlp_pred_series_for_cell,
)
from Modeling.neural_networks.gru.run_plots import (
    _build_pred_series_for_cell as build_gru_pred_series_for_cell,
)


# Carpeta de salida conjunta en la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent
TOTAL_OUTPUT_DIR = BASE_DIR / "total_output"
PLOTS_DIR = TOTAL_OUTPUT_DIR / "plots_cells"
TOTAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# Fijamos explícitamente los valores que queremos usar aquí
H_FIXED = 6
W_FIXED = 12


def parse_args() -> argparse.Namespace:
    """
    Parseo sencillo de argumentos de línea de comandos.

    Ejemplos de uso:
        python run_orchestrator.py --models all
        python run_orchestrator.py --models mlp,gru
        python run_orchestrator.py --models persistence,moving_average
    """
    parser = argparse.ArgumentParser(
        description="Orquestador para comparar varios modelos (H=6, W=12)."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help=(
            "Modelos a ejecutar: all, "
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
        print("[orchestrator] No se ha especificado ningún modelo válido. "
              "Se usarán todos por defecto.")
        return sorted(allowed)

    return sorted(set(models))


def run_selected_models(models: List[str]) -> None:
    """
    Lanza los scripts de evaluación/entrenamiento de los modelos seleccionados.

    Cada script:
      - Recorre las celdas.
      - Calcula métricas.
      - (En el caso de MLP/GRU) entrena y guarda modelos en disco.
    """
    if "persistence" in models:
        print("\n[orchestrator] Ejecutando modelo de persistencia...")
        run_persistence_main()

    if "moving_average" in models:
        print("\n[orchestrator] Ejecutando modelo de media móvil...")
        run_moving_average_main()

    if "mlp" in models:
        print("\n[orchestrator] Ejecutando modelo MLP...")
        run_mlp_for_all_cells()

    if "gru" in models:
        print("\n[orchestrator] Ejecutando modelo GRU...")
        run_gru_for_all_cells()


def _get_split_ranges(series: pd.Series) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Calcula los rangos temporales de train/val/test para una serie.

    Devuelve un diccionario:
        {
            "train": (t0_train, t1_train),
            "val":   (t0_val,   t1_val),
            "test":  (t0_test,  t1_test),
        }
    """
    s_train, s_val, s_test = split_series(series)

    ranges: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    if not s_train.empty:
        ranges["train"] = (s_train.index[0], s_train.index[-1])
    if not s_val.empty:
        ranges["val"] = (s_val.index[0], s_val.index[-1])
    if not s_test.empty:
        ranges["test"] = (s_test.index[0], s_test.index[-1])

    return ranges


def _predict_persistence(series: pd.Series, horizon: int) -> pd.Series:
    """
    Predicción de persistencia simple para un horizonte dado:

        ŷ(t+horizon) = y(t)

    Implementado como un desplazamiento de la serie original.
    """
    # shift(horizon) mueve los valores hacia "adelante" en el índice:
    #   y_pred[t_i] = y[t_{i-horizon}]
    return series.shift(horizon)


def _predict_moving_average(series: pd.Series, horizon: int, window: int) -> pd.Series:
    """
    Predicción de media móvil:

        - Calcula la media de los últimos 'window' puntos en cada instante t.
        - Usa esa media como predicción 'horizon' pasos hacia delante.

    Es equivalente a:
        rolling_mean = media móvil
        y_pred = rolling_mean.shift(horizon)
    """
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    return rolling_mean.shift(horizon)


def build_predictions_for_cell(
    cell_id: int,
    series: pd.Series,
    models: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]]:
    """
    Construye un DataFrame con la serie real y las predicciones de los modelos
    seleccionados para una celda concreta.

    Columnas del DataFrame (según modelos seleccionados):
        - 'y_real'
        - 'persistence'      (si se ha pedido este modelo)
        - 'moving_average'   (si se ha pedido este modelo)
        - 'mlp'              (si se ha pedido este modelo)
        - 'gru'              (si se ha pedido este modelo)
    """
    series = series.sort_index()
    full_index = series.index

    df = pd.DataFrame(
        {"y_real": series.astype(float)},
        index=full_index,
    )

    # Rangos de train/val/test para sombrear en la gráfica
    split_ranges = _get_split_ranges(series)

    # Modelos basados en reglas (no necesitan modelos guardados)
    if "persistence" in models:
        df["persistence"] = _predict_persistence(series, horizon=H_FIXED)

    if "moving_average" in models:
        df["moving_average"] = _predict_moving_average(
            series,
            horizon=H_FIXED,
            window=W_FIXED,
        )

    # MLP: reutilizamos la función de run_plots para reconstruir la serie de predicción
    if "mlp" in models:
        try:
            result_mlp = build_mlp_pred_series_for_cell(series, cell_id)
        except Exception as e:
            print(f"[orchestrator] Aviso: error al reconstruir predicción MLP para celda {cell_id}: {e}")
            result_mlp = None

        if result_mlp is not None:
            df_mlp, _ = result_mlp
            # df_mlp tiene columnas 'y_real' y 'y_pred' con el mismo índice
            df["mlp"] = df_mlp["y_pred"]

    # GRU: igual, reutilizando su run_plots
    if "gru" in models:
        try:
            result_gru = build_gru_pred_series_for_cell(series, cell_id)
        except Exception as e:
            print(f"[orchestrator] Aviso: error al reconstruir predicción GRU para celda {cell_id}: {e}")
            result_gru = None

        if result_gru is not None:
            df_gru, _ = result_gru
            df["gru"] = df_gru["y_pred"]

    return df, split_ranges


def plot_combined_for_cell(
    cell_id: int,
    df: pd.DataFrame,
    split_ranges: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
    models: List[str],
) -> None:
    """
    Dibuja y guarda la figura combinada para una celda.

    Contiene:
        - Serie real.
        - Predicción de cada modelo seleccionado.
        - Fondo sombreado para train / val / test.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    # Serie real
    ax.plot(
        df.index,
        df["y_real"],
        label="Real",
        linewidth=1.0,
    )

    # Predicciones de los modelos seleccionados (si la columna existe)
    model_label_map = {
        "persistence": "Persistencia",
        "moving_average": "Media móvil",
        "mlp": "MLP",
        "gru": "GRU",
    }

    for m in models:
        col_name = m
        if col_name in df.columns:
            ax.plot(
                df.index,
                df[col_name],
                label=model_label_map.get(m, m),
                linewidth=1.0,
                linestyle="--",
            )

    # Sombreado de los splits (train / val / test), con etiqueta para la leyenda
    colors = {
        "train": "green",
        "val": "orange",
        "test": "red",
    }

    for name, (t0, t1) in split_ranges.items():
        color = colors.get(name, "gray")
        ax.axvspan(t0, t1, alpha=0.05, color=color, label=f"{name} (fondo)")

    ax.set_title(f"Celda {cell_id} - Modelos combinados (H={H_FIXED}, ventana={W_FIXED})")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("internet_total")
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True, linestyle=":", linewidth=0.5)

    fig.tight_layout()

    out_path = PLOTS_DIR / f"cell_{cell_id}_combined.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[orchestrator] Figura combinada guardada para celda {cell_id}: {out_path}")


def load_metrics_for_model(model_name: str) -> pd.DataFrame:
    """
    Carga el CSV de resultados del modelo indicado y filtra:

        - split == 'test'
        - horizon == H_FIXED
        - window == W_FIXED (solo en el caso de moving_average)

    Devuelve un DataFrame con:
        - cell_id
        - model
        - MAPE
        - wMAPE
        - sMAPE
    """
    if model_name == "persistence":
        csv_path = PERSISTENCE_OUTPUT_DIR / "persistence_results.csv"
    elif model_name == "moving_average":
        csv_path = MOVING_AVG_OUTPUT_DIR / "moving_avg_results.csv"
    elif model_name == "mlp":
        csv_path = MLP_OUTPUT_DIR / "mlp_results.csv"
    elif model_name == "gru":
        csv_path = GRU_OUTPUT_DIR / "gru_results.csv"
    else:
        raise ValueError(f"Nombre de modelo desconocido: {model_name}")

    if not csv_path.exists():
        print(f"[orchestrator] Aviso: no se encontró {csv_path}. "
              f"Se ignora el modelo {model_name}.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Comprobaciones básicas de columnas
    required_cols = {"cell_id", "split", "horizon", "MAPE", "wMAPE", "sMAPE"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[orchestrator] Aviso: el archivo {csv_path} no tiene todas las columnas "
              f"necesarias ({missing}). Se ignora este modelo para la tabla de métricas.")
        return pd.DataFrame()

    # Filtro por horizonte y split test
    df = df[(df["horizon"] == H_FIXED) & (df["split"] == "test")].copy()

    # Para moving_average, filtramos también por ventana W_FIXED
    if model_name == "moving_average":
        if "window" not in df.columns:
            print(f"[orchestrator] Aviso: el archivo {csv_path} no tiene columna 'window'.")
            return pd.DataFrame()
        df = df[df["window"] == W_FIXED].copy()

    if df.empty:
        print(f"[orchestrator] Aviso: el modelo {model_name} no tiene filas de test "
              f"para H={H_FIXED} (y W={W_FIXED} si aplica).")
        return pd.DataFrame()

    # Nos quedamos solo con la información relevante
    df_out = df[["cell_id", "MAPE", "wMAPE", "sMAPE"]].copy()
    df_out["model"] = model_name

    return df_out


def build_metrics_table(models: List[str]) -> pd.DataFrame:
    """
    Construye la tabla con MAPE, wMAPE y sMAPE por celda y modelo
    (solo en el split de test, H=6, W=12 para media móvil).
    """
    all_dfs: List[pd.DataFrame] = []

    for m in models:
        df_m = load_metrics_for_model(m)
        if not df_m.empty:
            all_dfs.append(df_m)

    if not all_dfs:
        print("[orchestrator] No se pudo construir la tabla de métricas.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    # Ordenamos por celda y modelo para que sea más legible
    combined = combined.sort_values(["cell_id", "model"], kind="stable")
    return combined


def save_metrics_table(df_metrics: pd.DataFrame) -> Path:
    """
    Guarda la tabla de métricas en total_output/metrics_by_cell_H6_W12_test.csv
    y devuelve la ruta del fichero.
    """
    out_path = TOTAL_OUTPUT_DIR / "metrics_by_cell_H6_W12_test.csv"
    df_metrics.to_csv(out_path, index=False)
    print(f"[orchestrator] Tabla de métricas guardada en: {out_path}")
    return out_path


def main() -> None:
    """
    Flujo principal del orquestador:
      1) Parsear argumentos.
      2) Decidir qué modelos ejecutar.
      3) Lanzar los scripts de evaluación/entrenamiento de esos modelos.
      4) Para cada celda, reconstruir predicciones y dibujar figura combinada.
      5) Construir y guardar la tabla de métricas por celda y modelo.
    """
    args = parse_args()
    models = normalize_model_list(args.models)

    print(f"[orchestrator] Modelos seleccionados: {', '.join(models)}")

    # 1) Ejecutar los modelos seleccionados
    run_selected_models(models)

    # 2) Generar figuras combinadas por celda
    for cell_id, series in iter_cells_by_file():
        print(f"[orchestrator] Generando figura combinada para celda {cell_id}...")
        df_pred, split_ranges = build_predictions_for_cell(cell_id, series, models)
        plot_combined_for_cell(cell_id, df_pred, split_ranges, models)

    # 3) Construir y guardar la tabla de métricas (MAPE, wMAPE, sMAPE) por celda
    df_metrics = build_metrics_table(models)
    if not df_metrics.empty:
        save_metrics_table(df_metrics)


if __name__ == "__main__":
    main()
