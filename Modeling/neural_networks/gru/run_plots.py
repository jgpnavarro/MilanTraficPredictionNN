"""
Visualización de resultados de la GRU por celda.

Para cada celda:
  - Carga la serie original.
  - Reproduce (en versión simplificada) el pipeline de datos del entrenamiento:
      * split en train/val/test
      * normalización por máximo usando train
      * añadir features de calendario (día de la semana, hora, festivos, periodos especiales)
      * construcción de ventanas (H = NN_HORIZON, ventana = NN_INPUT_WINDOW)
  - Carga el modelo GRU entrenado desde disco (gru_cell_<id>.keras).
  - Predice en train/val/test, deshace la normalización y organiza
    las predicciones como una serie temporal.
  - Dibuja en una sola figura:
      * Serie real (toda la serie)
      * Predicción GRU (solo donde hay valor)
      * Regiones sombreadas para los splits train / val / test
  - Guarda una imagen por celda en:
        Modeling/neural_networks/gru/output/plots_all/cell_<id>_gru.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import load_model

from Modeling.config import (
    NN_HORIZON,
    NN_INPUT_WINDOW,
    GRU_OUTPUT_DIR,
    GRU_MODELS_DIR,
)
from Modeling.data_access import iter_cells_by_file, split_series
from Modeling.scaling import scale_splits_by_train_max
from Modeling.features_calendar import add_calendar_features


# Directorio donde guardaremos las figuras
PLOTS_DIR = GRU_OUTPUT_DIR / "plots_all"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _build_windows_for_split(
    df_split: pd.DataFrame,
    feature_cols: List[str],
    window: int,
    horizon: int,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Construye las ventanas X para un split concreto (train/val/test) y devuelve
    también el índice temporal de los objetivos futuros.

    Para cada posición t dentro del split se toma:
        - X_t: los 'window' pasos anteriores incluyendo t
               -> df_split[feature_cols].iloc[t-window+1 : t+1]
        - El objetivo estaría en t + horizon, y usamos su timestamp como índice.

    Devuelve:
        X : array de forma (n_muestras, window, n_features)
        idx_targets : índice de timestamps (DatetimeIndex) de longitud n_muestras,
                      que indica en qué instantes de la serie se están
                      realizando las predicciones.
    """
    # Aseguramos que el índice es temporal y está ordenado
    df_split = df_split.sort_index()
    values_features = df_split[feature_cols].to_numpy(dtype="float32")
    time_index = df_split.index.to_numpy()

    n = len(df_split)
    X_list: List[np.ndarray] = []
    idx_list: List[pd.Timestamp] = []

    last_input_idx = n - horizon - 1
    for t in range(window - 1, last_input_idx + 1):
        start = t - window + 1
        end = t + 1  # end exclusivo

        x_window = values_features[start:end, :]
        target_time = pd.Timestamp(time_index[t + horizon])

        X_list.append(x_window)
        idx_list.append(target_time)

    if not X_list:
        return (
            np.empty((0, window, len(feature_cols)), dtype="float32"),
            pd.DatetimeIndex([], name=df_split.index.name),
        )

    X = np.stack(X_list, axis=0)
    idx_targets = pd.DatetimeIndex(idx_list, name=df_split.index.name)
    return X, idx_targets


def _build_pred_series_for_cell(
    series: pd.Series,
    cell_id: int,
) -> Optional[Tuple[pd.DataFrame, Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]]]:
    """
    Calcula la serie de predicciones de la GRU para una celda y los rangos de splits.

    Este método:
      - Hace split temporal en train/val/test.
      - Escala por máximo usando solo train.
      - Construye un DataFrame completo con datetime + internet_total.
      - Añade features de calendario (día/semana, hora, festivos, periodos especiales).
      - Reconstruye los splits df_train/df_val/df_test respetando las longitudes.
      - Añade la columna 'traffic_scaled' a partir de las series escaladas.
      - Construye ventanas sobre las series escaladas y las features.
      - Carga el modelo GRU entrenado para la celda.
      - Predice en train/val/test y deshace la escala.
      - Devuelve:
          * Un DataFrame con dos columnas:
              'y_real' : serie real completa (no escalada)
              'y_pred' : predicción GRU (NaN donde no hay ventana)
          * Un diccionario con los rangos temporales de cada split:
              {'train': (t0_train, t1_train), 'val': (...), 'test': (...)}

    Si no existe el modelo para esa celda o no hay datos suficientes,
    devuelve None.
    """
    model_path = GRU_MODELS_DIR / f"gru_cell_{cell_id}.keras"
    if not model_path.exists():
        print(f"[gru_plots] Modelo no encontrado para la celda {cell_id}: {model_path}")
        return None

    # Serie real completa, ordenada por tiempo
    series = series.sort_index()
    full_index = series.index
    y_real = series.astype(float)

    # 1) Split temporal
    s_train, s_val, s_test = split_series(series)

    # Rangos temporales de cada split para el sombreado
    split_ranges: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for name, s in (("train", s_train), ("val", s_val), ("test", s_test)):
        if not s.empty:
            split_ranges[name] = (s.index[0], s.index[-1])

    # 2) Escalado por máximo (usando solo train)
    s_train_s, s_val_s, s_test_s, scale_factor = scale_splits_by_train_max(
        s_train, s_val, s_test
    )

    # 3) Construir un DataFrame completo con datetime + internet_total
    df_full = series.to_frame(name="internet_total").copy()
    df_full.index.name = "datetime"
    df_full = df_full.reset_index()  # columnas: ['datetime', 'internet_total']

    # 3b) Añadir features de calendario (día de la semana, hora, festivos, periodos especiales)
    special_periods_path = Path("Data/calendar/special_periods.csv")
    if special_periods_path.exists():
        df_full = add_calendar_features(df_full, special_periods_path=special_periods_path)
    else:
        df_full = add_calendar_features(df_full, special_periods_path=None)

    # 4) Volver a dividir df_full en train/val/test por posición,
    #    respetando las longitudes de s_train y s_val.
    n_train = len(s_train)
    n_val = len(s_val)

    df_train = df_full.iloc[:n_train].copy()
    df_val = df_full.iloc[n_train:n_train + n_val].copy()
    df_test = df_full.iloc[n_train + n_val:].copy()

    # 5) Añadir la columna de tráfico escalado usando las series escaladas
    df_train = df_train.set_index("datetime")
    df_val = df_val.set_index("datetime")
    df_test = df_test.set_index("datetime")

    df_train["traffic_scaled"] = s_train_s
    df_val["traffic_scaled"] = s_val_s
    df_test["traffic_scaled"] = s_test_s

    # Nos aseguramos de que el índice temporal esté ordenado
    df_train = df_train.sort_index()
    df_val = df_val.sort_index()
    df_test = df_test.sort_index()

    # 6) Definir las columnas de entrada para la GRU (mismas que en entrenamiento)
    feature_cols = ["traffic_scaled", "hour_of_day", "day_of_week", "is_public_holiday"]

    # 7) Construir ventanas (X) para cada split (en escala normalizada)
    X_train, idx_train = _build_windows_for_split(
        df_train,
        feature_cols=feature_cols,
        window=NN_INPUT_WINDOW,
        horizon=NN_HORIZON,
    )
    X_val, idx_val = _build_windows_for_split(
        df_val,
        feature_cols=feature_cols,
        window=NN_INPUT_WINDOW,
        horizon=NN_HORIZON,
    )
    X_test, idx_test = _build_windows_for_split(
        df_test,
        feature_cols=feature_cols,
        window=NN_INPUT_WINDOW,
        horizon=NN_HORIZON,
    )

    if X_train.shape[0] == 0:
        print(f"[gru_plots] Celda {cell_id}: no hay suficientes datos para ventanas.")
        return None

    # 8) Cargar el modelo entrenado
    model = load_model(model_path)

    # 9) Predicciones en escala normalizada
    y_pred_train_s = model.predict(X_train, verbose=0).flatten()
    y_pred_val_s = model.predict(X_val, verbose=0).flatten()
    y_pred_test_s = model.predict(X_test, verbose=0).flatten()

    # 10) Desescalar predicciones
    y_pred_train = y_pred_train_s * scale_factor
    y_pred_val = y_pred_val_s * scale_factor
    y_pred_test = y_pred_test_s * scale_factor

    # 11) Construir una serie completa de predicciones en el índice original
    y_pred_full = pd.Series(
        data=np.nan,
        index=full_index,
        name="y_pred",
        dtype=float,
    )

    # Series de predicciones por split con su índice temporal correspondiente
    y_pred_train_series = pd.Series(
        data=y_pred_train,
        index=idx_train,
        name="y_pred_train",
    )
    y_pred_val_series = pd.Series(
        data=y_pred_val,
        index=idx_val,
        name="y_pred_val",
    )
    y_pred_test_series = pd.Series(
        data=y_pred_test,
        index=idx_test,
        name="y_pred_test",
    )

    # Actualizamos la serie completa con las predicciones de cada split
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


def plot_gru_for_cell(cell_id: int, series: pd.Series) -> None:
    """
    Genera y guarda una figura para una celda con:

        - Serie real completa.
        - Predicción GRU (solo donde hay predicción disponible).
        - Regiones sombreadas para train / val / test.

    Guarda la figura como:
        Modeling/neural_networks/gru/output/plots_all/cell_<id>_gru.png
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

    # Predicción GRU (línea discontinua)
    ax.plot(
        df.index,
        df["y_pred"],
        label="Predicción GRU",
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

    ax.set_title(f"Celda {cell_id} - GRU (H={NN_HORIZON}, ventana={NN_INPUT_WINDOW})")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("internet_total")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", linewidth=0.5)

    fig.tight_layout()

    out_path = PLOTS_DIR / f"cell_{cell_id}_gru.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[gru_plots] Figura guardada para celda {cell_id}: {out_path}")


def run_plots_for_all_cells() -> None:
    """
    Recorre todas las celdas y genera una figura por celda.
    """
    for cell_id, series in iter_cells_by_file():
        print(f"[gru_plots] Generando figura para celda {cell_id}...")
        plot_gru_for_cell(cell_id, series)


if __name__ == "__main__":
    run_plots_for_all_cells()
