"""
Entrenamiento y evaluación de una GRU por celda para horizonte fijo.

Este script:
  - Recorre todas las celdas disponibles en los ficheros by_cell.
  - Para cada celda:
      * Hace el split temporal en train/val/test.
      * Normaliza la serie dividiendo por el máximo de train.
      * Añade features de calendario (día de la semana, hora, festivos, periodos especiales).
      * Construye ventanas deslizantes de longitud NN_INPUT_WINDOW
        para predecir a horizonte NN_HORIZON.
      * Entrena un modelo GRU (uno por celda).
      * Predice en train/val/test y vuelve a unidades reales.
      * Calcula métricas (MAE, RMSE, MAPE, wMAPE, sMAPE, Y_MEAN, n).
  - Agrega los resultados de todas las celdas en un DataFrame.
  - Imprime un resumen y guarda CSVs global y por celda en GRU_OUTPUT_DIR.

La arquitectura de la GRU está definida en Modeling/neural_networks/gru/gru.py.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping

from Modeling.config import (
    NN_HORIZON,
    NN_INPUT_WINDOW,
    NN_EPOCHS_MAX,
    NN_BATCH_SIZE,
    NN_EARLY_STOPPING_PATIENCE,
    GRU_OUTPUT_DIR,
    GRU_MODELS_DIR,
)

from Modeling.data_access import iter_cells_by_file, split_series
from Modeling.scaling import scale_splits_by_train_max
from Modeling.features_calendar import add_calendar_features
from Modeling.metrics import mae, rmse, mape, wmape, smape
from Modeling.reporting import (
    rows_from_eval_dict,
    aggregate_results,
    print_summary,
    print_per_cell,
    save_results,
    save_per_cell_csv,
)
from Modeling.neural_networks.gru.gru import build_gru


# Por simplicidad, reutilizamos los mismos hiperparámetros que en el MLP
GRU_EPOCHS_MAX = NN_EPOCHS_MAX
GRU_BATCH_SIZE = NN_BATCH_SIZE
GRU_EARLY_STOPPING_PATIENCE = NN_EARLY_STOPPING_PATIENCE


def _compute_metrics_for_split(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calcula las métricas básicas para un split concreto.

    y_true e y_pred deben estar en unidades originales (no escaladas).
    """
    n = int(len(y_true))
    if n == 0:
        return {
            "n": 0,
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "MAPE": float("nan"),
            "wMAPE": float("nan"),
            "sMAPE": float("nan"),
            "Y_MEAN": float("nan"),
        }

    mae_val = mae(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    wmape_val = wmape(y_true, y_pred)
    smape_val = smape(y_true, y_pred)
    y_mean = float(np.mean(y_true))

    return {
        "n": n,
        "MAE": mae_val,
        "RMSE": rmse_val,
        "MAPE": mape_val,
        "wMAPE": wmape_val,
        "sMAPE": smape_val,
        "Y_MEAN": y_mean,
    }


def _build_windows_for_split(
    df_split: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    window: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construye las ventanas (X, y) para un split concreto (train/val/test).

    Para cada posición t dentro del split se toma:
        - X_t: los 'window' pasos anteriores incluyendo t
        - y_t: el valor de target_col en t + horizon

    Devuelve:
        X : (n_muestras, window, n_features)
        y : (n_muestras,)
    """
    values_features = df_split[feature_cols].to_numpy(dtype="float32")
    values_target = df_split[target_col].to_numpy(dtype="float32")
    n = len(df_split)

    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    last_input_idx = n - horizon - 1
    for t in range(window - 1, last_input_idx + 1):
        start = t - window + 1
        end = t + 1  # end exclusivo

        x_window = values_features[start:end, :]
        y_target = values_target[t + horizon]

        X_list.append(x_window)
        y_list.append(y_target)

    if not X_list:
        return (
            np.empty((0, window, len(feature_cols)), dtype="float32"),
            np.empty((0,), dtype="float32"),
        )

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype="float32")
    return X, y


def run_gru_per_cell(cell_id: int, series) -> List[Dict]:
    """
    Entrena y evalúa una GRU para una celda y un horizonte fijo NN_HORIZON.

    Parámetros
    ----------
    cell_id : int
        Identificador de la celda.
    series : pd.Series
        Serie completa de la celda (DatetimeIndex, valores de internet_total).

    Devuelve
    --------
    rows : list[dict]
        Lista de filas de resultados para esta celda, listas para pasar
        a aggregate_results. Cada fila contiene métricas por split.
    """
    print(f"[gru] Procesando celda {cell_id}...")

    # 1) Split temporal en train/val/test sobre la serie original
    s_train, s_val, s_test = split_series(series)

    # 2) Escalado por máximo usando solo train
    s_train_s, s_val_s, s_test_s, scale_factor = scale_splits_by_train_max(
        s_train, s_val, s_test
    )

    # 3) Construir un DataFrame completo con datetime + internet_total
    #    para poder añadir las features de calendario.
    df_full = series.to_frame(name="internet_total").copy()
    df_full.index.name = "datetime"
    df_full = df_full.reset_index()  # columnas: ['datetime', 'internet_total']

    # Añadir features de calendario (día de la semana, hora, festivos, periodos especiales)
    special_periods_path = Path("Data/calendar/special_periods.csv")
    if special_periods_path.exists():
        df_full = add_calendar_features(df_full, special_periods_path=special_periods_path)
    else:
        df_full = add_calendar_features(df_full, special_periods_path=None)

    # 4) Volver a dividir df_full en train/val/test por posición,
    #    respetando las longitudes obtenidas con split_series.
    n_train = len(s_train)
    n_val = len(s_val)

    df_train = df_full.iloc[:n_train].copy()
    df_val = df_full.iloc[n_train:n_train + n_val].copy()
    df_test = df_full.iloc[n_train + n_val:].copy()

    # 5) Añadir la columna de tráfico escalado usando las series escaladas
    #    (alineadas por índice datetime).
    df_train = df_train.set_index("datetime")
    df_val = df_val.set_index("datetime")
    df_test = df_test.set_index("datetime")

    df_train["traffic_scaled"] = s_train_s
    df_val["traffic_scaled"] = s_val_s
    df_test["traffic_scaled"] = s_test_s

    df_train = df_train.reset_index()
    df_val = df_val.reset_index()
    df_test = df_test.reset_index()

    # 6) Definir las columnas de entrada para la GRU
    #    Usamos solo:
    #      - tráfico escalado
    #      - hora del día
    #      - día de la semana
    #      - festivo oficial
    #
    #    No usamos is_special_break como feature de entrada porque la mayor parte
    #    del periodo marcado cae en test y el modelo no tendría ejemplos en
    #    train/val para aprender su efecto. La mantenemos solo como etiqueta
    #    para análisis y visualización.
    feature_cols = ["traffic_scaled", "hour_of_day", "day_of_week", "is_public_holiday"]

    n_features = len(feature_cols)

    # 7) Construir ventanas (X, y) para cada split (en escala normalizada)
    X_train, y_train_s = _build_windows_for_split(
        df_train,
        feature_cols=feature_cols,
        target_col="traffic_scaled",
        window=NN_INPUT_WINDOW,
        horizon=NN_HORIZON,
    )
    X_val, y_val_s = _build_windows_for_split(
        df_val,
        feature_cols=feature_cols,
        target_col="traffic_scaled",
        window=NN_INPUT_WINDOW,
        horizon=NN_HORIZON,
    )
    X_test, y_test_s = _build_windows_for_split(
        df_test,
        feature_cols=feature_cols,
        target_col="traffic_scaled",
        window=NN_INPUT_WINDOW,
        horizon=NN_HORIZON,
    )

    if X_train.shape[0] == 0 or y_train_s.shape[0] == 0:
        print(f"[gru] Celda {cell_id}: no hay suficientes datos en train, se omite.")
        return []

    # 8) Construir el modelo GRU para esta celda
    model = build_gru(
        input_timesteps=NN_INPUT_WINDOW,
        n_features=n_features,
    )

    # 9) Definir EarlyStopping sobre la pérdida de validación
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=GRU_EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=0,
    )

    # 10) Entrenar el modelo
    model.fit(
        X_train,
        y_train_s,
        validation_data=(X_val, y_val_s),
        epochs=GRU_EPOCHS_MAX,
        batch_size=GRU_BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=0,
    )

    # 11) Guardar el modelo entrenado
    model_path = GRU_MODELS_DIR / f"gru_cell_{cell_id}.keras"
    model.save(model_path)

    # 12) Predicciones en train/val/test (en escala normalizada)
    y_pred_train_s = model.predict(X_train, verbose=0).flatten()
    y_pred_val_s = model.predict(X_val, verbose=0).flatten()
    y_pred_test_s = model.predict(X_test, verbose=0).flatten()

    # 13) Volver a unidades reales
    y_train_true = y_train_s * scale_factor
    y_val_true = y_val_s * scale_factor
    y_test_true = y_test_s * scale_factor

    y_train_pred = y_pred_train_s * scale_factor
    y_val_pred = y_pred_val_s * scale_factor
    y_test_pred = y_pred_test_s * scale_factor

    # 14) Calcular métricas por split
    eval_train = _compute_metrics_for_split(y_train_true, y_train_pred)
    eval_val = _compute_metrics_for_split(y_val_true, y_val_pred)
    eval_test = _compute_metrics_for_split(y_test_true, y_test_pred)

    eval_dict = {
        "train": eval_train,
        "val": eval_val,
        "test": eval_test,
    }

    # 15) Convertir a filas estándar para reporting
    rows = rows_from_eval_dict(
        cell_id=cell_id,
        horizon=NN_HORIZON,
        eval_dict=eval_dict,
    )

    # Añadir información del modelo y de la ventana a cada fila
    for row in rows:
        row["model"] = "gru"
        row["window"] = NN_INPUT_WINDOW
        row["n_features"] = n_features

    return rows


def run_gru_for_all_cells() -> None:
    """
    Ejecuta el flujo completo de GRU para todas las celdas.

    Recorre todas las celdas disponibles, entrena un modelo por celda
    y acumula los resultados de métricas por split en un DataFrame.
    """
    all_rows: List[Dict] = []

    for cell_id, series in iter_cells_by_file():
        cell_rows = run_gru_per_cell(cell_id, series)
        if cell_rows:
            all_rows.extend(cell_rows)

    if not all_rows:
        print("[gru] No se generaron resultados para ninguna celda.")
        return

    # Agregar resultados en un DataFrame
    df_results = aggregate_results(all_rows)

    # Imprimir resúmenes
    print_summary(df_results)
    print_per_cell(df_results)

    # Guardar resultados en disco
    save_results(
        df_results,
        filename="gru_results.csv",
        output_dir=GRU_OUTPUT_DIR,
    )

    save_per_cell_csv(
        df_results,
        filename_prefix="gru_cell_",
        output_dir=GRU_OUTPUT_DIR,
    )


if __name__ == "__main__":
    run_gru_for_all_cells()
