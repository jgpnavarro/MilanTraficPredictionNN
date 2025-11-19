"""
Entrenamiento y evaluación de un MLP por celda para horizonte fijo.

Este script:
  - Recorre todas las celdas disponibles en los ficheros by_cell.
  - Para cada celda:
      * Hace el split temporal en train/val/test.
      * Normaliza la serie dividiendo por el máximo de train.
      * Construye ventanas deslizantes de longitud NN_INPUT_WINDOW
        para predecir a horizonte NN_HORIZON.
      * Entrena un modelo MLP (uno por celda).
      * Predice en train/val/test y vuelve a unidades reales.
      * Calcula métricas (MAE, RMSE, MAPE, wMAPE, sMAPE, Y_MEAN, n).
  - Agrega los resultados de todas las celdas en un DataFrame.
  - Imprime un resumen y guarda CSVs global y por celda en MLP_OUTPUT_DIR.

La arquitectura del MLP está definida en Modeling/neural_networks/mlp.py.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from keras.callbacks import EarlyStopping

from Modeling.config import (
    NN_HORIZON,
    NN_INPUT_WINDOW,
    MLP_EPOCHS_MAX,
    MLP_BATCH_SIZE,
    MLP_EARLY_STOPPING_PATIENCE,
    MLP_OUTPUT_DIR,
    MLP_MODELS_DIR,
)
from Modeling.data_access import iter_cells_by_file, split_series
from Modeling.scaling import scale_splits_by_train_max
from Modeling.targets import make_windowed_xy_for_horizon_splits
from Modeling.metrics import mae, rmse, mape, wmape, smape
from Modeling.reporting import (
    rows_from_eval_dict,
    aggregate_results,
    print_summary,
    print_per_cell,
    save_results,
    save_per_cell_csv,
)
from Modeling.neural_networks.mlp.mlp import build_mlp


def _compute_metrics_for_split(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calcula las métricas básicas para un split concreto.

    Parámetros
    ----------
    y_true : np.ndarray
        Valores reales en unidades originales (no escaladas).
    y_pred : np.ndarray
        Predicciones en las mismas unidades que y_true.

    Devuelve
    --------
    dict
        Diccionario con:
            - "n": número de muestras
            - "MAE", "RMSE", "MAPE", "wMAPE", "sMAPE", "Y_MEAN"
    """
    n = int(len(y_true))
    if n == 0:
        # Si no hay datos, devolver métricas NaN con n=0.
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


def run_mlp_per_cell(cell_id: int, series) -> List[Dict]:
    """
    Entrena y evalúa un MLP para una celda y un horizonte fijo NN_HORIZON.

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
    print(f"[mlp] Procesando celda {cell_id}...")

    # 1) Split temporal en train/val/test
    s_train, s_val, s_test = split_series(series)

    # 2) Escalado por máximo usando solo train
    s_train_s, s_val_s, s_test_s, scale_factor = scale_splits_by_train_max(
        s_train, s_val, s_test
    )

    # 3) Construir ventanas deslizantes sobre las series escaladas
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

    # Comprobar que hay suficientes datos para entrenar
    if len(X_train) == 0 or len(y_train) == 0:
        print(f"[mlp] Celda {cell_id}: no hay suficientes datos en train, se omite.")
        return []

    # 4) Convertir a arrays numpy (float32) para Keras
    X_train_np = X_train.to_numpy(dtype="float32")
    y_train_np = y_train.to_numpy(dtype="float32")
    X_val_np = X_val.to_numpy(dtype="float32")
    y_val_np = y_val.to_numpy(dtype="float32")
    X_test_np = X_test.to_numpy(dtype="float32")
    y_test_np = y_test.to_numpy(dtype="float32")

    # 5) Construir el modelo MLP para esta celda
    model = build_mlp(input_dim=NN_INPUT_WINDOW)

    # 6) Definir EarlyStopping sobre la pérdida de validación
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=MLP_EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=0,
    )

    # 7) Entrenar el modelo
    history = model.fit(
        X_train_np,
        y_train_np,
        validation_data=(X_val_np, y_val_np),
        epochs=MLP_EPOCHS_MAX,
        batch_size=MLP_BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=0,  # cambiar a 1 si se quiere ver el progreso por época
    )

    # 8) Guardar el modelo entrenado (opcional pero recomendable)
    model_path = MLP_MODELS_DIR / f"mlp_cell_{cell_id}.keras"
    model.save(model_path)

    # 9) Predicciones en train/val/test (en escala normalizada)
    y_pred_train_s = model.predict(X_train_np, verbose=0).flatten()
    y_pred_val_s = model.predict(X_val_np, verbose=0).flatten()
    y_pred_test_s = model.predict(X_test_np, verbose=0).flatten()

    y_train_s_np = y_train_np  # alias para claridad
    y_val_s_np = y_val_np
    y_test_s_np = y_test_np

    # 10) Volver a unidades reales (desescalado)
    #     valor_real = valor_escalado * scale_factor
    y_train_true = y_train_s_np * scale_factor
    y_val_true = y_val_s_np * scale_factor
    y_test_true = y_test_s_np * scale_factor

    y_train_pred = y_pred_train_s * scale_factor
    y_val_pred = y_pred_val_s * scale_factor
    y_test_pred = y_pred_test_s * scale_factor

    # 11) Calcular métricas por split
    eval_train = _compute_metrics_for_split(y_train_true, y_train_pred)
    eval_val = _compute_metrics_for_split(y_val_true, y_val_pred)
    eval_test = _compute_metrics_for_split(y_test_true, y_test_pred)

    eval_dict = {
        "train": eval_train,
        "val": eval_val,
        "test": eval_test,
    }

    # 12) Convertir a filas estándar para reporting
    rows = rows_from_eval_dict(
        cell_id=cell_id,
        horizon=NN_HORIZON,
        eval_dict=eval_dict,
    )

    # Añadir información del modelo y de la ventana a cada fila
    for row in rows:
        row["model"] = "mlp"
        row["window"] = NN_INPUT_WINDOW

    return rows


def run_mlp_for_all_cells() -> None:
    """
    Ejecuta el flujo completo de MLP para todas las celdas.

    Recorre todas las celdas disponibles, entrena un modelo por celda
    y acumula los resultados de métricas por split en un DataFrame.
    """
    all_rows: List[Dict] = []

    for cell_id, series in iter_cells_by_file():
        cell_rows = run_mlp_per_cell(cell_id, series)
        if cell_rows:
            all_rows.extend(cell_rows)

    if not all_rows:
        print("[mlp] No se generaron resultados para ninguna celda.")
        return

    # Agregar resultados en un DataFrame
    df_results = aggregate_results(all_rows)

    # Imprimir resúmenes
    print_summary(df_results)
    print_per_cell(df_results)

    # Guardar resultados en disco
    save_results(
        df_results,
        filename="mlp_results.csv",
        output_dir=MLP_OUTPUT_DIR,
    )

    save_per_cell_csv(
        df_results,
        filename_prefix="mlp_cell_",
        output_dir=MLP_OUTPUT_DIR,
    )


if __name__ == "__main__":
    run_mlp_for_all_cells()
