"""
Utilidades de visualización para el modelo de persistencia.

Incluye funciones para:
- Pintar la serie completa y su predicción (overlay) y guardar en PNG.

Todas las gráficas se guardan en la carpeta de salida específica de persistencia:
  Modeling/persistence/output/plots/<cell_id>/...
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def _ensure_parent_dir(out_path: Path) -> None:
    """
    Crea el directorio padre del fichero de salida si no existe.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)


def plot_overlay_full_series(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str,
    out_path: Path,
) -> None:
    """
    Dibuja y guarda una gráfica con la serie real y la predicción superpuestas.
    Se usan estilos claramente distintos y se dibuja la verdad por encima.
    """
    _ensure_parent_dir(out_path)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4.5))

    # Predicción primero (debajo), discontinua y un poco más fina.
    plt.plot(
        y_pred.index, y_pred.values,
        label="Predicción (persistencia)",
        linestyle="-", linewidth=0.7, zorder=2
    )

    # Real después (encima), sólida y algo más gruesa.
    plt.plot(
        y_true.index, y_true.values,
        label="Real",
        linestyle="-", linewidth=0.7, zorder=1
    )

    plt.title(title)
    plt.xlabel("Tiempo")
    plt.ylabel("Tráfico (internet_total)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def plot_overlay_full_series_with_splits(
    y_true: pd.Series,
    y_pred: pd.Series,
    split_ranges: dict,
    title: str,
    out_path: Path,
) -> None:
    """
    Igual que plot_overlay_full_series, pero sombrea las zonas de train/val/test.

    split_ranges: diccionario con claves 'train', 'val', 'test' y tuplas (t_ini, t_fin).
    """
    _ensure_parent_dir(out_path)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4.5))

    # Predicción debajo, discontinua
    plt.plot(y_pred.index, y_pred.values, label="Predicción (persistencia)",
             linestyle="-", linewidth=0.7, zorder=2)
    # Real encima, sólida
    plt.plot(y_true.index, y_true.values, label="Real",
             linestyle="-", linewidth=0.7, zorder=1)

    # Sombreado de splits
    shades = {
        "train": {"alpha": 0.08},
        "val":   {"alpha": 0.20},
        "test":  {"alpha": 0.30},
    }
    for split_name, (t0, t1) in split_ranges.items():
        style = shades.get(split_name, {"alpha": 0.1})
        plt.axvspan(t0, t1, **style, zorder=0)

    plt.title(title)
    plt.xlabel("Tiempo")
    plt.ylabel("Tráfico (internet_total)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

