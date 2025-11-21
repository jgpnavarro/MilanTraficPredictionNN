"""
Pipeline de energía del TFM.

Este script asume que:
  - Los modelos ya se han entrenado/evaluado antes
    (baselines, MLP, GRU, etc.).
  - Por tanto, sólo ejecuta la parte de simulación energética:

      1) Cálculo de ahorro energético con todos los modelos:
         python -m Modeling.energy.run_energy --models all

      2) Generación de las gráficas de consumo acumulado:
         python -m Modeling.energy.run_plots

No re-entrena nada y no vuelve a ejecutar run_orchestrator.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import os


def run_cmd(description: str, cmd: list[str]) -> None:
    """
    Ejecuta un comando de consola y muestra un mensaje antes y después.

    Lanza excepción si el comando devuelve un código de error.
    """
    print(f"\n[run_energy_pipeline] {description}")
    print(f"[run_energy_pipeline] Ejecutando: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[run_energy_pipeline] OK: {description}")


def main() -> None:
    # Asegurarnos de estar en la raíz del repo
    repo_root = Path(__file__).resolve().parent
    print(f"[run_energy_pipeline] Repo root: {repo_root}")
    try:
        os.chdir(repo_root)
    except Exception as e:
        print(f"[run_energy_pipeline] Aviso: no se pudo cambiar el cwd a {repo_root}: {e}")

    # 1) Cálculo de ahorro energético con todos los modelos
    run_cmd(
        "Paso 1/2: cálculo de ahorro energético (todos los modelos)",
        [
            sys.executable,
            "-m",
            "Modeling.energy.run_energy",
            "--models",
            "all",
        ],
    )

    # 2) Plots de consumo acumulado base vs política
    run_cmd(
        "Paso 2/2: generación de plots de energía (consumo acumulado)",
        [
            sys.executable,
            "-m",
            "Modeling.energy.run_plots",
        ],
    )

    print("\n[run_energy_pipeline] Pipeline de energía completado.")


if __name__ == "__main__":
    main()
