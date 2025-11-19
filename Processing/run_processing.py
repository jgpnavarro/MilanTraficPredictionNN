# run_pipeline.py

from datetime import timedelta
from pathlib import Path

from Processing.config import (
    START_DATE,
    END_DATE,
    RAW_DIR,
    DELETE_RAW_AFTER_PROCESS,
)
from Processing.process_cells_internet import process_raw_file


def main():
    print("=== INICIO PIPELINE COMPLETO ===")
    current = START_DATE

    while current <= END_DATE:
        date_str = current.strftime("%Y-%m-%d")
        filename = f"sms-call-internet-mi-{date_str}.txt"
        raw_path = RAW_DIR / filename

        print(f"\n=== Día {date_str} ===")
        print(f"[pipeline] Buscando raw: {raw_path}")

        if not raw_path.exists():
            print(f"[pipeline] AVISO: no existe el raw {raw_path}, salto este día.")
            current += timedelta(days=1)
            continue

        # 1) Procesar el raw (filtrar celdas + sumar internet)
        processed_path = process_raw_file(raw_path)

        # 2) Borrar el raw si así lo indica la config
        if DELETE_RAW_AFTER_PROCESS and raw_path.exists():
            print(f"[pipeline] Borrando raw: {raw_path}")
            raw_path.unlink()

        print(f"[pipeline] Día {date_str} OK. Archivo procesado: {processed_path}")

        current += timedelta(days=1)

    print("\n=== PIPELINE COMPLETO TERMINADO ===")


if __name__ == "__main__":
    main()
