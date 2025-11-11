# Processing/process_cells_internet.py

from pathlib import Path

import pandas as pd

from Processing.config import RAW_DIR, PROCESSED_DIR, CELL_IDS

COLS = [
    "square_id",
    "time_interval",
    "country_code",
    "sms_in",
    "sms_out",
    "call_in",
    "call_out",
    "internet",
]

SEP = "\t"


def process_raw_file(
    raw_path: Path,
    output_file_name: str | None = None,
) -> Path:
    """
    Procesa un archivo raw:
      - filtra por CELL_IDS
      - suma 'internet' por (square_id, time_interval)
      - devuelve un único CSV con columnas:
            square_id, time_interval, internet_total
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"No encuentro el fichero raw: {raw_path}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if output_file_name is None:
        stem = raw_path.stem  # sms-call-internet-mi-YYYY-MM-DD
        output_file_name = f"{stem}_internet_total.csv"

    output_path = PROCESSED_DIR / output_file_name

    print(f"[process_raw] Leyendo y procesando: {raw_path}")
    print(f"[process_raw] Celdas de interés: {CELL_IDS}")
    print(f"[process_raw] Guardando en: {output_path}")

    if output_path.exists():
        print(f"[process_raw] {output_path} ya existe, lo borro para regenerarlo.")
        output_path.unlink()

    filtered_chunks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        raw_path,
        sep=SEP,
        header=None,
        names=COLS,
        chunksize=500_000,
    ):
        chunk = chunk[chunk["square_id"].isin(CELL_IDS)]

        if chunk.empty:
            continue

        filtered = chunk[["square_id", "time_interval", "internet"]]
        filtered_chunks.append(filtered)

    if not filtered_chunks:
        print("[process_raw] No hay filas para las celdas de interés. Archivo vacío.")
        empty_df = pd.DataFrame(columns=["square_id", "time_interval", "internet_total"])
        empty_df.to_csv(output_path, index=False)
        return output_path

    df = pd.concat(filtered_chunks, ignore_index=True)
    df["internet"] = df["internet"].fillna(0.0)

    agg = (
        df.groupby(["square_id", "time_interval"], as_index=False)["internet"]
        .sum()
        .rename(columns={"internet": "internet_total"})
    )

    agg.to_csv(output_path, index=False)

    print("[process_raw] Procesado terminado.")
    return output_path


if __name__ == "__main__":
    # prueba manual con un archivo concreto
    test_raw = RAW_DIR / "sms-call-internet-mi-2013-11-01.txt"
    process_raw_file(test_raw)
