# Processing/process_cells_internet.py

from pathlib import Path

import pandas as pd

from Processing.config import RAW_DIR, PROCESSED_DIR, CELL_IDS, RAW_FILE_NAME

# nombres de columnas según el dataset de Telecom Italia
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

SEP = "\t"  # si algún día ves que no es tabulado, lo cambiamos


def process_raw_file(
    raw_file_name: str | None = None,
    output_file_name: str | None = None,
) -> Path:
    """
    Procesa un archivo raw:
      - filtra por CELL_IDS
      - suma 'internet' por (square_id, time_interval)
      - devuelve un único CSV con columnas:
            square_id, time_interval, internet_total
    """
    # Si no se pasa nombre, usamos el de config
    raw_file_name = raw_file_name or RAW_FILE_NAME

    raw_path = RAW_DIR / raw_file_name

    if output_file_name is None:
        # p.ej. sms-call-internet-mi-2013-11-01.txt -> sms-call-internet-mi-2013-11-01_internet_total.csv
        stem = raw_path.stem  # nombre sin extensión
        output_file_name = f"{stem}_internet_total.csv"

    output_path = PROCESSED_DIR / output_file_name

    if not raw_path.exists():
        raise FileNotFoundError(f"No encuentro el fichero raw: {raw_path}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[process_raw] Leyendo y procesando: {raw_path}")
    print(f"[process_raw] Celdas de interés: {CELL_IDS}")
    print(f"[process_raw] Guardando en: {output_path}")

    if output_path.exists():
        print(f"[process_raw] {output_path} ya existe, lo borro para regenerarlo.")
        output_path.unlink()

    # Aquí iremos guardando los trozos filtrados (ya con pocas filas)
    filtered_chunks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        raw_path,
        sep=SEP,
        header=None,
        names=COLS,
        chunksize=500_000,
    ):
        # Filtramos solo las celdas de interés
        chunk = chunk[chunk["square_id"].isin(CELL_IDS)]

        if chunk.empty:
            continue

        # Nos quedamos solo con lo que te interesa para el resultado final
        filtered = chunk[["square_id", "time_interval", "internet"]]

        filtered_chunks.append(filtered)

    if not filtered_chunks:
        print("[process_raw] No hay filas para las celdas de interés. Archivo vacío.")
        # Creamos un CSV vacío con la estructura correcta
        empty_df = pd.DataFrame(
            columns=["square_id", "time_interval", "internet_total"]
        )
        empty_df.to_csv(output_path, index=False)
        return output_path

    # Unimos todos los trozos filtrados (siguen siendo pocos datos)
    df = pd.concat(filtered_chunks, ignore_index=True)

    # Rellenamos NaN de internet con 0 (no tráfico)
    df["internet"] = df["internet"].fillna(0.0)

    # Agrupamos por celda + instante y sumamos internet
    agg = (
        df
        .groupby(["square_id", "time_interval"], as_index=False)["internet"]
        .sum()
    )

    agg = agg.rename(columns={"internet": "internet_total"})

    # Guardamos un único archivo final
    agg.to_csv(output_path, index=False)

    print("[process_raw] Procesado terminado.")
    return output_path


if __name__ == "__main__":
    # Para probar este módulo directamente si quieres:
    process_raw_file()
