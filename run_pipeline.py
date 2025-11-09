# run_pipeline.py
from Processing.process_cells_internet import process_raw_file


def main():
    """
    Pipeline de datos (versión inicial):
    1) Procesar un archivo raw:
       - filtrar por celdas
       - sumar internet por (square_id, time_interval)
       - guardar un único CSV final
    """
    print("=== INICIO PIPELINE ===")

    # Paso único: procesar el archivo raw definido en config
    output_path = process_raw_file()

    print("=== PIPELINE TERMINADO ===")
    print(f"Archivo final generado en: {output_path}")


if __name__ == "__main__":
    main()
