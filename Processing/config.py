# Processing/config.py
from pathlib import Path

# Ruta a la raíz del proyecto (carpeta que contiene 'Processing', 'Data', etc.)
BASE_DIR = Path(__file__).resolve().parents[1]

# Directorios de datos
DATA_DIR = BASE_DIR / "Data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Celdas de interés
CELL_IDS = [4259, 4456, 4703, 5060, 5085, 5200]

# Nombre del archivo de prueba
RAW_FILE_NAME = "sms-call-internet-mi-2013-11-01.txt"