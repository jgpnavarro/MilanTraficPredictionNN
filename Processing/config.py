# Processing/config.py
from pathlib import Path
from datetime import date

# Ruta a la raíz del proyecto (carpeta que contiene 'Processing', 'Data', etc.)
BASE_DIR = Path(__file__).resolve().parents[1]

# Directorios de datos
DATA_DIR = BASE_DIR / "Data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Celdas de interés
CELL_IDS = [4259, 4456, 4703, 5060, 5085, 5200]

# Nombre de un archivo de prueba
RAW_FILE_NAME = "sms-call-internet-mi-2013-11-01.txt"
FILTERED_FILE_NAME = "sms-call-internet-mi-2013-11-01.csv"

# Rango de fechas
START_DATE = date(2013, 11, 1)
END_DATE = date(2014, 1, 1) 

# ¿Borrar el raw después de procesarlo?
DELETE_RAW_AFTER_PROCESS = False 
