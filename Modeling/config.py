"""
Parámetros de modelado (no de procesamiento).

Este módulo centraliza la configuración para entrenar y evaluar modelos,
sin duplicar rutas ni lógica del preprocesado. Cuando es posible, reutiliza las
rutas definidas en Processing.config para acceder a los datos ya preparados.
"""

from pathlib import Path

# Horizontes a evaluar: 1 paso (10 minutos) y 6 pasos (1 hora si freq=10 minutos).
H_LIST = [1, 6]
# Tamaños de ventana en número de puntos (10 minutos por punto)
MOVING_AVG_WINDOWS = [6, 12]

# Horizonte de predicción para Redes Neuronales.
NN_HORIZON = 6
# Ventana de entrada para Neural Networks (número de pasos hacia atrás).
NN_INPUT_WINDOW = 12

# Proporciones del split temporal: train, val, test (deben sumar 1.0).
SPLIT = (0.70, 0.15, 0.15)

# Hiperparámetros de entrenamiento básicos para el MLP
NN_EPOCHS_MAX = 50          # número máximo de épocas de entrenamiento
NN_BATCH_SIZE = 128         # tamaño de lote (batch size)
NN_EARLY_STOPPING_PATIENCE = 10  # paciencia para early stopping (en épocas)

# Frecuencia esperada de la serie temporal. "10T" equivale a 10 minutos.
FREQ = "10T"

# Control de guardado de resultados (por ejemplo, métricas en CSV).
SAVE_RESULTS = True

# Carpeta de salida para resultados de modelado (no datos crudos).
BASE_DIR = Path(__file__).resolve().parents[1]

PERSISTENCE_OUTPUT_DIR = BASE_DIR / "Modeling" / "baselines" / "persistence"/ "output"
PERSISTENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MOVING_AVG_OUTPUT_DIR = BASE_DIR / "Modeling" / "baselines" / "moving_average" / "output"
MOVING_AVG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NN_BASE_DIR = BASE_DIR / "Modeling" / "neural_networks"
MLP_OUTPUT_DIR = NN_BASE_DIR / "mlp" / "output"
MLP_MODELS_DIR = NN_BASE_DIR / "mlp" / "models"
MLP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MLP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
GRU_OUTPUT_DIR = NN_BASE_DIR/"gru/output"
GRU_MODELS_DIR = NN_BASE_DIR/"gru/models"
GRU_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRU_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Localización de los datos procesados. Se intenta tomar de Processing.config.
# Si no está disponible, se recurre a la ruta relativa por defecto "Data/processed".
try:
    from Processing.config import PROCESSED_DIR as _PROCESSED_DIR
    PROCESSED_DIR = _PROCESSED_DIR
except Exception:
    PROCESSED_DIR = BASE_DIR / "Data" / "processed"
