import os
import logging

# 1. Rutas absolutas hacia la raíz del proyecto
# Subimos 3 niveles: logger.py -> config/ -> src/ -> fym-rag-cli/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# 2. Crear la carpeta logs/ automáticamente si no existe
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, "app.log")

def get_logger(modulo: str):
    """
    Retorna un logger configurado para escribir en logs/app.log.
    Se le pasa el nombre del módulo para saber exactamente de dónde viene el error.
    """
    logger = logging.getLogger(modulo)
    
    # Evitar que se dupliquen los registros si se llama varias veces
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Definir el formato profesional del log
        formato = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | [%(name)s] | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Configurar el manejador de archivos
        file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
        file_handler.setFormatter(formato)
        
        logger.addHandler(file_handler)
        
    return logger