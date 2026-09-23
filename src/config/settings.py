# src/config/settings.py
"""
Módulo centralizado de configuración.
Carga las variables de entorno desde el archivo .env y las expone
como constantes para todo el proyecto.
"""
import os
from dotenv import load_dotenv

# Carga el .env desde la raíz del proyecto
load_dotenv()

# --- Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# --- API Server ---
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
