import os
from src.api.services import RAGService

# Rutas absolutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# 1. Instanciamos el servicio como un Singleton (única instancia global)
rag_service_instance = RAGService(chroma_path=CHROMA_PATH)

def get_rag_service() -> RAGService:
    """
    Dependencia de FastAPI. Inyecta la instancia única de RAGService 
    en los endpoints que la necesiten.
    """
    return rag_service_instance