from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.config.logger import get_logger

logger = get_logger("FastAPIServer")

app = FastAPI(
    title="⚛️ API RAG Física y Matemáticas",
    description="Backend 100% local potenciado por Qwen2.5 y ChromaDB.",
    version="1.0.0"
)

# Configuración CORS (Vital para cuando conectes tu frontend en React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción, cambia "*" por "http://localhost:3000"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.on_event("startup")
def startup_event():
    logger.info("=== Servidor FastAPI Iniciado ===")

@app.get("/")
def health_check():
    return {"status": "online", "message": "API RAG funcionando. Visita /docs para probar los endpoints."}