# src/api/schemas.py
from pydantic import BaseModel, Field

# --- ESQUEMAS PARA EL CHAT ---

class ChatRequest(BaseModel):
    query: str = Field(..., description="La pregunta matemática o física del usuario", example="¿Qué es la energía cinética?")
    session_id: str = Field(default="default", description="Identificador único para mantener el historial de la conversación")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="La respuesta generada por el LLM o el sistema de corrección")
    hallucination_intercepted: bool = Field(..., description="Indica si el guardián de coseno tuvo que intervenir")
    cosine_score: float = Field(..., description="El puntaje de similitud matemática de la respuesta")

# --- ESQUEMAS PARA LA INGESTA ---

class IngestRequest(BaseModel):
    file_path: str = Field(..., description="Ruta absoluta o relativa del PDF a ingerir")

class IngestResponse(BaseModel):
    status: str = Field(..., description="Estado de la operación")
    message: str = Field(..., description="Detalle del resultado de la ingesta")