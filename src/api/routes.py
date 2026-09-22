import os
import shutil
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from src.api.schemas import ChatRequest, ChatResponse, IngestRequest, IngestResponse
from src.api.services import RAGService
from src.api.dependencies import get_rag_service

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest, service: RAGService = Depends(get_rag_service)):
    """
    Recibe una pregunta y retorna la respuesta del LLM validada matemáticamente.
    """
    try:
        resultado = service.chat(query=request.query, session_id=request.session_id)
        return ChatResponse(**resultado)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del motor de chat: {str(e)}")

@router.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(
    file: UploadFile = File(...), 
    service: RAGService = Depends(get_rag_service)
):
    """
    Recibe un archivo PDF, lo guarda en el servidor y activa la vectorización.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF.")

    # 1. Aseguramos que la carpeta data/ exista
    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
    os.makedirs(data_folder, exist_ok=True)
    
    # 2. Guardamos el archivo subido en nuestra carpeta
    file_path = os.path.join(data_folder, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 3. Activamos la ingesta usando la ruta del archivo recién guardado
    exito, mensaje = service.ingest(file_path=file_path)
    
    if not exito:
        raise HTTPException(status_code=500, detail=mensaje)
        
    return IngestResponse(status="success", message=f"Archivo '{file.filename}' guardado e ingerido con éxito.")


