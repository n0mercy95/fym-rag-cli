import time

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 1. Importamos el logger
from src.config.logger import get_logger
from src.config.settings import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL

# 2. Inicializamos el logger para este módulo específico
logger = get_logger("VectorDB")

def get_chroma_db(chroma_path: str) -> Chroma:
    """
    Inicializa y retorna la conexión a ChromaDB.
    Se utiliza al levantar el servidor web para mantener una conexión persistente.
    """
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    
    return Chroma(persist_directory=chroma_path, embedding_function=embeddings)

def guardar_chunks_en_chroma(chunks: list, db_instance: Chroma, batch_size: int = 200):
    """
    Recibe una lista de fragmentos y los guarda en ChromaDB usando lotes (batching)
    para evitar desbordamientos de memoria. Utiliza una conexión ya existente (db_instance)
    para evitar bloqueos de lectura/escritura (Error 1032).
    """
    print("🧠 Preparando inserción usando la base vectorial activa...")
    logger.info("Recibida instancia de ChromaDB viva. Preparando inserción.")
    
    print(f"📦 Insertando fragmentos en lotes de {batch_size} para proteger la RAM...")
    logger.info(f"Iniciando inserción de {len(chunks)} fragmentos en lotes de {batch_size}.")
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_ids = [chunk.metadata["id"] for chunk in batch]
        
        try:
            # Usamos directamente db_instance en lugar de abrir una conexión nueva
            db_instance.add_documents(documents=batch, ids=batch_ids)
            print(f"  -> Guardados {min(i + batch_size, len(chunks))}/{len(chunks)} fragmentos.")
        
        except Exception as e:
            error_msg = str(e)
            
            # Interceptamos el error específico de Ollama (status code 500 / EOF)
            if "status code: 500" in error_msg or "EOF" in error_msg:
                mensaje_limpio = f"Ollama rechazó el lote (posible texto corrupto al final del PDF). Detalle: {error_msg}"
                print(f"  ⚠️ Lote {i} al {i + batch_size} rechazado por Ollama. Saltando lote...")
                
                # Registramos en el log de forma limpia, SIN exc_info=True
                logger.warning(f"Fallo en lote {i}-{i+batch_size}: {mensaje_limpio}")
            else:
                import traceback
                # Imprimimos el error directamente en la terminal
                print(f"  ❌ Error inesperado en el lote {i} al {i + batch_size}: {e}")
                traceback.print_exc() # Esto forzará el rastro técnico (Traceback) en la consola
                
                # Mantenemos el logger por si la carpeta existe
                logger.error(f"Fallo crítico al insertar el lote {i}-{i+batch_size}: {e}", exc_info=True)
            time.sleep(5)
            continue
        
        time.sleep(1)
    
    logger.info("Proceso de inserción en ChromaDB finalizado con éxito.")