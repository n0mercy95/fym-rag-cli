import time

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def get_chroma_db(chroma_path: str) -> Chroma:
    """
    Inicializa y retorna la conexión a ChromaDB.
    ¡Esto es clave! Ahora puedes importar esta función también en chat_engine.py
    para no repetir este código de conexión.
    """
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    
    return Chroma(persist_directory=chroma_path, embedding_function=embeddings)

def guardar_chunks_en_chroma(chunks: list, chroma_path: str, batch_size: int = 200):
    """
    Recibe una lista de fragmentos y los guarda en ChromaDB usando lotes (batching)
    para evitar desbordamientos de memoria.
    """
    print("🧠 Conectando con Ollama y ChromaDB...")
    db = get_chroma_db(chroma_path)

    print(f"📦 Insertando fragmentos en lotes de {batch_size} para proteger la RAM...")
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_ids = [chunk.metadata["id"] for chunk in batch]
        
        try:
            db.add_documents(documents=batch, ids=batch_ids)
            print(f"  -> Guardados {min(i + batch_size, len(chunks))}/{len(chunks)} fragmentos.")
        
        except Exception as e:
            print(f"  ⚠️ Error en el lote {i} al {i + batch_size}. Fragmento ilegible detectado. Saltando lote...")
            # Aquí he agregado el mensaje de error real por si acaso falla algo específico
            print(f"  [Detalle del error]: {e}")
            time.sleep(5)
            continue
        
        time.sleep(1)