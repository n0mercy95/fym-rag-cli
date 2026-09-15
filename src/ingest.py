import os
import time
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Rutas Absolutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Sears_Zemansky_F_sica_Universitaria_Vol_1.pdf")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ No se encontró el archivo: {DATA_PATH}")
        return

    print("📄 Cargando documento...")
    loader = PyPDFLoader(DATA_PATH)
    documentos_brutos = loader.load()

    print("🏷️ Etiquetando capítulos...")
    capitulo_actual = "Prefacio_o_Indice"
    
    # Expresión regular: Busca "Capitulo X" al inicio de una línea
    patron_capitulo = re.compile(r'(?m)^\s*CAP[IÍ]TULO\s+(\d+)')

    for doc in documentos_brutos:
        match = patron_capitulo.search(doc.page_content)
        if match:
            numero_cap = match.group(1)
            capitulo_actual = f"Capitulo_{numero_cap}"
        
        doc.metadata["capitulo"] = capitulo_actual

    print("✂️ Dividiendo el texto en fragmentos (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documentos_brutos)
    print(f"✅ Se generaron {len(chunks)} fragmentos etiquetados.")

    print("🆔 Generando IDs únicos para cada fragmento...")
    for i, chunk in enumerate(chunks):
        fuente = chunk.metadata.get("source", "desconocido")
        pagina = chunk.metadata.get("page", 0)
        capitulo = chunk.metadata.get("capitulo", "Sin_Capitulo")
        
        chunk.metadata["id"] = f"{fuente}:{capitulo}:{pagina}:{i}"

    print("🧠 Conectando con Ollama y ChromaDB...")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    print("📦 Insertando fragmentos por lotes (Batching) para proteger la RAM...")
    BATCH_SIZE = 200
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_ids = [chunk.metadata["id"] for chunk in batch]
        
        try:
            db.add_documents(documents=batch, ids=batch_ids)
            print(f"  -> Guardados {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} fragmentos.")
        
        except Exception as e:
            print(f"  ⚠️ Error en el lote {i} al {i + BATCH_SIZE}. Fragmento ilegible detectado. Saltando lote...")
            time.sleep(5)
            continue
        
        time.sleep(1)
    
    print(f"🎉 ¡Ingesta completada con éxito en '{CHROMA_PATH}'!")

if __name__ == "__main__":
    main()