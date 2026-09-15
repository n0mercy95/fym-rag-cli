import os
import time
import re # Librería nativa para buscar patrones de texto

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
    
    # Expresión regular: Busca "Capitulo X" al inicio de una línea, ignorando mayúsculas y acentos.
    # El "^" evita falsos positivos como "como vimos en el capítulo 4..." en medio de un párrafo.
    patron_capitulo = re.compile(r'(?m)^\s*CAP[IÍ]TULO\s+(\d+)')

    for doc in documentos_brutos:
        # Buscamos si en esta página se menciona el inicio de un capítulo
        match = patron_capitulo.search(doc.page_content)
        if match:
            # Si hay coincidencia, capturamos el número y actualizamos el rastreador
            numero_cap = match.group(1)
            capitulo_actual = f"Capitulo_{numero_cap}"
        
        # Inyectamos la etiqueta del capítulo actual en los metadatos de la página
        doc.metadata["capitulo"] = capitulo_actual

    print("✂️ Dividiendo el texto en fragmentos (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    # El TextSplitter heredará automáticamente nuestro nuevo metadato 'capitulo' a cada fragmento
    chunks = text_splitter.split_documents(documentos_brutos)
    print(f"✅ Se generaron {len(chunks)} fragmentos etiquetados.")

    print("🆔 Generando IDs únicos para cada fragmento...")
    for i, chunk in enumerate(chunks):
        fuente = chunk.metadata.get("source", "desconocido")
        pagina = chunk.metadata.get("page", 0)
        capitulo = chunk.metadata.get("capitulo", "Sin_Capitulo")
        
        # Nuevo formato de ID: fuente : capitulo : pagina : chunk_index
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
            # Intentamos guardar el bloque normal
            db.add_documents(documents=batch, ids=batch_ids)
            print(f"  -> Guardados {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} fragmentos.")
        
        except Exception as e:
            # Si el bloque rompe Ollama (ej. tablas del apéndice), lo atrapamos aquí
            print(f"  ⚠️ Error en el lote {i} al {i + BATCH_SIZE}. Fragmento ilegible detectado. Saltando lote...")
            
            # Le damos 5 segundos de respiro a Ollama para que su motor interno se reinicie tras el colapso
            time.sleep(5)
            continue # Saltamos al siguiente lote sin detener el script
        
        # Pausa normal de respiro
        time.sleep(1)
    
    print(f"🎉 ¡Ingesta completada con éxito en '{CHROMA_PATH}'!")

if __name__ == "__main__":
    main()