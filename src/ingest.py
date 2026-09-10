import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Rutas
DATA_PATH = "data/Sears_Zemansky_F_sica_Universitaria_Vol_1_Cap_1.pdf"
CHROMA_PATH = "chroma_db"

def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ No se encontró el archivo: {DATA_PATH}")
        return

    print("📄 Cargando documento...")
    loader = PyPDFLoader(DATA_PATH)
    documentos = loader.load()

    print("✂️ Dividiendo el texto en fragmentos (chunks)...")
    # Fragmentamos el texto para no exceder la ventana de contexto del LLM
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"✅ Se generaron {len(chunks)} fragmentos.")

    print("🆔 Generando IDs únicos para cada fragmento...")
    # Imitamos la lógica de asignar un ID determinista: "fuente:pagina:chunk_index"
    for i, chunk in enumerate(chunks):
        fuente = chunk.metadata.get("source", "desconocido")
        pagina = chunk.metadata.get("page", 0)
        # ID formato: data/zemansky_cap1.pdf:5:12
        chunk_id = f"{fuente}:{pagina}:{i}"
        chunk.metadata["id"] = chunk_id

    print("🧠 Generando embeddings y guardando en ChromaDB...")
    # Conectamos con el modelo de Ollama local
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434" # Usa el puerto estándar de Ollama local
    )

    # Insertamos en la base de datos vectorial
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH,
        ids=[chunk.metadata["id"] for chunk in chunks]
    )
    
    print(f"🎉 ¡Ingesta completada! Los datos están en '{CHROMA_PATH}'.")

if __name__ == "__main__":
    main()