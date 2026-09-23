import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.config.settings import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

def main():
    print("🔌 Conectando con ChromaDB para auditoría...")
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # Probamos las dos queries clave
    queries = ["energia cinetica", "tensor"]
    
    for q in queries:
        print(f"\n========================================")
        print(f"🔍 Evaluando query: '{q}'")
        print(f"========================================")
        
        resultados = db.similarity_search_with_score(q, k=3)
        
        for i, (doc, score) in enumerate(resultados):
            print(f"\n  [Resultado {i+1}] Score/Distancia: {score:.4f}")
            print(f"  Capítulo: {doc.metadata.get('capitulo')} | Pág: {doc.metadata.get('page')}")
            print(f"  Texto: {doc.page_content[:150]}...")

if __name__ == "__main__":
    main()