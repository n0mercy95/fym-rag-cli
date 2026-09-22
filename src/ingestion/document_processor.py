import os

from src.chunking.text_splitter import procesar_pdf_a_chunks
from src.vector_db.chroma_manager import guardar_chunks_en_chroma

import os

from src.chunking.text_splitter import procesar_pdf_a_chunks
from src.vector_db.chroma_manager import guardar_chunks_en_chroma

def run_ingestion(pdf_path: str, db_instance):
    if not os.path.exists(pdf_path):
        print(f"❌ No se encontró el archivo: {pdf_path}")
        return

    # Paso 1: Delegar la extracción y corte
    print("📄 Iniciando procesamiento del documento...")
    chunks_preparados = procesar_pdf_a_chunks(pdf_path)
    
    # Paso 2: Delegar el guardado vectorial
    if chunks_preparados:
        print("📦 Enviando datos a la base vectorial...")
        # Entregamos la conexión viva en lugar de la ruta de texto
        guardar_chunks_en_chroma(chunks=chunks_preparados, db_instance=db_instance)
        print("🎉 ¡Ingesta completada con éxito!")
    else:
        print("⚠️ No se generaron fragmentos para guardar.")