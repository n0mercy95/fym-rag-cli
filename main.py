import os
import argparse

# Importamos las funciones refactorizadas desde sus nuevos submódulos
from src.ingestion.document_processor import run_ingestion
from src.llm.chat_engine import start_interactive_chat

# 1. CENTRALIZACIÓN DE RUTAS ABSOLUTAS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
DEFAULT_PDF = os.path.join(BASE_DIR, "data", "Sears_Zemansky_F_sica_Universitaria_Vol_1.pdf")

def main():
    # 2. CONFIGURACIÓN DE LA INTERFAZ DE LÍNEA DE COMANDOS (CLI)
    parser = argparse.ArgumentParser(
        description="⚛️ Asistente RAG de Física y Matemáticas 100% Local"
    )
    
    parser.add_argument(
        "--mode", 
        type=str,
        choices=["ingest", "chat"], 
        required=True, 
        help="Elige 'ingest' para procesar un PDF o 'chat' para iniciar el asistente."
    )
    
    parser.add_argument(
        "--file", 
        type=str, 
        default=DEFAULT_PDF, 
        help="Ruta del PDF a ingerir (solo aplica si usas --mode ingest)."
    )

    args = parser.parse_args()

    # 3. ENRUTAMIENTO LÓGICO Y EJECUCIÓN
    if args.mode == "ingest":
        print(f"\n[INFO] Iniciando modo Ingesta para el archivo:\n{args.file}\n")
        run_ingestion(pdf_path=args.file, chroma_path=CHROMA_PATH)
    
    elif args.mode == "chat":
        print("\n[INFO] Iniciando motor de Chat...\n")
        start_interactive_chat(chroma_path=CHROMA_PATH)

if __name__ == "__main__":
    main()