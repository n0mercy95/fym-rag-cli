import os
import argparse

# Importamos las funciones refactorizadas desde sus nuevos submódulos
from src.ingestion.document_processor import run_ingestion
from src.llm.chat_engine import start_interactive_chat
from src.config.logger import get_logger  # <-- Importamos el configurador de logs

# Inicializamos el logger para este archivo
logger = get_logger("Main")

# 1. CENTRALIZACIÓN DE RUTAS ABSOLUTAS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
DEFAULT_PDF = os.path.join(BASE_DIR, "data", "Sears_Zemansky_F_sica_Universitaria_Vol_1.pdf")

def main():
    logger.info("=== Nueva ejecución de fym-rag-cli iniciada ===")
    
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
        # Mantenemos el print para la interfaz visual, pero registramos en el log internamente
        print(f"\n[INFO] Iniciando modo Ingesta para el archivo:\n{args.file}\n")
        logger.info(f"Modo ingesta seleccionado. Archivo objetivo: {args.file}")
        
        try:
            run_ingestion(pdf_path=args.file, chroma_path=CHROMA_PATH)
            logger.info("Proceso de ingesta finalizado sin interrupciones.")
        except Exception as e:
            logger.error(f"Fallo crítico durante la ingesta: {e}", exc_info=True)
            print("\n[bold red]❌ Ocurrió un error inesperado. Revisa logs/app.log para más detalles.[/bold red]")
    
    elif args.mode == "chat":
        print("\n[INFO] Iniciando motor de Chat...\n")
        logger.info("Modo chat seleccionado. Levantando motor interactivo de Ollama.")
        
        try:
            start_interactive_chat(chroma_path=CHROMA_PATH)
            logger.info("Sesión de chat finalizada por el usuario de forma natural.")
        except Exception as e:
            logger.error(f"Fallo crítico en el motor de chat: {e}", exc_info=True)
            print("\n[bold red]❌ El motor de chat colapsó. Revisa logs/app.log para más detalles.[/bold red]")

if __name__ == "__main__":
    main()