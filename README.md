# fym-rag-cli

Un asistente de línea de comandos (CLI) basado en la arquitectura RAG (Retrieval-Augmented Generation) para consultar textos académicos de física y matemáticas de forma 100% local.

Este proyecto busca procesar libros densos, almacenarlos de manera vectorial y permitir al usuario hacer consultas teóricas mediante un chat interactivo en la terminal, aprovechando modelos de lenguaje (LLMs) open-source a través de Ollama.

## Estado del Proyecto
Actualmente en fase de Producto Mínimo Viable (MVP). El enfoque inicial está en el procesamiento de texto plano y almacenamiento en ChromaDB, con miras a escalar en el futuro hacia un RAG Visual capaz de interpretar ecuaciones y gráficos complejos.

## Estructura del Proyecto

La arquitectura inicial está modularizada de la siguiente manera para facilitar su ejecución en contenedores:

* **`data/`**: Directorio para almacenar los documentos académicos crudos (PDFs).
* **`chroma_db/`**: Base de datos vectorial embebida generada automáticamente.
* **`src/`**: Directorio principal del código fuente.
  * `ingest.py`: Script encargado de extraer, fragmentar (chunking) y vectorizar el texto de los PDFs.
  * `chat.py`: Bucle principal que gestiona la interfaz interactiva en la terminal y la comunicación con el LLM.
  * `prompts.py`: Plantillas e instrucciones de sistema defensivas para mitigar alucinaciones y ruido del OCR.
* **`dockerfile` / `docker-compose.yml`**: Configuración para aislar el entorno de Python manteniendo la ejecución de Ollama nativa en el host.
* **`requirements.txt`**: Dependencias clave del proyecto (`langchain`, `chromadb`, `pypdf`, `rich`, etc.).
* **`.env`**: Archivo para la gestión segura de variables de entorno e integraciones.

---
*Documentación en construcción. Las instrucciones de instalación y uso se añadirán en futuras actualizaciones.*