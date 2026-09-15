# fym-rag-cli

Un asistente de línea de comandos (CLI) basado en la arquitectura RAG (Retrieval-Augmented Generation) para consultar textos académicos de física y matemáticas de forma 100% local.

Este proyecto busca procesar libros densos, almacenarlos de manera vectorial y permitir al usuario hacer consultas teóricas mediante un chat interactivo en la terminal, aprovechando modelos de lenguaje (LLMs) open-source a través de Ollama.

## Estado del Proyecto y Novedades
Actualmente en fase de Producto Mínimo Viable (MVP) altamente calibrado. El enfoque inicial está en el procesamiento de texto plano y almacenamiento en ChromaDB, con miras a escalar en el futuro hacia un RAG Visual capaz de interpretar ecuaciones y gráficos complejos.

**Últimas implementaciones destacadas:**
* **Sistema Anti-Alucinaciones (Doble Filtro):** Implementación de un cortafuegos de entrada (umbral estricto en el Retriever) y un guardián de salida basado en similitud de coseno (`scikit-learn`) calibrado empíricamente a 0.78 para bloquear respuestas fuera del contexto del libro.
* **Memoria Conversacional Inteligente:** Capacidad de recordar el contexto de la charla, con un mecanismo de limpieza automática (`chat_history.clear()`) cuando el sistema detecta intentos de alucinación, evitando la contaminación a futuro.
* **Modo Rayos X (Debug):** Visualización en tiempo real de los fragmentos exactos recuperados, sus puntajes de similitud y páginas de origen directamente en la terminal usando `rich`.

## Estructura del Proyecto

La arquitectura inicial está modularizada de la siguiente manera para facilitar su ejecución local y en contenedores:

* **`data/`**: Directorio para almacenar los documentos académicos crudos (PDFs).
* **`chroma_db/`**: Base de datos vectorial embebida generada automáticamente.
* **`src/`**: Directorio principal del código fuente.
  * `ingest.py`: Script encargado de extraer, fragmentar (chunking), etiquetar capítulos con Regex, e insertar vectores por lotes (batching) en ChromaDB para proteger la RAM.
  * `chat.py`: Bucle principal que gestiona la interfaz interactiva en la terminal, el doble filtro semántico y la comunicación con el LLM.
  * `prompts.py`: Plantillas e instrucciones de sistema defensivas para mitigar alucinaciones y ruido del OCR.
* **`dockerfile` / `docker-compose.yml`**: Configuración para aislar el entorno de Python manteniendo la ejecución de Ollama nativa en el host.
* **`requirements.txt`**: Dependencias clave del proyecto (`langchain`, `chromadb`, `pypdf`, `rich`, `scikit-learn`, etc.).
* **`.env`**: Archivo para la gestión segura de variables de entorno e integraciones (ignorado en Git por seguridad).

---

## Guía de Instalación y Uso Local

Para levantar el proyecto en tu entorno local usando Ollama, sigue estos pasos:

### 1. Preparar los Modelos en Ollama
Asegúrate de tener [Ollama](https://ollama.com/) instalado en tu sistema. Necesitaremos dos modelos: uno para generación de texto y otro para los embeddings matemáticos.

Abre una terminal y descarga los modelos:
```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

### 2. Iniciar el Motor de Ollama
Asegúrate de que el servicio de Ollama esté corriendo en el fondo. Si usas Mac/Windows con la aplicación de escritorio, basta con tenerla abierta. Si estás en Linux o prefieres la terminal, ejecuta:
```bash
ollama serve
```

### 3. Configurar el Entorno Virtual (Python)
Abre otra pestaña en tu terminal, sitúate en la raíz del proyecto y prepara el entorno de dependencias:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Ingesta de Datos (Vectorización del Libro)
Antes de chatear, el sistema necesita leer el PDF (que debe estar en la carpeta data/) y generar la base de datos vectorial en chroma_db/. Este proceso requiere algo de RAM y toma unos minutos:
```bash
python src/ingest.py
```
**Nota**: este paso se hace solo 1 vez por cada PDF adjuntado. Si el proceso falla por algun motivo, borrar la carpeta de chroma_db y ejecutar de nuevo ingest.py.

### 5. Iniciar el Asistente
Una vez completada la ingesta, puedes arrancar la interfaz de línea de comandos interactiva:

```bash
python src/chat.py
```
