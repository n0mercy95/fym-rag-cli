# fym-rag-cli

Un asistente de línea de comandos (CLI) y API REST basado en la arquitectura RAG (Retrieval-Augmented Generation) para consultar textos académicos de física y matemáticas de forma 100% local.

Este proyecto busca procesar libros densos, almacenarlos de manera vectorial y permitir al usuario hacer consultas teóricas mediante un chat interactivo en la terminal o vía peticiones HTTP, aprovechando modelos de lenguaje (LLMs) open-source a través de Ollama.

## Estado del Proyecto y Novedades
Actualmente en fase de Producto Mínimo Viable (MVP) altamente calibrado. El enfoque inicial está en el procesamiento de texto plano y almacenamiento en ChromaDB, con miras a escalar en el futuro hacia un RAG Visual capaz de interpretar ecuaciones y gráficos complejos.

**Últimas implementaciones destacadas:**
* **Búsqueda Avanzada (Reranker):** Combinación de búsqueda vectorial y léxica (BM25), filtrada por un Cross-Encoder (`BAAI/bge-reranker-base`) que discrimina la teoría pura de los ejercicios matemáticos.
* **Sistema Anti-Alucinaciones (Doble Filtro):** Implementación de un cortafuegos de entrada (*Negative Prompting* estricto) y un guardián de salida basado en similitud de coseno (`scikit-learn`) calibrado empíricamente a 0.65 para bloquear respuestas fuera del contexto del libro permitiendo paráfrasis didáctica.
* **Memoria Conversacional Inteligente:** Capacidad de recordar el contexto de la charla, guardando las intercepciones del guardián en el historial para evitar contaminación sin causar amnesia a corto plazo.
* **Arquitectura API:** Migración estructural a FastAPI para ofrecer los servicios RAG a través de endpoints RESTful.

## Estructura del Proyecto

La arquitectura inicial está modularizada y orientada al dominio para facilitar su ejecución local, pruebas unitarias y su despliegue como API:

* **`main.py`**: Enrutador y punto de entrada principal (CLI) del proyecto.
* **`data/`**: Directorio para almacenar los documentos académicos crudos (PDFs).
* **`chroma_db/`**: Base de datos vectorial embebida generada automáticamente.
* **`logs/`**: Directorio autogenerado que almacena los registros de eventos (`app.log`) y errores del sistema.
* **`src/`**: Directorio principal del código fuente refactorizado.
  * `api/`: Controladores y rutas del servidor FastAPI (`server.py`, `routes.py`, `services.py`).
  * `chunking/`: Lógica para extraer, etiquetar (Regex) y fragmentar el texto.
  * `config/`: Configuraciones globales, incluyendo el sistema de *logging* (`logger.py`).
  * `ingestion/`: Orquestador del flujo de procesamiento de nuevos documentos.
  * `llm/`: Motor de chat interactivo, *retriever* semántico avanzado y filtros de validación (*guardrails*).
  * `prompts/`: Plantillas e instrucciones de sistema estrictas para el LLM.
  * `utils/`: Funciones auxiliares y herramientas genéricas.
  * `vector_db/`: Gestión de la conexión a ChromaDB y vectorización por lotes protegida contra desbordamientos de RAM.
* **`dockerfile` / `docker-compose.yml`**: Configuración para aislar el entorno de Python manteniendo la ejecución de Ollama nativa en el host.
* **`requirements.txt`**: Dependencias clave del proyecto (`fastapi`, `langchain`, `chromadb`, `sentence-transformers`, `scikit-learn`, etc.).
* **`.env`**: Archivo para la gestión segura de variables de entorno e integraciones.
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
Antes de chatear, el sistema necesita leer el PDF (que debe estar en la carpeta data/) y generar la base de datos vectorial en chroma_db/. Este proceso requiere algo de RAM y toma unos minutos. Utilizamos el enrutador principal con el modo de ingesta:

```bash
python main.py --mode ingest
```

**Nota**: Este paso se hace solo 1 vez por cada PDF adjuntado. Si el proceso falla por algún motivo, borra la carpeta chroma_db/ y vuelve a ejecutar el comando.

### 5. Iniciar el Asistente
Una vez completada la ingesta, puedes arrancar la interfaz interactiva delegando la ejecución al modo chat:

```bash
python main.py --mode chat
```

**Opcional**: para ver todos los comandos disponibles:

```bash
python main.py --help
```

### 6. Levantar el Servidor Web
Si prefieres interactuar con el sistema a través de peticiones HTTP (ideal para conectar un frontend o usar Postman), levanta el servidor integrado:

```bash
uvicorn src.api.server:app --reload
```

El servidor quedará escuchando en http://127.0.0.1:8000. Puedes ver la documentación interactiva (Swagger) visitando http://127.0.0.1:8000/docs.

### 7. Consultar la API (Postman / cURL)
7.1 Ingesta de Datos (POST /api/v1/ingest)
Para vectorizar un documento a través de la API, debes enviar el archivo PDF directamente utilizando el formato multipart/form-data:

Opción A: Usando Postman

Ve a la pestaña Body y selecciona la opción form-data (evita raw o JSON).

En la columna Key, escribe file. Pasa el ratón sobre esa celda y cambia el tipo de "Text" a "File".

En la columna Value, haz clic en el botón "Select Files" y elige tu documento PDF desde el explorador de archivos.

Opción B: Usando cURL
Asegúrate de reemplazar la ruta después del @ con la ubicación real de tu archivo si no estás ejecutando el comando desde la raíz del proyecto.

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/v1/ingest' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@./data/Sears_Zemansky_F_sica_Universitaria_Vol_1.pdf'
```

7.2 Chatear con el Documento (POST /api/v1/chat)
Envía tus consultas teóricas manteniendo un ID de sesión para la memoria conversacional:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/v1/chat' \
  -H 'Content-Type: application/json' \
  -d '{
        "session_id": "mi_sesion_estudio",
        "query": "hola, dime qué es el momento de inercia"
      }'
```

Respuesta Exitosa Esperada (puede variar):

{ \
    "answer": "El momento de inercia es una medida física que indica cuán difícil es cambiar el estado de rotación de un cuerpo. Es una propiedad del cuerpo y depende tanto de su forma como de la distribución de masa dentro del cuerpo. Cuanto mayor sea el valor del momento de inercia, más resistente será al cambio en su estado de rotación.\n\nEl momento de inercia se puede expresar como una suma para todas las partículas que componen el cuerpo, cada una de las cuales está a una distancia perpendicular al eje de rotación. Este concepto es crucial en la comprensión del movimiento rotacional y juega un papel importante en muchas aplicaciones físicas, incluyendo la mecánica de fluidos y la ingeniería.\n\nEn términos matemáticos, el momento de inercia I de un cuerpo alrededor de un eje dado se expresa como una suma de los productos de las masas de cada particula por sus respectivas distancias al eje.", \
    "hallucination_intercepted": false, \
    "cosine_score": 0.9242213791825862 \
}
