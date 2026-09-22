# Conceptos de LLM & RAG — fym-rag-cli

Inventario de todos los conceptos de inteligencia artificial y procesamiento de lenguaje natural que se están utilizando en este proyecto, junto con sugerencias de conceptos que podrían integrarse en el futuro.

---

## Conceptos Implementados Actualmente

### 1. RAG (Retrieval-Augmented Generation)
- **Qué es:** Arquitectura que combina la recuperación de información desde una base de conocimiento con la generación de texto de un LLM, para que las respuestas estén fundamentadas en documentos reales.
- **Dónde se usa:** Es la arquitectura central del proyecto. El flujo completo vive entre [`chat_engine.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/llm/chat_engine.py), [`retriever_builder.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/llm/retriever_builder.py) y [`chroma_manager.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/vector_db/chroma_manager.py).

---

### 2. Embeddings (Vectorización de Texto)
- **Qué es:** Representación numérica de texto en un espacio vectorial de alta dimensión. Textos semánticamente similares quedan cerca entre sí.
- **Modelo usado:** `nomic-embed-text` (servido localmente vía Ollama).
- **Dónde se usa:** [`chroma_manager.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/vector_db/chroma_manager.py#L18-L21) — `OllamaEmbeddings` genera los vectores tanto para la ingesta como para las queries.

---

### 3. Vector Store / Base de Datos Vectorial
- **Qué es:** Base de datos especializada en almacenar y buscar vectores (embeddings) mediante similitud semántica en lugar de coincidencia textual exacta.
- **Tecnología usada:** ChromaDB (embebida, persistente en disco).
- **Dónde se usa:** [`chroma_manager.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/vector_db/chroma_manager.py) — gestiona conexión, inserción por lotes y consultas.

---

### 4. Chunking (Fragmentación de Documentos)
- **Qué es:** Proceso de dividir documentos largos en fragmentos más pequeños y manejables para que el modelo de embeddings pueda procesarlos eficientemente.
- **Estrategia usada:** `RecursiveCharacterTextSplitter` con `chunk_size=1000` y `chunk_overlap=200`.
- **Dónde se usa:** [`text_splitter.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/chunking/text_splitter.py#L28-L34).

---

### 5. Similarity Search con Score Threshold (Retriever con Umbral)
- **Qué es:** Búsqueda por similitud semántica que solo devuelve resultados que superen un puntaje mínimo de confianza, actuando como cortafuegos de entrada para filtrar contenido irrelevante.
- **Configuración:** `search_type="similarity_score_threshold"`, `score_threshold=0.5`, `k=5`.
- **Dónde se usa:** [`retriever_builder.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/llm/retriever_builder.py#L8-L15) — el retriever base rechaza documentos con baja similitud.

---

### 6. Metadata Filtering (Filtrado por Metadatos)
- **Qué es:** Capacidad de filtrar documentos en la búsqueda vectorial usando atributos estructurados (metadatos) además de la similitud semántica.
- **Uso concreto:** Se excluyen fragmentos del prefacio/índice con `"filter": {"capitulo": {"$ne": "Prefacio_o_Indice"}}`.
- **Dónde se usa:** [`retriever_builder.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/llm/retriever_builder.py#L13) y [`text_splitter.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/chunking/text_splitter.py#L16-L25) (etiquetado de capítulos vía Regex).

---

### 7. History-Aware Retriever (Retriever con Memoria Conversacional)
- **Qué es:** Retriever que toma en cuenta el historial de la conversación para reformular la pregunta actual antes de buscar, permitiendo que preguntas de seguimiento como *"¿y su fórmula?"* se resuelvan correctamente.
- **Dónde se usa:** [`retriever_builder.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/llm/retriever_builder.py#L17-L23) — usa `create_history_aware_retriever` de LangChain.

---

### 8. Query Rephrasing / Reformulación de Consultas
- **Qué es:** Técnica donde el LLM reformula la pregunta del usuario para hacerla independiente del contexto conversacional previo, mejorando la calidad de la búsqueda vectorial.
- **Dónde se usa:** [`prompts.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/prompts/prompts.py#L1-L5) — `REPHRASE_TEMPLATE` instruye al modelo a reformular sin responder.

---

### 9. Stuff Documents Chain (Inyección de Contexto)
- **Qué es:** Estrategia de LangChain que "mete" (stuffs) todos los documentos recuperados directamente dentro del prompt del LLM como contexto. Es la más simple y funciona bien cuando el contexto combinado cabe en la ventana del modelo.
- **Dónde se usa:** [`chat_engine.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/llm/chat_engine.py#L39) — `create_stuff_documents_chain`.

---

### 10. System Prompt Engineering (Prompt Defensivo)
- **Qué es:** Diseño cuidadoso de instrucciones de sistema para controlar el comportamiento del LLM: limitar sus fuentes, definir formato de respuesta y forzar honestidad cuando no tiene información.
- **Dónde se usa:** [`prompts.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/prompts/prompts.py#L8-L21) — `PROMPT_TEMPLATE` con instrucciones estrictas de no usar conocimiento externo.

---

### 11. Guardrails / Anti-Alucinaciones (Validación Post-Generación)
- **Qué es:** Sistema de validación que compara la respuesta generada contra los documentos fuente usando similitud de coseno, interceptando y bloqueando respuestas que se desvíen demasiado del contexto original.
- **Umbral calibrado:** `0.77` (similitud coseno mínima).
- **Dónde se usa:** [`guardrails.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/llm/guardrails.py) — `validate_answer_with_cosine` usando `scikit-learn`.

---

### 12. Cosine Similarity (Similitud de Coseno)
- **Qué es:** Métrica matemática que mide el ángulo entre dos vectores. Un valor cercano a 1.0 indica alta similitud semántica; cercano a 0 indica baja relación.
- **Dónde se usa:** [`guardrails.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/llm/guardrails.py#L23) — compara embedding de la respuesta vs. embeddings de los fragmentos fuente.

---

### 13. Streaming de Respuestas
- **Qué es:** Generación incremental de texto del LLM, mostrando tokens al usuario conforme se producen en lugar de esperar la respuesta completa. Mejora la experiencia de usuario percibida.
- **Dónde se usa:** [`chat_engine.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/llm/chat_engine.py#L73-L77) — `qa_chain.stream(...)` con impresión token por token.

---

### 14. Conversational Memory (Memoria Conversacional)
- **Qué es:** Mecanismo para mantener un historial de mensajes (pares pregunta-respuesta) que se pasa al LLM en cada turno, permitiendo conversaciones de múltiples turnos con contexto.
- **Implementación:** Lista de objetos `HumanMessage` / `AIMessage` en memoria.
- **Dónde se usa:** [`chat_engine.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/llm/chat_engine.py#L41-L96) — `chat_history` se extiende en cada turno, incluyendo respuestas corregidas del guardián.

---

### 15. LLM-as-Judge (LLM como Juez)
- **Qué es:** Uso de un LLM para evaluar la calidad o fidelidad de la respuesta de otro LLM (o de sí mismo), dictaminando si una respuesta está respaldada por el contexto fuente.
- **Estado:** Prompt definido pero **no integrado activamente** en el flujo de chat.
- **Dónde se usa:** [`prompts.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/prompts/prompts.py#L24-L40) — `JUDGE_PROMPT_TEMPLATE` evalúa SI/NO si hay alucinación.

---

### 16. Batching (Inserción por Lotes)
- **Qué es:** Procesamiento de datos en lotes de tamaño fijo en lugar de todos a la vez, para controlar el uso de memoria y evitar desbordamientos durante la ingesta masiva de embeddings.
- **Configuración:** `batch_size=200` con `time.sleep(1)` entre lotes.
- **Dónde se usa:** [`chroma_manager.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/vector_db/chroma_manager.py#L25-L66) — `guardar_chunks_en_chroma`.

---

### 17. Temperature = 0 (Generación Determinista)
- **Qué es:** Parámetro que controla la aleatoriedad del LLM. Con `temperature=0`, el modelo siempre elige el token más probable, produciendo respuestas más consistentes y deterministas — ideal para un asistente factual.
- **Dónde se usa:** [`chat_engine.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/llm/chat_engine.py#L29) — `ChatOllama(temperature=0)`.

---

### 18. Local LLM Inference (Inferencia Local con Ollama)
- **Qué es:** Ejecución de modelos de lenguaje directamente en la máquina del usuario sin depender de APIs en la nube, garantizando privacidad y cero costos de API.
- **Modelos usados:** `qwen2.5:3b` (generación) + `nomic-embed-text` (embeddings).
- **Dónde se usa:** Todo el proyecto — Ollama como backend en `http://localhost:11434`.

---

## Conceptos Pendientes de Integrar

Los siguientes conceptos son candidatos naturales para mejorar la calidad, robustez y capacidades del sistema RAG:

| # | Concepto | Qué aporta | Prioridad |
|---|---------|-------------|-----------|
| 1 | **Reranking** (Cross-Encoder) | Reordena los documentos recuperados usando un modelo más preciso (ej: `ms-marco-MiniLM`), mejorando la relevancia del top-K antes de pasarlo al LLM. | 🔴 Alta |
| 2 | **Hybrid Search** (BM25 + Vectorial) | Combina búsqueda léxica (palabras clave) con semántica (vectores) para capturar tanto coincidencias exactas de fórmulas/variables como similitud conceptual. | 🔴 Alta |
| 3 | **LLM-as-Judge (Activación)** | Ya tienes el prompt `JUDGE_PROMPT_TEMPLATE` definido. Activarlo como segundo filtro anti-alucinaciones (doble validación: coseno + juez LLM). | 🟡 Media |
| 4 | **Multi-Query Retrieval** | Genera múltiples variantes de la pregunta del usuario para buscar desde distintos ángulos semánticos, aumentando la cobertura de documentos relevantes. | 🟡 Media |
| 5 | **Contextual Compression** | Comprime los documentos recuperados para extraer solo las oraciones relevantes a la pregunta, reduciendo ruido en el contexto del LLM. | 🟡 Media |
| 6 | **Semantic Chunking** | En lugar de cortar por caracteres fijos, detecta fronteras semánticas naturales (cambios de tema) para crear chunks más coherentes. | 🟡 Media |
| 7 | **Parent-Child Document Retrieval** | Almacena chunks pequeños para búsqueda precisa pero recupera el documento padre (más amplio) para darle más contexto al LLM. | 🟡 Media |
| 8 | **RAG Fusion** | Combina resultados de múltiples queries con Reciprocal Rank Fusion (RRF) para obtener un ranking final más robusto. | 🟢 Baja |
| 9 | **Self-RAG / Corrective RAG** | El modelo evalúa si necesita buscar más información y puede autocorregir su respuesta iterativamente. | 🟢 Baja |
| 10 | **Agentic RAG** | Transforma el sistema en un agente que decide cuándo buscar, qué herramienta usar y cuándo ya tiene suficiente información para responder. | 🟢 Baja |
| 11 | **Fine-tuning / LoRA** | Ajustar el modelo base con datos específicos del dominio (física) para mejorar la calidad de las respuestas sin aumentar el contexto. | 🟢 Baja |
| 12 | **RAG Visual (Multimodal)** | Procesar imágenes, ecuaciones y gráficos del PDF con modelos de visión (ej: LLaVA) — ya mencionado en el README como objetivo futuro. | 🟢 Baja |
| 13 | **Evaluation Framework** (RAGAS) | Framework para evaluar automáticamente la calidad del RAG con métricas como *faithfulness*, *answer relevancy* y *context precision*. | 🟡 Media |
| 14 | **Persistent Conversational Memory** | Guardar el historial de chat en disco/BD para retomar conversaciones entre sesiones. | 🟢 Baja |
| 15 | **Chain of Thought (CoT) Prompting** | Instruir al LLM a razonar paso a paso antes de dar la respuesta final, mejorando la precisión en problemas de física. | 🟡 Media |

---

## Mapa de Flujo Actual

```
PDF
 │
 ▼
┌──────────────┐    ┌───────────────┐    ┌─────────────┐
│  PDF Loader  │───▶│   Chunking    │───▶│  Embeddings │
│  (PyPDF)     │    │ (Recursive)   │    │ (nomic)     │
└──────────────┘    └───────────────┘    └──────┬──────┘
                                                │
                                                ▼
                                        ┌──────────────┐
                                        │  ChromaDB    │
                                        │ (Vector DB)  │
                                        └──────┬───────┘
                                               │
        ┌──────────────────────────────────────┘
        │
        ▼
┌───────────────┐    ┌──────────────┐    ┌─────────────┐
│  History-     │───▶│   Stuff      │───▶│  Streaming  │
│  Aware        │    │   Chain      │    │  Output     │
│  Retriever    │    │  (LLM Gen)   │    │             │
└───────────────┘    └──────────────┘    └──────┬──────┘
        ▲                                       │
        │                                       ▼
┌───────────────┐                       ┌──────────────┐
│  Query        │                       │  Guardrails  │
│  Rephrasing   │                       │  (Cosine)    │
└───────────────┘                       └──────┬───────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │  Respuesta   │
                                        │  Validada    │
                                        └──────────────┘
```
