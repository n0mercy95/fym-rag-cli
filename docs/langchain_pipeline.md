# LangChain: Del Pregunta a la Respuesta

Guía técnica que desglosa **paso a paso** cómo LangChain orquesta el viaje completo de una pregunta del usuario hasta la respuesta validada en `fym-rag-cli`. Cada paso está mapeado al archivo y función exacta del proyecto donde ocurre.

---

## Visión General del Flujo

```
 ┌─────────────────────────────────────────────────────────────────────────────────────┐
 │                        EL VIAJE DE UNA PREGUNTA                                    │
 │                                                                                    │
 │   Usuario                                                                          │
 │     │                                                                              │
 │     │  "¿Qué es la aceleración?"                                                   │
 │     ▼                                                                              │
 │   ┌──────────────────────┐                                                         │
 │   │  1. REFORMULACIÓN    │  LangChain reformula la pregunta usando el historial     │
 │   │     (Query Rephrase) │  para hacerla independiente del contexto previo          │
 │   └──────────┬───────────┘                                                         │
 │              ▼                                                                     │
 │   ┌──────────────────────┐                                                         │
 │   │  2. RECUPERACIÓN     │  Búsqueda híbrida: BM25 (léxica) + Vectorial            │
 │   │     HÍBRIDA          │  (semántica) combinadas 50/50                            │
 │   └──────────┬───────────┘                                                         │
 │              ▼                                                                     │
 │   ┌──────────────────────┐                                                         │
 │   │  3. RERANKING        │  Cross-Encoder reordena y filtra a los                   │
 │   │     (Compresión)     │  3 mejores fragmentos                                   │
 │   └──────────┬───────────┘                                                         │
 │              ▼                                                                     │
 │   ┌──────────────────────┐                                                         │
 │   │  4. GENERACIÓN       │  El LLM recibe los docs + historial + pregunta           │
 │   │     (Stuff Chain)    │  y genera la respuesta                                  │
 │   └──────────┬───────────┘                                                         │
 │              ▼                                                                     │
 │   ┌──────────────────────┐                                                         │
 │   │  5. VALIDACIÓN       │  Similitud de coseno verifica que la respuesta            │
 │   │     (Guardrails)     │  esté anclada al contexto original                      │
 │   └──────────┬───────────┘                                                         │
 │              ▼                                                                     │
 │   Respuesta validada → Usuario                                                     │
 └─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Paso 0: La Infraestructura que LangChain Levanta al Iniciar

Antes de que el usuario escriba una sola palabra, LangChain ya construyó toda la maquinaria. Esto ocurre **una sola vez** al iniciar el servicio.

### ¿Qué se inicializa?

| Componente | Clase de LangChain | Archivo |
|---|---|---|
| Conexión al LLM | `ChatOllama` | [`services.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/api/services.py#L23) |
| Embeddings | `OllamaEmbeddings` | [`chroma_manager.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/vector_db/chroma_manager.py#L17-L20) |
| Base vectorial | `Chroma` | [`chroma_manager.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/vector_db/chroma_manager.py#L22) |
| Pipeline de recuperación | `create_history_aware_retriever` | [`retriever_builder.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/llm/retriever_builder.py#L90-L92) |
| Cadena de generación | `create_stuff_documents_chain` | [`services.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/api/services.py#L33) |

### ¿Por qué importa?

LangChain permite **componer** estos componentes como piezas de LEGO. El `history_aware_retriever` envuelve al `compression_retriever`, que envuelve al `hybrid_retriever`, que combina `BM25Retriever` + `vector_retriever`. Toda esta cadena se construye con una sola llamada y después se invoca como si fuera una sola función.

---

## Paso 1: Reformulación de la Pregunta (Query Rephrasing)

> **Problema que resuelve:** El usuario pregunta _"¿y su fórmula?"_ — sin contexto previo, ChromaDB no tiene idea de qué fórmula buscar.

### ¿Cómo interviene LangChain?

LangChain usa `create_history_aware_retriever` para envolver el retriever con una capa de "memoria". Antes de buscar, le pasa al LLM el **historial de conversación** junto con la pregunta actual, y le pide que la **reformule** como una consulta autónoma de búsqueda.

### El prompt de reformulación

Definido en [`prompts.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/prompts/prompts.py#L2-L15):

```python
REPHRASE_TEMPLATE = """
Dada la siguiente historia de conversación y la pregunta final del usuario, 
reformula la pregunta para que sea una consulta de búsqueda óptima...

Historial de conversación:
{chat_history}

Pregunta del usuario: 
{input}

REGLA: Añade términos como "definición", "concepto teórico" o "explicación"...

Pregunta reformulada para buscar:
"""
```

### Ejemplo concreto

| Turno | Pregunta original | Pregunta reformulada (por el LLM) |
|---|---|---|
| 1 | "¿Qué es la aceleración?" | "definición concepto teórico aceleración física" |
| 2 | "¿Y su fórmula?" | "fórmula ecuación aceleración definición" |

Sin esta capa de LangChain, la pregunta 2 buscaría literalmente _"¿Y su fórmula?"_ en la base de datos y no encontraría nada útil.

### ¿Dónde se construye?

En [`retriever_builder.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/llm/retriever_builder.py#L88-L92):

```python
# 4. Capa Conversacional
rephrase_prompt = PromptTemplate.from_template(REPHRASE_TEMPLATE)
history_aware_retriever = create_history_aware_retriever(
    llm, compression_retriever, rephrase_prompt
)
```

---

## Paso 2: Recuperación Híbrida (Retrieval)

> **Problema que resuelve:** La búsqueda puramente semántica puede fallar con fórmulas o términos técnicos exactos. La búsqueda puramente léxica no entiende sinónimos.

### ¿Cómo interviene LangChain?

LangChain proporciona tres abstracciones clave que se combinan:

#### 2a. Retriever Vectorial (Semántico)

Busca por **significado**. "Cambio de velocidad" encuentra documentos sobre "aceleración" aunque no contengan esa palabra exacta.

```python
# retriever_builder.py — Línea 50-53
vector_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 15}
)
```

`vectorstore.as_retriever()` es un método de LangChain que convierte cualquier Vector Store (en nuestro caso, `Chroma`) en un objeto `Retriever` compatible con el resto de la cadena.

#### 2b. Retriever BM25 (Léxico)

Busca por **coincidencia de palabras** con ponderación estadística. Encuentra documentos que contengan exactamente "F = ma".

```python
# retriever_builder.py — Líneas 61-68
bm25_retriever = BM25Retriever.from_documents(docs_para_bm25)
bm25_retriever.k = 15
```

`BM25Retriever` es una clase de `langchain_community` que implementa el algoritmo BM25 (Best Matching 25) sin necesidad de un servidor externo.

#### 2c. Fusión con EnsembleRetriever

LangChain fusiona ambos en uno solo con `EnsembleRetriever`:

```python
# retriever_builder.py — Líneas 71-74
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)
```

Los `weights=[0.5, 0.5]` significan que ambas estrategias tienen **la misma importancia** en el resultado final. LangChain se encarga de normalizar y fusionar los rankings internamente.

### Resultado de este paso

Se obtienen hasta **30 fragmentos candidatos** (15 de cada retriever, con posibles duplicados eliminados por la fusión).

---

## Paso 3: Reranking con Cross-Encoder

> **Problema que resuelve:** De los ~30 candidatos, muchos son marginalmente relevantes. Necesitamos quedarnos solo con los **3 mejores**.

### ¿Cómo interviene LangChain?

LangChain provee `ContextualCompressionRetriever`, un wrapper que toma cualquier retriever base y le aplica un "compresor" antes de devolver resultados.

```python
# retriever_builder.py — Líneas 79-86
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=model, top_n=3)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=hybrid_retriever
)
```

### ¿Qué hace el Cross-Encoder?

A diferencia de los embeddings (que comparan vectores precalculados), el Cross-Encoder recibe el **par (pregunta, documento)** juntos y calcula un score de relevancia mucho más preciso. Es más lento, por eso solo lo usamos sobre los ~30 candidatos finales, no sobre toda la base de datos.

### Rol de LangChain aquí

LangChain abstrae la complejidad de:
1. Recibir los documentos del retriever híbrido
2. Pasarlos uno por uno al Cross-Encoder junto con la pregunta
3. Ordenarlos por score descendente
4. Devolver solo los `top_n=3` mejores

Todo esto con una interfaz idéntica a cualquier otro retriever: `.invoke()`.

---

## Paso 4: Generación de la Respuesta (LLM Generation)

> **Problema que resuelve:** Tenemos 3 fragmentos relevantes, pero necesitamos una respuesta coherente en lenguaje natural.

### ¿Cómo interviene LangChain?

LangChain orquesta la generación con dos componentes:

#### 4a. ChatPromptTemplate — Estructura del prompt

```python
# services.py — Líneas 28-32
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", PROMPT_TEMPLATE),    # Instrucciones de comportamiento
    MessagesPlaceholder("chat_history"),  # Historial de conversación
    ("human", "{input}")            # La pregunta actual
])
```

`ChatPromptTemplate` de LangChain permite construir prompts **multi-turno** con roles (`system`, `human`, `assistant`). `MessagesPlaceholder` inserta dinámicamente el historial completo de la conversación como mensajes individuales, no como texto plano.

#### 4b. Stuff Documents Chain — Inyección de contexto

```python
# services.py — Línea 33
qa_chain = create_stuff_documents_chain(llm, qa_prompt)
```

`create_stuff_documents_chain` es una función factory de LangChain que:
1. Toma los documentos recuperados (paso 3)
2. Los **concatena** en un solo bloque de texto
3. Los inyecta dentro de la variable `{context}` del prompt
4. Envía todo al LLM

#### 4c. Invocación

```python
# services.py — Líneas 62-66
respuesta_completa = self.qa_chain.invoke({
    "context": docs, 
    "input": query, 
    "chat_history": chat_history
})
```

LangChain se encarga de:
- Formatear los `Document` objects como texto legible
- Insertar los mensajes del historial en el orden correcto
- Manejar la comunicación HTTP con Ollama
- Devolver la respuesta como string

### El Prompt Defensivo

El `PROMPT_TEMPLATE` en [`prompts.py`](file:///Users/matias95lopez/Desktop/fym-rag-cli/src/prompts/prompts.py#L18-L34) es crucial. Le dice al LLM:

- ✅ Usar **exclusivamente** el contexto proporcionado
- ❌ **Prohibido** inventar ejemplos, fórmulas o información externa
- 🛑 Si no tiene info suficiente, debe decir exactamente: _"No tengo suficiente información"_

---

## Paso 5: Validación Post-Generación (Guardrails)

> **Problema que resuelve:** Incluso con un buen prompt, el LLM puede alucinar. Necesitamos una **verificación matemática** de la respuesta.

### ¿Cómo interviene LangChain?

Aquí LangChain participa **indirectamente**: los embeddings del vector store (creados con LangChain via `OllamaEmbeddings`) se reutilizan para validar la respuesta.

```python
# guardrails.py — Líneas 21-23
vec_respuesta = embeddings.embed_query(respuesta_completa)
textos_contexto = [doc.page_content for doc in docs]
vecs_contexto = embeddings.embed_documents(textos_contexto)
```

Los objetos `embeddings` y `docs` (tipo `Document` de LangChain) fluyen desde la cadena RAG hasta el módulo de guardrails, manteniendo una interfaz consistente.

### La lógica de validación

```python
# guardrails.py — Líneas 25-31
UMBRAL_COSENO = 0.7

for vec_ctx in vecs_contexto:
    score = cosine_similarity([vec_ctx], [vec_respuesta])[0][0]
    if score > max_score:
        max_score = score

es_valida = max_score >= UMBRAL_COSENO
```

Si el **mejor score de similitud** entre la respuesta y cualquier fragmento fuente es menor a `0.7`, la respuesta se considera una **alucinación** y se reemplaza por el mensaje seguro.

### Resultado

| Score | Resultado | Acción |
|---|---|---|
| `>= 0.7` | ✅ Respuesta válida | Se entrega al usuario y se guarda en el historial |
| `< 0.7` | ⚠️ Alucinación detectada | Se reemplaza por _"No tengo suficiente información"_ |

---

## Paso 6: Gestión de Memoria Conversacional

> **Problema que resuelve:** Necesitamos que el sistema recuerde qué se preguntó antes, pero sin contaminar la memoria con alucinaciones.

### ¿Cómo interviene LangChain?

LangChain proporciona los tipos `HumanMessage` y `AIMessage` de `langchain_core.messages` para representar el historial de forma estructurada:

```python
# services.py — Línea 101
chat_history.extend([
    HumanMessage(content=query), 
    AIMessage(content=respuesta_completa)
])
```

### Estrategia anti-contaminación

Si el guardrail intercepta una alucinación, **no se borra la memoria**, sino que se guarda la respuesta corregida:

```python
# services.py — Línea 78
chat_history.extend([
    HumanMessage(content=query), 
    AIMessage(content=respuesta_segura)  # "No tengo suficiente información"
])
```

Esto le enseña al modelo (vía historial) que para esa pregunta específica, no tenía información — evitando que reinvente la respuesta en turnos futuros.

---

## Resumen: Los 6 Módulos de LangChain en Acción

| Paso | Módulo de LangChain | Clase/Función Principal | Qué Hace |
|---|---|---|---|
| **0** | Inicialización | `ChatOllama`, `Chroma`, `OllamaEmbeddings` | Levanta conexiones al LLM, embeddings y vector store |
| **1** | Reformulación | `create_history_aware_retriever` | Transforma preguntas ambiguas en queries de búsqueda autónomas |
| **2** | Recuperación | `EnsembleRetriever`, `BM25Retriever`, `.as_retriever()` | Búsqueda híbrida semántica + léxica con fusión 50/50 |
| **3** | Reranking | `ContextualCompressionRetriever`, `CrossEncoderReranker` | Filtra los 3 mejores documentos con un modelo de precisión |
| **4** | Generación | `ChatPromptTemplate`, `create_stuff_documents_chain` | Inyecta contexto + historial en el prompt y genera la respuesta |
| **5** | Validación | `OllamaEmbeddings` (reutilizado) | Verifica la respuesta contra los documentos fuente con coseno |
| **6** | Memoria | `HumanMessage`, `AIMessage` | Mantiene el historial limpio para conversaciones multi-turno |

---

## Diagrama Completo: LangChain como Orquestador

```
                            ┌──────────────────────────────────┐
                            │         USUARIO                  │
                            │  "¿Qué es la aceleración?"       │
                            └──────────────┬───────────────────┘
                                           │
                    ┌──────────────────────────────────────────────────┐
                    │          LANGCHAIN: CAPA DE ORQUESTACIÓN         │
                    │                                                  │
                    │  ┌────────────────────────────────────────────┐  │
                    │  │ create_history_aware_retriever             │  │
                    │  │                                            │  │
                    │  │  chat_history + input                      │  │
                    │  │       │                                    │  │
                    │  │       ▼                                    │  │
                    │  │  ┌─────────────────────┐                  │  │
                    │  │  │ LLM: Reformula      │                  │  │
                    │  │  │ la pregunta         │                  │  │
                    │  │  └─────────┬───────────┘                  │  │
                    │  │            ▼                               │  │
                    │  │  ┌─────────────────────────────────────┐  │  │
                    │  │  │ ContextualCompressionRetriever      │  │  │
                    │  │  │                                     │  │  │
                    │  │  │  ┌───────────────────────────────┐  │  │  │
                    │  │  │  │ EnsembleRetriever             │  │  │  │
                    │  │  │  │                               │  │  │  │
                    │  │  │  │  BM25 (léxico)    0.5 ──┐     │  │  │  │
                    │  │  │  │                         ├─►   │  │  │  │
                    │  │  │  │  Chroma (semántico) 0.5─┘     │  │  │  │
                    │  │  │  │           ~30 candidatos       │  │  │  │
                    │  │  │  └──────────────┬────────────────┘  │  │  │
                    │  │  │                 ▼                    │  │  │
                    │  │  │  ┌─────────────────────────────┐    │  │  │
                    │  │  │  │ CrossEncoderReranker        │    │  │  │
                    │  │  │  │ top_n=3 mejores documentos  │    │  │  │
                    │  │  │  └─────────────────────────────┘    │  │  │
                    │  │  └─────────────────────────────────────┘  │  │
                    │  └────────────────────────────────────────────┘  │
                    │                       │                          │
                    │                       ▼  3 documentos           │
                    │  ┌────────────────────────────────────────────┐  │
                    │  │ create_stuff_documents_chain               │  │
                    │  │                                            │  │
                    │  │  System Prompt (defensivo)                 │  │
                    │  │  + chat_history (HumanMessage/AIMessage)   │  │
                    │  │  + context (3 docs concatenados)           │  │
                    │  │  + input (pregunta original)               │  │
                    │  │       │                                    │  │
                    │  │       ▼                                    │  │
                    │  │  ┌─────────────────────┐                  │  │
                    │  │  │ LLM: Genera         │                  │  │
                    │  │  │ respuesta            │                  │  │
                    │  │  └─────────┬───────────┘                  │  │
                    │  └────────────┼───────────────────────────────┘  │
                    └───────────────┼──────────────────────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────────┐
                    │  GUARDRAILS (scikit-learn)        │
                    │                                  │
                    │  OllamaEmbeddings.embed_query()  │  ◄── LangChain provee
                    │  cosine_similarity()             │      los embeddings
                    │                                  │
                    │  score >= 0.7 → ✅ Válida        │
                    │  score <  0.7 → ⚠️ Alucinación   │
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────┐
                    │         RESPUESTA FINAL          │
                    │  → Se entrega al usuario         │
                    │  → Se guarda en chat_history      │
                    └──────────────────────────────────┘
```

---

## Conclusión

LangChain no es el LLM ni la base de datos — es el **director de orquesta** que conecta todas las piezas. Sin LangChain, tendríamos que implementar manualmente:

- La reformulación de preguntas con historial
- La fusión de múltiples estrategias de búsqueda
- El pipeline de reranking
- La inyección de documentos en prompts multi-turno
- La gestión de tipos de mensajes para la memoria

Con LangChain, cada uno de estos pasos es **una clase** o **una función factory** que se compone con las demás como piezas de LEGO, permitiendo cambiar cualquier componente (modelo, base de datos, estrategia de búsqueda) sin reescribir el resto del sistema.
