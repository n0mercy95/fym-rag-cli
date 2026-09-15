import os
import time
from sklearn.metrics.pairwise import cosine_similarity

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

from rich.console import Console
from rich.prompt import Prompt

from prompts import REPHRASE_TEMPLATE, PROMPT_TEMPLATE, JUDGE_PROMPT_TEMPLATE

# Rutas Absolutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

def main():
    console = Console()
    console.print("[bold green]⚛️ Iniciando Asistente de Física RAG (Modo Debug con Rayos X + Doble Filtro)...[/bold green]")

    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # Retriever estricto con umbral y filtro de metadatos
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.5, 
            "k": 5,
            "filter": {"capitulo": {"$ne": "Prefacio_o_Indice"}}
        }
    )
    
    llm = ChatOllama(model="qwen2.5:3b", base_url="http://localhost:11434", temperature=0)
    
    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", REPHRASE_TEMPLATE),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, rephrase_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT_TEMPLATE),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    chat_history = []

    while True:
        query_text = Prompt.ask("\n[bold blue]Haz una pregunta (o escribe 'salir')[/bold blue]")
        if query_text.lower() in ["salir", "exit", "quit"]:
            console.print("[bold red]¡Hasta luego![/bold red]")
            break

        console.print("\n[bold yellow]🧠 Consultando ChromaDB (buscando entre los 4000 fragmentos)...[/bold yellow]")
        
        # PASO 1: Buscar documentos (Filtro de Entrada / Retriever)
        docs = history_aware_retriever.invoke({
            "input": query_text,
            "chat_history": chat_history
        })

        # PASO 2: EL CORTAFUEGOS DE ENTRADA (Si el umbral los rechaza, frena aquí)
        if not docs:
            respuesta_fallida = "No tengo suficiente información en el texto para responder esto."
            console.print(f"\n[bold red]Respuesta:[/bold red]\n{respuesta_fallida}\n")
            chat_history.extend([
                HumanMessage(content=query_text),
                AIMessage(content=respuesta_fallida)
            ])
            continue

        # PASO 3: RAYOS X - Imprimir exactamente qué texto encontró ChromaDB
        console.print("\n[magenta]--- [DEBUG] TEXTO REAL RECUPERADO DE LA BD ---[/magenta]")
        for i, doc in enumerate(docs):
            console.print(f"[dim]Fragmento {i+1} | Origen: {doc.metadata.get('capitulo')} (Pág {doc.metadata.get('page')})[/dim]")
            console.print(f"[dim]{doc.page_content[:200]}...[/dim]\n")
        console.print("[magenta]--------------------------------------------[/magenta]\n")

        # PASO 4: Generar respuesta preliminar del modelo
        console.print("[bold green]Respuesta del Modelo:[/bold green]")
        
        respuesta_completa = ""
        for chunk in qa_chain.stream({
            "context": docs,
            "input": query_text,
            "chat_history": chat_history
        }):
            texto_chunk = chunk if isinstance(chunk, str) else str(chunk)
            print(texto_chunk, end="", flush=True)
            respuesta_completa += texto_chunk

        print("\n")

        # PASO 5: EL GUARDIÁN DE SALIDA (Doble Filtro por Similitud de Coseno)
        console.print("[dim]⏳ Cambiando motores en Ollama y calculando validación matemática...[/dim]")
        time.sleep(1) # Le damos 1 segundo a la Mac para liberar RAM del LLM y cargar nomic-embed-text
        
        # 1. Vectorizamos la respuesta del modelo por separado
        vec_respuesta = embeddings.embed_query(respuesta_completa)
        
        # 2. Extraemos los textos de los fragmentos recuperados
        textos_contexto = [doc.page_content for doc in docs]
        
        # 3. Vectorizamos los fragmentos individuales (Ollama los maneja mejor separados)
        vecs_contexto = embeddings.embed_documents(textos_contexto)

        # 4. Comparamos la respuesta contra CADA fragmento y guardamos el mejor puntaje
        max_score = 0
        for vec_ctx in vecs_contexto:
            score = cosine_similarity([vec_ctx], [vec_respuesta])[0][0]
            if score > max_score:
                max_score = score

        console.print(f"[dim][DEBUG] Score de Similitud Coseno (Mejor coincidencia): {max_score:.4f}[/dim]")

        # Umbral estricto de validación semántica
        UMBRAL_COSENO = 0.77

        if max_score < UMBRAL_COSENO or "No tengo suficiente información" in respuesta_completa:
            respuesta_final = "No tengo suficiente información en el texto para responder esto."
            console.print(f"\n[bold red]⚠️ Alucinación o desvío interceptado por el Guardián de Coseno. Respuesta corregida:[/bold red]\n{respuesta_final}\n")
            
            # ¡NUEVO! Limpiamos la historia para que el "No sé" y el contexto erróneo no contaminen la siguiente búsqueda
            chat_history.clear() 
        else:
            respuesta_final = respuesta_completa
            # Solo guardamos el historial si la respuesta fue exitosa y fundamentada
            chat_history.extend([
                HumanMessage(content=query_text),
                AIMessage(content=respuesta_final)
            ])

        # Registrar en la historia de la conversación la respuesta validada
        chat_history.extend([
            HumanMessage(content=query_text),
            AIMessage(content=respuesta_final)
        ])

if __name__ == "__main__":
    main()