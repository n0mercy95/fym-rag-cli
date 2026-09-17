import time

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

from rich.console import Console
from rich.prompt import Prompt

# Importamos nuestros módulos especializados
from src.vector_db.chroma_manager import get_chroma_db
from src.prompts.prompts import PROMPT_TEMPLATE

from src.llm.retriever_builder import build_smart_retriever
from src.llm.guardrails import validate_answer_with_cosine

def start_interactive_chat(chroma_path: str):
    console = Console()
    console.print("[bold green]⚛️ Iniciando Asistente de Física RAG...[/bold green]")

    # 1. Inicializamos dependencias principales
    db = get_chroma_db(chroma_path)
    llm = ChatOllama(model="qwen2.5:3b", base_url="http://localhost:11434", temperature=0)
    
    # 2. Construimos cadenas (Chains)
    history_aware_retriever = build_smart_retriever(db, llm)
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT_TEMPLATE),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    chat_history = []

    # 3. Bucle Principal de Interacción
    while True:
        query_text = Prompt.ask("\n[bold blue]Haz una pregunta (o escribe 'salir')[/bold blue]")
        if query_text.lower() in ["salir", "exit", "quit"]:
            break

        # PASO 1: Recuperación
        docs = history_aware_retriever.invoke({"input": query_text, "chat_history": chat_history})

        if not docs:
            console.print("\n[bold red]Respuesta:[/bold red]\nNo tengo suficiente información.\n")
            continue

        # PASO 2: RAYOS X - Imprimir exactamente qué texto encontró ChromaDB
        console.print("\n[magenta]--- [DEBUG] TEXTO REAL RECUPERADO DE LA BD ---[/magenta]")
        for i, doc in enumerate(docs):
            console.print(f"[dim]Fragmento {i+1} | Origen: {doc.metadata.get('capitulo', 'Desconocido')} (Pág {doc.metadata.get('page', 'Desconocida')})[/dim]")
            console.print(f"[dim]{doc.page_content[:200]}...[/dim]\n")
        console.print("[magenta]--------------------------------------------[/magenta]\n")

        # PASO 3: Generación
        console.print("[bold green]Respuesta del Modelo:[/bold green]")
        respuesta_completa = ""
        for chunk in qa_chain.stream({"context": docs, "input": query_text, "chat_history": chat_history}):
            texto = chunk if isinstance(chunk, str) else str(chunk)
            print(texto, end="", flush=True)
            respuesta_completa += texto
        print("\n")

        # PASO 4: Validación (Guardián)
        console.print("[dim]⏳ Validando matemáticamente...[/dim]")
        # Extraemos el modelo de embeddings desde la db para pasárselo al guardián
        es_valida, score = validate_answer_with_cosine(respuesta_completa, docs, db.embeddings)
        console.print(f"[dim][DEBUG] Score de Similitud Coseno: {score:.4f}[/dim]")

        if not es_valida:
            console.print("\n[bold red]⚠️ Alucinación interceptada. Respuesta corregida.[/bold red]\n")
            chat_history.clear() 
        else:
            chat_history.extend([HumanMessage(content=query_text), AIMessage(content=respuesta_completa)])