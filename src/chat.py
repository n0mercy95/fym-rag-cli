import os

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

from rich.console import Console
from rich.prompt import Prompt

from prompts import REPHRASE_TEMPLATE, PROMPT_TEMPLATE

# Rutas Absolutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

def main():
    console = Console()
    console.print("[bold green]⚛️ Iniciando Asistente de Física RAG (Modo Debug con Rayos X)...[/bold green]")

    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # Subimos el umbral a 0.65 para evitar que pase basura matemática lejana
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
        
        # PASO 1: Buscar documentos
        docs = history_aware_retriever.invoke({
            "input": query_text,
            "chat_history": chat_history
        })

        # PASO 2: EL CORTAFUEGOS DE PYTHON (Si el umbral de 0.65 los rechaza, frena aquí)
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

        # PASO 4: Generar respuesta si pasó los filtros
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
        
        chat_history.extend([
            HumanMessage(content=query_text),
            AIMessage(content=respuesta_completa)
        ])

if __name__ == "__main__":
    main()