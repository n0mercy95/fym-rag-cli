import os

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

# Rutas
CHROMA_PATH = "chroma_db"

# Prompt 1: Reformulador de Preguntas (El "Traductor" de memoria)
REPHRASE_TEMPLATE = """
Dada la siguiente historia de conversación y la pregunta final del usuario, 
que podría hacer referencia al contexto anterior, reformula la pregunta para que 
sea una pregunta independiente y clara. NO la respondas, solo reformúlala.
"""

# Prompt 2: Defensivo Principal y Reglas Matemáticas
PROMPT_TEMPLATE = """
Eres un asistente experto en física. Responde a la pregunta basándote únicamente en el contexto teórico proporcionado. 
Ignora cualquier fragmento de texto que parezca código basura, caracteres matemáticos rotos o ecuaciones mal formateadas por el OCR. 

REGLAS ESTRICTAS DE FORMATO MATEMÁTICO:
- ESTÁ ESTRICTAMENTE PROHIBIDO usar sintaxis de LaTeX (no uses $, $$, \\, \mathbf, \int, \frac, etc.).
- Debes escribir TODAS las ecuaciones y fórmulas en texto plano usando caracteres Unicode estándar soportados por UTF-8.
- Usa símbolos legibles como: ∫, ∑, ∂, √, α, β, θ, π.
- Escribe las ecuaciones de forma lineal. Ejemplo de fracción: (a + b) / c. Ejemplo de potencia: x^2. Ejemplo de integral: ∫_a^b f(x) dx.

Si el contexto no contiene la teoría para responder, di "No tengo suficiente información en el texto para responder esto".

Contexto:
{context}
"""

def main():
    console = Console()
    console.print("[bold green]⚛️ Iniciando Asistente de Física RAG con Memoria...[/bold green]")

    # 1. Conexiones Base
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    # Convertimos la BD en un "recuperador" estándar de LangChain
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")
    
    # 2. Cadena de Conciencia del Historial
    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", REPHRASE_TEMPLATE),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, rephrase_prompt)

    # 3. Cadena principal de Respuesta (QA)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT_TEMPLATE),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 4. Unir ambas cadenas en un flujo RAG Conversacional
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # 5. Iniciar la memoria vacía de la sesión actual
    chat_history = []

    # Bucle interactivo
    while True:
        query_text = Prompt.ask("\n[bold blue]Haz una pregunta (o escribe 'salir')[/bold blue]")
        if query_text.lower() in ["salir", "exit", "quit"]:
            console.print("[bold red]¡Hasta luego![/bold red]")
            break

        console.print("\n[bold yellow]🧠 Consultando memoria, buscando en el libro y generando respuesta...[/bold yellow]")
        
        console.print("\n[bold green]Respuesta:[/bold green]")
        
        respuesta_completa = ""
        fuentes_recuperadas = []

        # Ejecutar la cadena en modo STREAMING (efecto máquina de escribir)
        for chunk in rag_chain.stream({
            "input": query_text,
            "chat_history": chat_history
        }):
            # Imprimir palabra por palabra en tiempo real
            if "answer" in chunk:
                print(chunk["answer"], end="", flush=True)
                respuesta_completa += chunk["answer"]
            
            # Guardar las fuentes para mostrarlas al final
            if "context" in chunk:
                fuentes_recuperadas = chunk["context"]

        print("\n") # Salto de línea al terminar la respuesta

        # Imprimir fuentes
        console.print("\n[dim]Fuentes recuperadas del Zemansky:[/dim]")
        for doc in fuentes_recuperadas:
            console.print(f"[dim]- ID: {doc.metadata.get('id', 'Desconocido')}[/dim]")

        # Guardar la interacción actual en la memoria
        chat_history.extend([
            HumanMessage(content=query_text),
            AIMessage(content=respuesta_completa)
        ])

if __name__ == "__main__":
    main()