import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

# Rutas
CHROMA_PATH = "chroma_db"

# Prompt Defensivo
PROMPT_TEMPLATE = """
Eres un asistente experto en física. Responde a la pregunta basándote únicamente en el contexto teórico proporcionado. 
Ignora cualquier fragmento de texto que parezca código basura, caracteres matemáticos rotos o ecuaciones mal formateadas por el OCR. 
Si el contexto no contiene la teoría para responder, di "No tengo suficiente información en el texto para responder esto".

Contexto:
{context}

Pregunta: {question}
"""

def main():
    console = Console()
    console.print("[bold green]⚛️ Iniciando Asistente de Física RAG...[/bold green]")

    # 1. Conectar a la base de datos vectorial
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # 2. Cargar el modelo generativo (asegúrate de haber descargado llama3.1 u otro ligero)
    llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434")

    # 3. Bucle interactivo
    while True:
        query_text = Prompt.ask("\n[bold blue]Haz una pregunta (o escribe 'salir')[/bold blue]")
        if query_text.lower() in ["salir", "exit", "quit"]:
            console.print("[bold red]¡Hasta luego![/bold red]")
            break

        # Buscar los 3 fragmentos más relevantes en la base de datos
        resultados = db.similarity_search(query_text, k=3)
        if len(resultados) == 0:
            console.print("[red]No se encontró contexto relevante en el libro.[/red]")
            continue

        # Unir los textos recuperados
        context_text = "\n\n---\n\n".join([doc.page_content for doc in resultados])

        # Crear el prompt final
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        console.print("\n[bold yellow]🧠 Analizando el libro y generando respuesta...[/bold yellow]")
        
        # Invocar al modelo
        response = llm.invoke(prompt)

        # Mostrar la respuesta formateada con Rich
        console.print("\n[bold green]Respuesta:[/bold green]")
        console.print(Markdown(response.content))

        # Mostrar las fuentes utilizadas
        console.print("\n[dim]Fuentes recuperadas del Zemansky:[/dim]")
        for doc in resultados:
            console.print(f"[dim]- ID: {doc.metadata.get('id', 'Desconocido')}[/dim]")

if __name__ == "__main__":
    main()