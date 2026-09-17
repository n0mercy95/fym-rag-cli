import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def procesar_pdf_a_chunks(pdf_path: str) -> list:
    """
    Carga un PDF, etiqueta los capítulos y divide el texto en fragmentos (chunks)
    con identificadores únicos. Retorna la lista de fragmentos.
    """
    print("📄 Cargando documento...")
    loader = PyPDFLoader(pdf_path)
    documentos_brutos = loader.load()

    print("🏷️ Etiquetando capítulos...")
    capitulo_actual = "Prefacio_o_Indice"
    patron_capitulo = re.compile(r'(?m)^\s*CAP[IÍ]TULO\s+(\d+)')

    for doc in documentos_brutos:
        match = patron_capitulo.search(doc.page_content)
        if match:
            numero_cap = match.group(1)
            capitulo_actual = f"Capitulo_{numero_cap}"
        
        doc.metadata["capitulo"] = capitulo_actual

    print("✂️ Dividiendo el texto en fragmentos (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documentos_brutos)
    print(f"✅ Se generaron {len(chunks)} fragmentos etiquetados.")

    print("🆔 Generando IDs únicos para cada fragmento...")
    for i, chunk in enumerate(chunks):
        fuente = chunk.metadata.get("source", "desconocido")
        pagina = chunk.metadata.get("page", 0)
        capitulo = chunk.metadata.get("capitulo", "Sin_Capitulo")
        
        chunk.metadata["id"] = f"{fuente}:{capitulo}:{pagina}:{i}"

    return chunks