import logging
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from src.prompts.prompts import REPHRASE_TEMPLATE

# 1. Importación defensiva para EnsembleRetriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain_community.retrievers import EnsembleRetriever
    except ImportError:
        from langchain_classic.retrievers import EnsembleRetriever

# 2. Importación defensiva para ContextualCompressionRetriever
try:
    from langchain.retrievers import ContextualCompressionRetriever
except ImportError:
    try:
        from langchain_community.retrievers import ContextualCompressionRetriever
    except ImportError:
        from langchain_classic.retrievers import ContextualCompressionRetriever

# 3. Importación defensiva para CrossEncoderReranker
try:
    from langchain.retrievers.document_compressors import CrossEncoderReranker
except ImportError:
    try:
        from langchain_community.document_compressors.cross_encoder import CrossEncoderReranker
    except ImportError:
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

def build_smart_retriever(vectorstore, llm):
    """
    Construye un pipeline de recuperación avanzado:
    BM25 + Vectorial -> Fusión -> Cross-Encoder Reranking -> Memoria
    """
    try:
        logger.info("Construyendo pipeline de recuperación avanzado...")
        
        # 1. Recuperación Amplia: Vectorial (Traemos 15 candidatos)
        # Quitamos el threshold estricto aquí para no perder candidatos que el Reranker podría salvar
        vector_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 15}
        )
        
        # 2. Recuperación Amplia: BM25 (Traemos 15 candidatos léxicos)
        db_data = vectorstore.get()
        if not db_data or not db_data.get("documents"):
            logger.warning("ChromaDB vacío. Fallback a búsqueda vectorial.")
            hybrid_retriever = vector_retriever
        else:
            textos = db_data["documents"]
            metadatas = db_data["metadatas"]
            docs_para_bm25 = [
                Document(page_content=txt, metadata=meta) 
                for txt, meta in zip(textos, metadatas)
            ]
            bm25_retriever = BM25Retriever.from_documents(docs_para_bm25)
            bm25_retriever.k = 15
            
            # Fusión 50/50
            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.5, 0.5]
            )

        # 3. La Magia: Cross-Encoder Reranker
        logger.info("Cargando modelo Cross-Encoder (esto puede tardar unos segundos la primera vez)...")
        # Usamos BAAI/bge-reranker-base, excelente para precisión técnica multilingüe
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = CrossEncoderReranker(model=model, top_n=3)
        
        # Envolvemos el híbrido con el compresor
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=hybrid_retriever
        )
        
        # 4. Capa Conversacional
        rephrase_prompt = PromptTemplate.from_template(REPHRASE_TEMPLATE)
        history_aware_retriever = create_history_aware_retriever(
            llm, compression_retriever, rephrase_prompt
        )
        
        logger.info("Pipeline RAG (Híbrido + Reranker) instanciado con éxito.")
        return history_aware_retriever

    except Exception as e:
        logger.error(f"Error crítico al construir el Retriever: {e}", exc_info=True)
        raise e