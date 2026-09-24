# src/api/services.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from src.config.logger import get_logger
from src.config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL, REDIS_URL
from src.vector_db.chroma_manager import get_chroma_db
from src.prompts.prompts import PROMPT_TEMPLATE
from src.llm.retriever_builder import build_smart_retriever
from src.llm.guardrails import validate_answer_with_cosine, evaluate_with_llm_judge
from src.ingestion.document_processor import run_ingestion
from src.utils.redis_manager import RedisMemoryManager

class RAGService:
    def __init__(self, chroma_path: str):
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(f"Levantando RAGService... Conectando a ChromaDB en: {chroma_path}")
        
        # Conexiones persistentes
        self.chroma_path = chroma_path
        self.db = get_chroma_db(chroma_path)
        self.llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
        
        # Construcción de Cadenas de Langchain
        self.history_aware_retriever = build_smart_retriever(self.db, self.llm)
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_TEMPLATE),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        self.qa_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # NUEVO: Instancia del gestor de memoria conectada al contenedor de Redis
        self.redis_manager = RedisMemoryManager(redis_url=REDIS_URL)

    def chat(self, query: str, session_id: str) -> dict:
        self.logger.info(f"[{session_id}] Nueva consulta: '{query}'")
        
        try:
            # 1. Recuperamos el historial directamente desde Redis
            session_history = self.redis_manager.get_session_history(session_id)
            chat_history = session_history.messages # Lista de mensajes para inyectar al prompt
            
            # PASO 1: Recuperación
            docs = self.history_aware_retriever.invoke({"input": query, "chat_history": chat_history})
            
            if not docs:
                self.logger.warning(f"[{session_id}] Sin contexto suficiente.")
                return {
                    "answer": "No tengo suficiente información.",
                    "hallucination_intercepted": False,
                    "cosine_score": 0.0
                }

            # PASO 2: Generación
            respuesta_completa = self.qa_chain.invoke({
                "context": docs, 
                "input": query, 
                "chat_history": chat_history
            })
            
            self.logger.info(f"[{session_id}] Respuesta generada por el LLM:\n{respuesta_completa}\n{'-'*40}")

            # PASO 3: Validación Matemática (Coseno)
            es_valida_coseno, score = validate_answer_with_cosine(respuesta_completa, docs, self.db.embeddings)
            
            if not es_valida_coseno:
                self.logger.warning(f"[{session_id}] Alucinación interceptada por Coseno. Score: {score:.4f}")
                respuesta_segura = "No tengo suficiente información."
                
                # Guardamos la respuesta segura en Redis en lugar de la alucinación
                session_history.add_user_message(query)
                session_history.add_ai_message(respuesta_segura)
                
                return {
                    "answer": respuesta_segura,
                    "hallucination_intercepted": True,
                    "cosine_score": score
                }

            # (Si en el futuro descomentas el PASO 4 del LLM Judge, aplica la misma lógica de add_user_message/add_ai_message aquí)

            # PASO 5: Guardado y retorno exitoso
            self.logger.info(f"[{session_id}] Respuesta aprobada. Guardando en Redis.")
            
            # Guardamos la interacción real en Redis
            session_history.add_user_message(query)
            session_history.add_ai_message(respuesta_completa)
            
            return {
                "answer": respuesta_completa,
                "hallucination_intercepted": False,
                "cosine_score": score
            }
            
        except Exception as e:
            self.logger.error(f"[{session_id}] Error durante el procesamiento del chat: {e}", exc_info=True)
            raise

    def ingest(self, file_path: str) -> tuple[bool, str]:
        self.logger.info(f"Petición de ingesta recibida para: {file_path}")
        try:
            run_ingestion(pdf_path=file_path, db_instance=self.db)
            return True, "Ingesta completada con éxito."
        except Exception as e:
            self.logger.error(f"Fallo durante la ingesta: {e}", exc_info=True)
            return False, str(e)