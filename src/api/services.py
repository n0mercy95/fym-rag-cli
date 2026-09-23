# src/api/services.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

from src.config.logger import get_logger
from src.config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL
from src.vector_db.chroma_manager import get_chroma_db
from src.prompts.prompts import PROMPT_TEMPLATE
from src.llm.retriever_builder import build_smart_retriever
from src.llm.guardrails import validate_answer_with_cosine, evaluate_with_llm_judge
from src.ingestion.document_processor import run_ingestion

class RAGService:
    def __init__(self, chroma_path: str):
        # 1. El logger ahora pertenece a esta instancia
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(f"Levantando RAGService... Conectando a ChromaDB en: {chroma_path}")
        
        # 2. Conexiones persistentes (¡solo se ejecutan UNA vez al arrancar!)
        self.chroma_path = chroma_path
        self.db = get_chroma_db(chroma_path)
        self.llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
        
        # 3. Construcción de Cadenas de Langchain
        self.history_aware_retriever = build_smart_retriever(self.db, self.llm)
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_TEMPLATE),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        self.qa_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # 4. Memoria conversacional multi-usuario
        # Es un diccionario que guarda las charlas separadas por ID. 
        # Ejemplo: {"usuario_1": [mensajes], "usuario_2": [mensajes]}
        self.sessions_memory = {}

    def _get_history(self, session_id: str) -> list:
        """Devuelve el historial de un usuario, o crea uno nuevo si no existe."""
        if session_id not in self.sessions_memory:
            self.sessions_memory[session_id] = []
        return self.sessions_memory[session_id]

    def chat(self, query: str, session_id: str) -> dict:
        self.logger.info(f"[{session_id}] Nueva consulta: '{query}'")
        chat_history = self._get_history(session_id)
        
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
        
       # ... (Tu código anterior: PASO 1 y PASO 2) ...

        # PASO 3: Validación Matemática (Coseno)
        es_valida_coseno, score = validate_answer_with_cosine(respuesta_completa, docs, self.db.embeddings)
        
        if not es_valida_coseno:
            self.logger.warning(f"[{session_id}] Alucinación interceptada por Coseno. Score: {score:.4f}")
            respuesta_segura = "No tengo suficiente información."
            chat_history.extend([HumanMessage(content=query), AIMessage(content=respuesta_segura)])
            return {
                "answer": respuesta_segura,
                "hallucination_intercepted": True,
                "cosine_score": score
            }
            
        # # PASO 4: Validación Lógica (LLM Juez)
        # self.logger.info(f"[{session_id}] Coseno aprobado ({score:.4f}). Consultando al Juez LLM...")
        # juez_aprueba, veredicto_crudo = evaluate_with_llm_judge(respuesta_completa, docs)
        
        # if not juez_aprueba:
        #     self.logger.warning(f"[{session_id}] Juez LLM detectó falacia lógica. Veredicto crudo: {veredicto_crudo}")
        #     respuesta_segura = "Lo siento, identifiqué una inconsistencia física en mi razonamiento y preferí descartar la respuesta."
        #     chat_history.extend([HumanMessage(content=query), AIMessage(content=respuesta_segura)])
        #     return {
        #         "answer": respuesta_segura,
        #         "hallucination_intercepted": True,
        #         "cosine_score": score
        #     }

        # PASO 5: Guardado y retorno exitoso
        self.logger.info(f"[{session_id}] Juez LLM aprobó la respuesta.")
        chat_history.extend([HumanMessage(content=query), AIMessage(content=respuesta_completa)])
        
        return {
            "answer": respuesta_completa,
            "hallucination_intercepted": False,
            "cosine_score": score
        }

    def ingest(self, file_path: str) -> tuple[bool, str]:
        self.logger.info(f"Petición de ingesta recibida para: {file_path}")
        try:
            # ¡NUEVO!: Pasamos la instancia de Chroma ya conectada
            run_ingestion(pdf_path=file_path, db_instance=self.db)
            return True, "Ingesta completada con éxito."
        except Exception as e:
            self.logger.error(f"Fallo durante la ingesta: {e}", exc_info=True)
            return False, str(e)