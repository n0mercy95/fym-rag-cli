from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever

from src.prompts.prompts import REPHRASE_TEMPLATE

def build_smart_retriever(db, llm):
    # Retriever estricto con umbral y filtro de metadatos
    base_retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.5, 
            "k": 5,
            "filter": {"capitulo": {"$ne": "Prefacio_o_Indice"}}
        }
    )
    
    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", REPHRASE_TEMPLATE),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    return create_history_aware_retriever(llm, base_retriever, rephrase_prompt)