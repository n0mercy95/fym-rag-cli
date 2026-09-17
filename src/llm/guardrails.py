from sklearn.metrics.pairwise import cosine_similarity
from src.config.logger import get_logger

# Inicializamos el logger para este módulo
logger = get_logger("Guardrails")

def validate_answer_with_cosine(respuesta_completa: str, docs: list, embeddings) -> tuple[bool, float]:
    """
    Compara la respuesta del LLM contra los fragmentos originales.
    Retorna (es_valida, max_score).
    """
    if "No tengo suficiente información" in respuesta_completa:
        # Registramos que el modelo fue honesto por su cuenta
        logger.info("El modelo admitió no tener información suficiente. Validación por coseno omitida.")
        return False, 0.0

    vec_respuesta = embeddings.embed_query(respuesta_completa)
    textos_contexto = [doc.page_content for doc in docs]
    vecs_contexto = embeddings.embed_documents(textos_contexto)

    max_score = 0.0
    for vec_ctx in vecs_contexto:
        score = cosine_similarity([vec_ctx], [vec_respuesta])[0][0]
        if score > max_score:
            max_score = score

    UMBRAL_COSENO = 0.77
    es_valida = max_score >= UMBRAL_COSENO
    
    # Registramos el resultado de la validación silenciosamente en el log
    if es_valida:
        logger.info(f"Respuesta validada correctamente. Score: {max_score:.4f} (Umbral: {UMBRAL_COSENO})")
    else:
        # Usamos nivel WARNING porque el sistema interceptó un comportamiento no deseado (alucinación)
        logger.warning(f"Alucinación interceptada. Score insuficiente: {max_score:.4f} (Umbral mínimo: {UMBRAL_COSENO})")
    
    return es_valida, max_score