from sklearn.metrics.pairwise import cosine_similarity

def validate_answer_with_cosine(respuesta_completa: str, docs: list, embeddings) -> tuple[bool, float]:
    """
    Compara la respuesta del LLM contra los fragmentos originales.
    Retorna (es_valida, max_score).
    """
    if "No tengo suficiente información" in respuesta_completa:
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
    
    return es_valida, max_score