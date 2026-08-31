from datacollection.vector_store import vector_store


def retrieve_docs(query_string: str) -> list:
    """Returns selected docs using max margianl relevance"""
    docs_returned = 4
    docs_fetched = 20
    diversity = 0.7

    return vector_store.max_marginal_relevance_search(
        query_string, docs_returned, docs_fetched, diversity
    )
