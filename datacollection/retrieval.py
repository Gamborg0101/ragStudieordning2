from rapidfuzz import fuzz

from datacollection.vector_store import vector_store


def get_title_lookup():
    """Returns a dictionary mapping source filenames to document titles"""
    title_lookup = {}

    for record in vector_store.store.values():
        title = record["metadata"]["title"]
        source = record["metadata"]["source"]
        title_lookup[source] = title
    return title_lookup


def retrieve_docs(query_string: str) -> list:
    """Returns selected docs using max margianl relevance"""
    docs_returned = 4
    docs_fetched = 20
    diversity = 0.7

    source_id = match_question_to_source(query_string)

    retrieved_documents2 = vector_store.max_marginal_relevance_search(
        query_string,
        docs_returned,
        docs_fetched,
        diversity,
    )

    is_from_source = from_source(source_id, retrieved_documents2)

    retrieved_documents = vector_store.max_marginal_relevance_search(
        query_string,
        docs_returned,
        docs_fetched,
        diversity,
        filter=(is_from_source if source_id else None),
    )

    return retrieved_documents


def from_source(source_id, retrieved_documents):
    """Checks if any retrieved document is from the specified source"""
    value_true = False
    for doc in retrieved_documents:
        source_values = doc.metadata["source"].split(".")[0]
        if source_values == source_id:
            value_true = True
    return value_true


def match_question_to_source(query_string: str):
    """Returns source_id and score based on match"""
    source_id = 0.0
    highest_score = 0.0
    titles = get_title_lookup()

    for title in titles.items():
        match_value = fuzz.partial_ratio(title[1], query_string)
        if match_value > highest_score:
            highest_score = match_value
            if highest_score > 80:
                source_id = title[0].split(".")[0]
            else:
                source_id = None
    return {"source_id": source_id, "highest_score": highest_score}
