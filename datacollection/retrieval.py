import re

from rapidfuzz import fuzz

from .vector_store import vector_store


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

    def from_source(docs) -> bool:
        """Check if document has same source_id as source"""
        return docs.metadata["source"].split(".")[0] == source_id["source_id"]

    retrieved_documents = vector_store.max_marginal_relevance_search(
        query_string,
        docs_returned,
        docs_fetched,
        diversity,
        filter=(from_source if source_id["source_id"] else None),
    )

    return retrieved_documents


def match_question_to_source(query_string: str) -> dict[str, str]:
    """Returns source_id and score based on match"""
    source_id = None
    highest_score = 0.0
    best_year = None
    score_tolerance = 80.0
    tie_tolerance = 5.0
    titles = get_title_lookup()

    for source, title in titles.items():
        match_value = fuzz.partial_ratio(
            title, query_string, processor=lambda str: str.lower()
        )

        year_match = re.search(r"\((\d{4})\)", title)
        candidate_year = int(year_match.group(1)) if year_match else None

        is_tie = abs(match_value - highest_score) <= tie_tolerance
        is_new_best = match_value > highest_score

        if is_tie:
            candidate_wins_tie = candidate_year is not None and (
                best_year is None or candidate_year > best_year
            )
            if not candidate_wins_tie:
                continue
        elif not is_new_best:
            continue

        highest_score = max(match_value, highest_score)
        best_year = candidate_year
        source_id = source.split(".")[0] if highest_score > score_tolerance else None

    return {"source_id": source_id, "highest_score": highest_score}


test = match_question_to_source(
    "Hvor mange ECTS er kandidattilvalget i latin normeret til, og hvornår trådte studieordningen i kraft?"
)

# for item in test:
# print(item)
