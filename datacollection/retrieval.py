from datacollection.vector_store import vector_store


def retrieve_docs(query_string: str) -> list:
    """Returns selected docs using max margianl relevance"""
    docs_returned = 4
    docs_fetched = 20
    diversity = 0.7

    return vector_store.max_marginal_relevance_search(
        query_string, docs_returned, docs_fetched, diversity
    )


def fuzzy_matcher(title_lookup: dict, question: str):
    """Returns the match between title and question to identify if title is inside of question for optimized retrival"""
    title_and_source = {}
    higest_score = 0

    for item in title_lookup.items():
        # print(item.split("(")[0]) #Title here
        print(item[0])
        print(item[1])
    # for item in title_lookup.items():
    #     ratio = fuzz.partial_ratio(item["title"][1], question)
    #     if ratio > 80:
    #         title_and_source = {
    #             item["metadata"]["title"][1],
    #             item["metadata"]["source"],
    #         }
    # return title_and_source


# Return (title, source) - skal kun return hvis ratio > 80
