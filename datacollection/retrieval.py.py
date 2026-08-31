from datacollection.vector_store import vector_store

def retrieve_docs(query_string:str) -> list:
        """ Returns selected docs using max margianl relevance"""
        DOCS_RETURNED = 4
        DOCS_FETCHED = 20
        DEGREE_OF_DIVERSITY = 0.7
        
        return vector_store.max_marginal_relevance_search(
            query_string, DOCS_RETURNED, DOCS_FETCHED, DEGREE_OF_DIVERSITY
        )
