SEARCH_STUDIEORDNINGER_PROMPT = """Search the indexed studieordning (study regulation) corpus and save matching chunks to the agent filesystem.

    Args:
        query: Natural language search query.

    Returns:
        File paths where retrieved chunks were saved under /retrieved/.
    """
