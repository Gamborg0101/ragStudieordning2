from prompts.chunk_analyst_instructions import CHUNK_ANALYST_INSTRUCTIONS
from prompts.rag_workflow_instructions import RAG_WORKFLOW_INSTRUCTIONS
from prompts.search_studieordninger_prompt import search_studieordninger_prompt
from prompts.subagent_delegation_instructions import SUBAGENT_DELEGATION_INSTRUCTIONS

__all__ = [
    "CHUNK_ANALYST_INSTRUCTIONS",
    "RAG_WORKFLOW_INSTRUCTIONS",
    "SUBAGENT_DELEGATION_INSTRUCTIONS",
    "search_studieordninger_prompt",
]
