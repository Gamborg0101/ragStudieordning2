import uuid
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool
from prompts.rag_workflow_instructions import (
    RAG_WORKFLOW_INSTRUCTIONS as rag_workflow_instructions,
)
from prompts.chunk_analyst_instructions import (
    CHUNK_ANALYST_INSTRUCTIONS as chunk_analyst_instructions,
)
from prompts.search_studieordninger_prompt import search_studieordninger_prompt
from prompts.subagent_delegation_instructions import (
    SUBAGENT_DELEGATION_INSTRUCTIONS as subagent_delegation_instructions,
)
from datacollection import vector
from datacollection.load_corpus import VECTOR_STORE_PATH, CORPUS_DIR, load_corpus_docs
from deepagents.backends import StateBackend


def search_studieordninger(query: str) -> str:
    retrieved_docs = vector.vector_store.similarity_search(query, k=4)
    batch_id = uuid.uuid4().hex[:8]
    uploads: list[tuple[str, bytes]] = []
    saved_paths: list[str] = []

    for index, doc in enumerate(retrieved_docs, start=1):
        path = f"/retrieved/{batch_id}/chunk_{index}.md"
        content = (
            f"# Source: {doc.metadata.get('source', 'unknown')}\n\n{doc.page_content}"
        )
        uploads.append((path, content.encode("utf-8")))
        saved_paths.append(path)

    backend.upload_files(uploads)
    return f"Saved {len(saved_paths)} studieordning chunks:\n" + "\n".join(saved_paths)


search_studieordninger.__doc__ = search_studieordninger_prompt


max_concurrent_analysts = 3

INSTRUCTIONS = (
    rag_workflow_instructions
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + subagent_delegation_instructions.format(
        max_concurrent_analysts=max_concurrent_analysts,
    )
)

chunk_analyst_subagent = {
    "name": "chunk-analyst",
    "description": (
        "Analyze one retrieved studieordning chunk file. "
        "Pass the user question and a single file path under /retrieved/."
    ),
    "system_prompt": chunk_analyst_instructions,
}

model = init_chat_model(model="ollama:qwen2.5:7b")

backend = StateBackend()

agent = create_deep_agent(
    model=model,
    tools=[search_studieordninger],
    backend=backend,
    system_prompt=INSTRUCTIONS,
    subagents=[chunk_analyst_subagent],
)

EXAMPLE_QUERY = "Hvor mange ECTS er kandidattilvalget i latin normeret til, og hvornår trådte studieordningen i kraft?"

if __name__ == "__main__":
    result = agent.invoke({"messages": [HumanMessage(content=EXAMPLE_QUERY)]})

    for msg in result.get("messages", []):
        if msg.text:
            print(msg.text)
