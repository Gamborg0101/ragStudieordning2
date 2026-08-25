from pathlib import Path

import numpy as np
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from datacollection.load_corpus import CORPUS_DIR, load_corpus_docs


def save_vector_store(vector_store: InMemoryVectorStore, path: Path) -> None:
    """Persist chunk text, metadata, and embeddings to a compact .npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(vector_store.store.keys())
    vectors = np.array([vector_store.store[i]["vector"] for i in ids], dtype=np.float32)
    texts = np.array([vector_store.store[i]["text"] for i in ids])
    sources = np.array(
        [vector_store.store[i]["metadata"].get("source", "") for i in ids]
    )
    np.savez_compressed(
        path,
        ids=np.array(ids),
        vectors=vectors,
        texts=texts,
        sources=sources,
    )


def load_vector_store(path: Path, embedding: OllamaEmbeddings) -> InMemoryVectorStore:
    """Rebuild an InMemoryVectorStore from a persisted .npz file without re-embedding."""
    data = np.load(path)
    vector_store = InMemoryVectorStore(embedding=embedding)
    vector_store.store = {
        str(doc_id): {
            "id": str(doc_id),
            "vector": vector.tolist(),
            "text": str(text),
            "metadata": {"source": str(source)},
        }
        for doc_id, vector, text, source in zip(
            data["ids"], data["vectors"], data["texts"], data["sources"]
        )
    }
    return vector_store


embeddings = OllamaEmbeddings(model="nomic-embed-text")

VECTOR_STORE_PATH = Path("data/vector_store.npz")

if VECTOR_STORE_PATH.exists():
    vector_store = load_vector_store(VECTOR_STORE_PATH, embeddings)
    print(f"Loaded {len(vector_store.store)} chunks from {VECTOR_STORE_PATH}.")
else:
    docs = load_corpus_docs(CORPUS_DIR)
    print(f"Loaded {len(docs)} documents from {CORPUS_DIR}.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    print(f"Split corpus into {len(all_splits)} chunks.")

    for split in all_splits:
        split.page_content = f"{split.metadata['title']}\n\n{split.page_content}"

    vector_store = InMemoryVectorStore(embedding=embeddings)

    EMBED_BATCH_SIZE = 200
    for start in range(0, len(all_splits), EMBED_BATCH_SIZE):
        batch = all_splits[start : start + EMBED_BATCH_SIZE]
        vector_store.add_documents(documents=batch)
        print(f"Indexed {start + len(batch)}/{len(all_splits)} chunks.")

    save_vector_store(vector_store, VECTOR_STORE_PATH)
    print(f"Persisted vector store to {VECTOR_STORE_PATH}.")
