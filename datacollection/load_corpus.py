from pathlib import Path
from bs4 import BeautifulSoup
from langchain_core.documents import Document


CORPUS_DIR = Path("./corpus")
VECTOR_STORE_PATH = Path("data/vector_store.npz")


def load_corpus_docs(corpus_dir: Path) -> list[Document]:
    """Load and parse the local HTML corpus into Documents."""
    docs: list[Document] = []
    for path in corpus_dir.glob(
        "Regelgrundlag for masteruddannelserne - Master i it.html"
    ):
        soup = BeautifulSoup(path.read_text(), "html.parser")
        for tag in soup(["nav", "header", "footer", "script", "style"]):
            tag.decompose()
        text = soup.get_text()
        print(text)
        docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs
