import re
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document

CORPUS_DIR = Path(__file__).parent.parent / "corpus"


def load_corpus_docs(corpus_dir: Path) -> list[Document]:
    """Load and parse the local HTML corpus into Documents."""
    docs: list[Document] = []
    for path in corpus_dir.glob("*.html"):
        soup = BeautifulSoup(path.read_text(), "html.parser")

        found_title = soup.find("title")
        if found_title:
            title = found_title.get_text()
        else:
            title = "Did not find a title"

        for tag in soup(["nav", "header", "footer", "script", "style"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text()).strip()

        docs.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "title": title},
            )
        )
    return docs
