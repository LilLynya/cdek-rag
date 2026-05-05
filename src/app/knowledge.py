from __future__ import annotations

from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .config import settings

COUNTRY_FILES: dict[str, str] = {
    "germany_rules.txt": "germany",
    "france_rules.txt": "france",
}


def load_documents(data_dir: Path | str | None = None) -> list[Document]:
    """Считывает все .txt из data_dir в LangChain Document с метаданными."""
    base = Path(data_dir) if data_dir is not None else Path(settings.data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Директория с базой знаний не найдена: {base.resolve()}")

    documents: list[Document] = []
    for path in sorted(base.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "country": COUNTRY_FILES.get(path.name),
                },
            )
        )

    if not documents:
        raise RuntimeError(f"В директории {base.resolve()} не найдено ни одного .txt файла")

    return documents


class KnowledgeBase:
    """Лёгкая обёртка над BM25Retriever со знанием о страновых документах."""

    def __init__(self, data_dir: Path | str | None = None, top_k: int = 4) -> None:
        self.documents = load_documents(data_dir)
        self.top_k = top_k
        self._retriever = BM25Retriever.from_documents(self.documents)
        self._retriever.k = max(top_k, len(self.documents))

    def search(self, query: str, country: str | None = None, k: int | None = None) -> list[Document]:
        """Возвращает релевантные документы.

        Если задана `country` — документы про другие страны исключаются,
        а профильный страновой файл гарантированно попадает в выборку.
        """
        limit = k or self.top_k
        ranked = self._retriever.invoke(query)

        if country is None:
            return ranked[:limit]

        filtered = [d for d in ranked if d.metadata.get("country") in (None, country)]

        country_doc = next(
            (d for d in self.documents if d.metadata.get("country") == country),
            None,
        )
        if country_doc is not None and country_doc not in filtered:
            filtered.insert(0, country_doc)

        return filtered[:limit]
