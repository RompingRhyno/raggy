"""BM25 lexical retrieval: building the persisted ``bm25s`` index and reading it.

The dense vector store (Chroma) has no lexical understanding, so hybrid
retrieval pairs it with a sparse BM25 pass. :func:`save_bm25_index` writes the
index to ``<db_directory>/bm25_index`` when the vector DB is built (see
:mod:`raggy.indexing`); :func:`get_bm25_retriever` loads it back at
retrieval time.
"""

import json
import os
import shutil
from pathlib import Path

import bm25s
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

BM25_INDEX_DIRNAME = "bm25_index"
METADATA_FILENAME = "chunks_metadata.json"


def save_bm25_index(splits: list[Document], db_directory: str) -> None:
    """Build and persist a ``bm25s`` index (plus chunk metadata) to disk.

    The index is written to ``<db_directory>/bm25_index`` at DB build
    time so the retrieval step can load it later without re-indexing. Corpus
    entries are aligned by index with the per-chunk metadata so the
    ``Bm25sRetriever`` can reconstruct the original ``Document``s.
    """
    index_dir = Path(db_directory) / BM25_INDEX_DIRNAME
    if not splits:
        # An empty corpus can't be indexed; drop the old index rather than
        # leaving one that would keep returning removed chunks.
        shutil.rmtree(index_dir, ignore_errors=True)
        return
    corpus = [split.page_content for split in splits]
    bm25 = bm25s.BM25()
    bm25.index(bm25s.tokenize(corpus, show_progress=False), show_progress=False)
    index_dir.mkdir(parents=True, exist_ok=True)
    bm25.save(str(index_dir), corpus=corpus, show_progress=False)
    metadata = [dict(split.metadata) for split in splits]
    (index_dir / METADATA_FILENAME).write_text(json.dumps(metadata), encoding="utf-8")


# How many extra hits a source-filtered pass asks for before discarding the
# chunks it is not allowed to return. BM25 is the cheap half of hybrid
# retrieval, so over-fetching costs little and keeps a filtered pass from
# coming back with fewer candidates than the budget asked for.
FILTER_OVERFETCH = 50


class Bm25sRetriever(BaseRetriever):
    """A LangChain retriever wrapping a persisted ``bm25s`` index.

    Loads the BM25 index and the per-chunk metadata saved alongside it during
    building, and returns ``Document`` objects whose ``page_content`` and
    ``metadata`` match the original indexed chunks.

    ``sources`` narrows the pass to the listed files (``None`` or empty means no
    restriction). The index has no filter of its own — ``bm25s`` ranks the
    whole corpus — so the restriction is applied to the hits, and the pass asks
    for more of them than it needs to keep the budget spendable (see
    :data:`FILTER_OVERFETCH`). Paths are compared as given, so callers pass them
    already ``normcase``-folded (see :func:`raggy.pipeline.source_filter`).
    """

    bm25: bm25s.BM25
    chunks_metadata: list[dict]
    k: int = 10
    sources: frozenset[str] | None = None

    def _get_relevant_documents(self, query: str) -> list[Document]:
        # bm25s raises if k exceeds the corpus size, so a small corpus (or a
        # large retrieval budget) would otherwise fail the query outright.
        want = min(self.k, len(self.chunks_metadata))
        if want <= 0:
            return []
        # A filtered pass has to rank past the hits it will discard, or the arm
        # comes back with fewer candidates than the budget it was given.
        ask = min(max(want, FILTER_OVERFETCH), len(self.chunks_metadata))
        k = ask if self.sources else want
        tokenized = bm25s.tokenize([query], show_progress=False)
        hits, _ = self.bm25.retrieve(
            tokenized, corpus=self.bm25.corpus, k=k, show_progress=False
        )
        docs: list[Document] = []
        for entry in hits[0]:
            idx = int(entry["id"])
            metadata = dict(self.chunks_metadata[idx])
            if self.sources and not _selected(
                str(metadata.get("source", "")), self.sources
            ):
                continue
            docs.append(Document(page_content=entry["text"], metadata=metadata))
            if len(docs) >= want:
                break
        return docs


def _selected(source: str, sources: frozenset[str]) -> bool:
    """True when ``source`` is one of the files a pass is allowed to return."""
    return os.path.normcase(source) in sources


def get_bm25_retriever(
    db_directory: str, k: int = 10, sources: frozenset[str] | None = None
) -> Bm25sRetriever:
    """Load the persisted ``bm25s`` index from ``db_directory``.

    ``sources`` restricts the pass to those files; pass paths already
    ``normcase``-folded (see :func:`raggy.pipeline.source_filter`).

    Raises ``FileNotFoundError`` if the index was never built (e.g. hybrid
    search enabled without ever running the DB build step).
    """
    index_dir = Path(db_directory) / BM25_INDEX_DIRNAME
    metadata_path = index_dir / METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found in '{index_dir}'. Rebuild the DB to "
            "restore the lexical half of hybrid retrieval."
        )

    bm25 = bm25s.BM25()
    bm25 = bm25.load(str(index_dir), load_corpus=True, show_progress=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return Bm25sRetriever(bm25=bm25, chunks_metadata=metadata, k=k, sources=sources)
