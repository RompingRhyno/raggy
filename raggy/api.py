"""Programmatic entry point for a front end: query a corpus, index a corpus.

``run_pipeline``/``run_pipeline_stream`` in :mod:`raggy.raggy` are built around a
single global vector store resolved from a config *path* — fine for the CLI,
wrong for a GUI holding several corpora at once. The functions here take the
already-built pieces (validated settings and an open store) so a front end can
own the lifecycle: it decides when a corpus is (re)indexed and keeps each
corpus's store alive between questions.

No chat history is involved. Multi-turn memory is an explicit non-goal for the
GUI, and the chain built here is the same retrieval/rerank/generation path the
CLI uses, minus the conversation.
"""

import logging
from collections.abc import Sequence
from typing import Any

from langchain_chroma import Chroma

from .llm_factory import ensure_ollama_model
from .pipeline import build_rag_chain
from .progress import ProgressCallback

logger = logging.getLogger(__name__)


def retrieve_and_answer(
    query: str,
    cfg: dict[str, Any],
    vectorstore: Chroma,
    progress: ProgressCallback | None = None,
    include_sources: Sequence[str] | None = None,
    exclude_sources: Sequence[str] | None = None,
    all_sources: Sequence[str] | None = None,
) -> tuple[str, list]:
    """Answer ``query`` from ``vectorstore``; return ``(answer, retrieved_docs)``.

    ``retrieved_docs`` are the exact chunks the model saw, in order, each
    carrying its source path, page or line range, and reranker score — which is
    what the GUI turns into clickable citations. ``cfg`` is a validated settings
    dict (see :func:`raggy.corpora.load_corpus_settings`).

    ``include_sources``/``exclude_sources`` scope retrieval to a set of files
    (empty/None searches everything), and ``all_sources`` is the corpus's file
    list, which an exclusion needs to become an exact allow-list (see
    :func:`raggy.pipeline.source_filter`). Restricting retrieval is not the same
    as restricting the answer: the model still answers from whatever it is
    given, and the citations say which file each passage came from.

    Generation runs once and returns a whole answer: token-by-token streaming is
    a non-goal for this build.
    """
    if cfg["llm_provider"] == "ollama":
        ensure_ollama_model(cfg["llm_model"], progress=progress)

    doc_sink: list = []
    rag_chain, _ = build_rag_chain(
        vectorstore=vectorstore,
        llm_model=cfg["llm_model"],
        llm_provider=cfg["llm_provider"],
        system_prompt=cfg["system_prompt"],
        retrieve_k=cfg["retrieve_k"],
        llm_temperature=cfg["llm_temperature"],
        rerank_model=cfg["rerank_model"],
        rerank_k=cfg["rerank_k"],
        rerank_threshold=cfg["rerank_threshold"],
        db_directory=cfg["db_directory"],
        hybrid_alpha=cfg["hybrid_alpha"],
        doc_sink=doc_sink,
        chat_history=None,
        include_sources=include_sources,
        exclude_sources=exclude_sources,
        all_sources=all_sources,
    )
    answer = rag_chain.invoke({"question": query, "chat_history": []})
    return answer, (doc_sink[-1] if doc_sink else [])
