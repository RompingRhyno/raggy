"""Per-corpus runtime: one open DB, one lock, one question at a time.

The GUI holds several corpora and switches between them, which the library's
module-level vector-store cache cannot express (it caches exactly one store,
keyed by nothing). This module keeps a :class:`CorpusRuntime` per corpus id
instead: the validated settings, an open Chroma store, and a lock.

The lock is the important part. Indexing writes to ``chroma.sqlite3`` while a
question reads from it, both through the same handle, and neither is thread
safe; serializing them per corpus means a second question waits rather than
corrupting an in-flight index. Different corpora never touch the same files, so
they are free to run in parallel.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from langchain_chroma import Chroma

from ..api import retrieve_and_answer
from ..corpora import Corpus, CorpusError, CorpusStore, load_corpus_settings
from ..documents import build_spec
from ..indexing import chunk_counts, close_vectorstore, get_vectorstore, read_manifest
from ..loaders import _extracted_text, source_files, source_label
from ..pipeline import SCORE_KEY, source_filter
from ..raggy import ensure_models
from ..refresh import refresh_index
from ..render import CONVERTIBLE_EXTENSIONS, ExtractedTextCache, RenderCache

logger = logging.getLogger(__name__)


@dataclass
class StatusTracker:
    """The single status line a long operation reports on.

    ``progress`` callbacks from the library are written here and read by the
    GUI's poll endpoint, so the browser can show "indexing report.pdf ..." while
    the request that started the work is still blocking.
    """

    message: str = ""
    started_at: float | None = None
    updated_at: float | None = None
    updates: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __call__(
        self, message: str, completed: int | None = None, total: int | None = None
    ) -> None:
        with self._lock:
            self.message = message
            self.updated_at = time.time()
            self.updates += 1

    def start(self, message: str = "") -> None:
        with self._lock:
            self.started_at = time.time()
            self.updated_at = self.started_at
            self.message = message
            self.updates = 0

    def as_dict(self, busy: bool = False) -> dict:
        with self._lock:
            elapsed = None
            if self.started_at is not None:
                elapsed = round(
                    (self.updated_at or self.started_at) - self.started_at, 2
                )
            return {
                "busy": busy,
                "message": self.message,
                "elapsed_seconds": elapsed,
                "updates": self.updates,
            }


class CorpusRuntime:
    """Everything needed to index and query one corpus."""

    def __init__(self, corpus: Corpus, store: CorpusStore):
        self.corpus = corpus
        self.store = store
        self.settings: dict[str, Any] = load_corpus_settings(corpus)
        self.status = StatusTracker()
        self._lock = threading.Lock()
        self._vectorstore: Chroma | None = None

    # -- lifecycle ---------------------------------------------------------
    @property
    def db_directory(self) -> str:
        return str(self.corpus.db_directory)

    def close(self) -> None:
        """Release the Chroma handle (called when a corpus is replaced/removed)."""
        if self._vectorstore is not None:
            close_vectorstore(self._vectorstore)
            self._vectorstore = None

    def vectorstore(self) -> Chroma:
        """The open store, opening it (without indexing) on first use."""
        if self._vectorstore is None:
            self._vectorstore = get_vectorstore(
                self.db_directory, self.settings["embedding_model"]
            )
        return self._vectorstore

    # -- indexing ----------------------------------------------------------
    def refresh(self) -> Any:
        """Re-index the corpus, returning the run's :class:`~raggy.indexing.IndexReport`.

        Raises ``RuntimeError`` if another long operation already holds the
        corpus, which the server turns into a 409 rather than letting two
        indexers fight over one SQLite file.
        """
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(
                f"'{self.corpus.name}' is busy with another operation; wait for it to finish."
            )
        try:
            self.status.start("checking models ...")
            ensure_models(self.settings, progress=self.status)
            self.status("scanning sources ...")
            self.close()
            vectorstore, report = refresh_index(
                db_directory=self.db_directory,
                embedding_model=self.settings["embedding_model"],
                sources=self.settings["sources"],
                chunk_size=self.settings["chunk_size"],
                chunk_overlap=self.settings["chunk_overlap"],
                embed_batch_size=self.settings["embed_batch_size"],
                progress=self.status,
                convert=True,
                retry_failed=True,
                allow_missing=False,
            )
            self._vectorstore = vectorstore
            self.status(f"done: {report.summary}")
            return report
        finally:
            self._lock.release()

    # -- questions ---------------------------------------------------------
    def query(
        self,
        text: str,
        include_sources: list[str] | None = None,
        exclude_sources: list[str] | None = None,
    ) -> dict:
        """Answer ``text`` and return the answer with its clickable citations.

        ``include_sources``/``exclude_sources`` are the ask pane's scope: one
        file instead of the corpus, minus whatever the user clicked off. A
        question whose scope selects nothing is refused here, before the models
        are pulled, rather than being answered from an empty context — the
        answer to "nothing" is a confident-sounding "I cannot find that", which
        reads as a fact about the corpus instead of about the selection.
        """
        # Reading the corpus's file list costs a pass over its chunks, so it is
        # only paid when the question actually carries a scope. `known` stays
        # unread — and `all_sources` unset — for "ask the whole corpus".
        scoped = bool(include_sources or exclude_sources)
        known = self._known_sources() if scoped else None
        if scoped:
            self._refuse_empty_scope(include_sources, exclude_sources, known)

        if not self._lock.acquire(blocking=False):
            raise RuntimeError(
                f"'{self.corpus.name}' is busy indexing; try again once it finishes."
            )
        try:
            self.status.start("retrieving ...")
            self.status("pulling models if needed ...")
            ensure_models(self.settings, progress=self.status)
            self.status("retrieving and generating ...")
            answer, docs = retrieve_and_answer(
                text,
                self.settings,
                self.vectorstore(),
                progress=self.status,
                include_sources=include_sources,
                exclude_sources=exclude_sources,
                all_sources=known,
            )
            self.status("done")
            return {
                "answer": answer,
                "citations": [self.citation(doc) for doc in docs],
                "corpus": self.corpus.id,
                "query": text,
            }
        finally:
            self._lock.release()

    def _refuse_empty_scope(
        self,
        include_sources: list[str] | None,
        exclude_sources: list[str] | None,
        known: list[str],
    ) -> None:
        """Raise when a question's scope names no file this corpus can search.

        ``known`` is what an exclusion is resolved against, so an exclusion of
        everything the corpus has shows up here as an empty allow-list. The
        browser is handed the server's own paths, so a miss can only mean a stale
        file list — a file deleted (by a refresh) or hidden since the scope was
        chosen — and a 400 that says so beats an answer built from nothing.
        """
        _, selected = source_filter(include_sources, exclude_sources, known)
        if selected is None:
            return
        if not any(os.path.normcase(path) in selected for path in known):
            raise ValueError(
                "no files are selected for this question: the chosen file is "
                "not in the corpus, or it is hidden from context"
            )

    def _known_sources(self) -> list[str]:
        """Every source path the corpus knows: its chunks and its manifest.

        Both are consulted because a file that is on disk but not yet indexed
        has no chunks, and "that file is not in the corpus" would be a lie told
        about a file the file list is showing. This is also what an exclusion is
        resolved against (see :func:`raggy.pipeline.source_filter`).
        """
        known = set(self._manifest_sources())
        try:
            known |= set(chunk_counts(self.vectorstore()))
        except Exception as e:  # noqa: BLE001 - a scope check must not break queries
            logger.warning(
                "Could not read chunk sources for '%s': %s", self.corpus.id, e
            )
        return sorted(known)

    def _manifest_sources(self) -> list[str]:
        """The paths recorded in the manifest, indexed or failed."""
        manifest = read_manifest(self.db_directory) or {}
        paths: list[str] = []
        for key in ("files", "failed"):
            files = manifest.get(key)
            if isinstance(files, dict):
                paths.extend(str(path) for path in files)
        return paths

    def citation(self, doc) -> dict:
        """One retrieved chunk as a citation: label, snippet, score, jump target."""
        target = None
        try:
            target = build_spec(
                doc, self.db_directory, self.content_url, self.corpus.id
            ).as_dict()
        except (ValueError, OSError) as e:
            logger.warning("Could not build a jump target for a citation: %s", e)
        return {
            "label": _label(doc),
            "snippet": " ".join((doc.page_content or "").split())[:280],
            "text": doc.page_content or "",
            "score": doc.metadata.get(SCORE_KEY),
            "metadata": {
                key: doc.metadata[key]
                for key in ("source", "page", "start_line", "end_line", "source_kind")
                if key in doc.metadata
            },
            "target": target,
        }

    # -- content -----------------------------------------------------------
    def content_url(
        self, path: str, corpus_id: str | None = None, cache_file: bool = False
    ) -> str:
        """URL that serves ``path`` to the browser (see ``GET /api/corpora/{id}/file``)."""
        from urllib.parse import quote

        corpus_id = corpus_id or self.corpus.id
        flag = "&cache=1" if cache_file else ""
        return f"/api/corpora/{quote(corpus_id)}/file?path={quote(str(path))}{flag}"

    def authorize_path(self, path: str, cache_file: bool = False) -> Path | None:
        """Resolve ``path`` if the corpus is allowed to serve it, else None.

        Only files under the corpus's own sources, or inside its own render
        cache, can be read: a request naming ``~/.ssh/id_rsa`` must not become a
        file-read primitive just because the server runs locally.
        """
        target = Path(path).expanduser()
        try:
            resolved = target.resolve()
        except OSError:
            return None

        roots = [Path(source).resolve() for source in self.corpus.sources]
        cache_root = (Path(self.db_directory) / "render_cache").resolve()
        roots.append(cache_root)
        for root in roots:
            if (resolved == root or root in resolved.parents) and resolved.is_file():
                return resolved
        return None

    def documents(self) -> dict:
        """The corpus's indexed files, plus source files not indexed yet."""
        manifest = read_manifest(self.db_directory) or {}
        files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
        skipped = (
            manifest.get("failed") if isinstance(manifest.get("failed"), dict) else {}
        )

        counts: dict[str, int] = {}
        opened = self._vectorstore is not None or bool(files)
        if opened and files:
            try:
                counts = chunk_counts(self.vectorstore())
            except Exception as e:  # noqa: BLE001 - listing must not fail the UI
                logger.warning(
                    "Could not read chunk counts for '%s': %s", self.db_directory, e
                )

        entries = []
        for path in sorted(files):
            entries.append(self._document_entry(path, counts.get(path, 0), "indexed"))
        for path, error in sorted(skipped.items()):
            entries.append(self._document_entry(path, 0, "failed", detail=str(error)))
        for path in self._unindexed_sources():
            entries.append(self._document_entry(path, 0, "new"))

        present = {entry["path"] for entry in entries}
        for path in sorted(set(files) - present):
            entries.append(self._document_entry(path, counts.get(path, 0), "indexed"))

        entries.sort(key=lambda entry: (entry["kind"], entry["name"].lower()))
        # A file with no chunks and no text to show for itself: either it could
        # not be read, or nothing extractable came out of it. Such a file
        # contributes nothing to an answer, so the GUI keeps it out of the list
        # behind a toggle — and it is told the count rather than re-deriving the
        # rule. A file that is merely *new* (on disk, not indexed yet) is not one
        # of these: it has not been read, so its text is unknown, not absent.
        without_text = [
            entry
            for entry in entries
            if not entry["chunks"] and entry["status"] != "new"
        ]
        # What OCR reads out of each image, so the viewer can show it beside the
        # image: an image is one document and never split, so its text has no
        # chunk to travel in, and without this every OCR panel would be empty.
        # Present for every image file — empty when there is nothing to read.
        for entry in entries:
            if entry["kind"] == "image" and Path(entry["path"]).exists():
                entry["ocr_text"] = self._load_image_document(entry["path"])
        return {
            "corpus": self.corpus.id,
            "name": self.corpus.name,
            "db_directory": self.db_directory,
            "indexed": (Path(self.db_directory) / "manifest.yaml").exists(),
            "counts": {
                "documents": len(entries),
                "chunks": sum(counts.values()),
                "failed": len(skipped),
                "without_text": len(without_text),
            },
            "documents": entries,
        }

    def _unindexed_sources(self) -> list[str]:
        """Source files on disk that the manifest does not know about yet."""
        manifest = read_manifest(self.db_directory) or {}
        attempted = manifest.get("attempted")
        if not isinstance(attempted, dict):
            attempted = (
                manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
            )
        try:
            present = source_files(self.corpus.sources)
        except FileNotFoundError:
            return []
        return [str(path) for path in present if str(path) not in attempted]

    def _load_image_document(self, path: str) -> str:
        """The text OCR read out of an image file; an empty string if none.

        Read from the corpus's extracted-text cache, which indexing fills as it
        OCRs each image (`ExtractedTextCache`). The alternative — re-running OCR
        here — is what made listing an image-heavy corpus take seconds: a pass
        over every image, on every listing, for text that had already been read
        once. Reading the vector store instead does not work: an image's text is
        split into chunks like any other, and stitching them back is not exact.

        A miss (a corpus indexed before the cache existed, or an image whose
        bytes changed since) costs one OCR pass, whose result is then cached, so
        the second listing is fast even then. A failure costs the viewer its text
        panel, never the file list.
        """
        try:
            return _extracted_text(
                Path(path), ExtractedTextCache(self.db_directory)
            ).strip()
        except Exception as e:  # noqa: BLE001 - OCR must not break the listing
            logger.warning("Could not read the text of '%s' for display: %s", path, e)
            return ""

    def _document_entry(
        self, path: str, chunks: int, status: str, detail: str = ""
    ) -> dict:
        """One row of the file list, with the URL its viewer should open.

        For a converted document that URL is the **cached render**, not the
        source: the viewer is a PDF viewer, and handing it a ``.docx`` is a
        guaranteed "could not be read as a PDF". This is the file-list twin of
        what the citation path does from the chunk's own metadata.
        """
        file_path = Path(path)
        return {
            "path": path,
            "name": file_path.name,
            "kind": _kind_of(file_path),
            "exists": file_path.exists(),
            "chunks": chunks,
            "status": status,
            "detail": detail,
            "size": file_path.stat().st_size if file_path.exists() else None,
            "content_url": self._viewer_url(path),
            "source_url": self.content_url(path),
        }

    def _viewer_url(self, path: str) -> str:
        """The URL a viewer should open for the source file ``path``."""
        suffix = Path(path).suffix.lower()
        if suffix not in CONVERTIBLE_EXTENSIONS:
            return self.content_url(path)
        try:
            rendered = RenderCache(self.db_directory).cached_path(path)
        except OSError:
            rendered = None
        if rendered is None:
            return self.content_url(path)
        return self.content_url(str(rendered), cache_file=True)


class AppState:
    """Owns one :class:`CorpusRuntime` per corpus and the active corpus choice."""

    def __init__(self, store: CorpusStore):
        self.store = store
        self._runtimes: dict[str, CorpusRuntime] = {}
        self._lock = threading.RLock()

    def runtime(self, corpus_id: str | None = None) -> CorpusRuntime:
        """The runtime for ``corpus_id`` (default: the active corpus)."""
        with self._lock:
            corpus = self.store.get(corpus_id) if corpus_id else self.store.active()
            if corpus is None:
                raise CorpusError(
                    f"unknown corpus: {corpus_id}"
                    if corpus_id
                    else "no corpus is configured"
                )
            runtime = self._runtimes.get(corpus.id)
            if runtime is None or runtime.corpus != corpus:
                if runtime is not None:
                    runtime.close()
                runtime = CorpusRuntime(corpus, self.store)
                self._runtimes[corpus.id] = runtime
            return runtime

    def activate(self, corpus_id: str) -> None:
        self.store.set_active(corpus_id)

    def drop(self, corpus_id: str) -> None:
        """Forget a corpus's runtime, closing its DB handle first."""
        with self._lock:
            runtime = self._runtimes.pop(corpus_id, None)
        if runtime is not None:
            runtime.close()

    def status(self, corpus_id: str | None = None) -> dict:
        runtime = self.runtime(corpus_id)
        payload = runtime.status.as_dict()
        payload["busy"] = runtime._lock.locked()
        payload["corpus"] = runtime.corpus.id
        payload["name"] = runtime.corpus.name
        return payload

    def close(self) -> None:
        """Close every open store (used on shutdown)."""
        with self._lock:
            runtimes = list(self._runtimes.values())
            self._runtimes.clear()
        for runtime in runtimes:
            runtime.close()


def _label(doc) -> str:
    """Human-readable citation label, matching what the CLI prints."""
    return source_label(doc)


_KIND_BY_EXTENSION = {
    ".pdf": "pdf",
    ".docx": "document",
    ".pptx": "presentation",
    ".txt": "text",
    ".md": "text",
    ".markdown": "text",
    ".html": "web",
    ".htm": "web",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".bmp": "image",
}


def _kind_of(path: Path) -> str:
    return _KIND_BY_EXTENSION.get(path.suffix.lower(), "other")


def load_registry_summary(store: CorpusStore) -> dict:
    """Small helper for debugging: the registry as YAML sees it."""
    if not store.registry_path.exists():
        return {}
    return yaml.safe_load(store.registry_path.read_text(encoding="utf-8")) or {}
