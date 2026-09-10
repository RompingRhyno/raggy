import logging
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from .bm25 import save_bm25_index
from .hashes import HASH_BLOCK_SIZE, hash_file
from .loaders import (
    FileFailure,
    FileSkip,
    LoadReport,
    annotate_line_numbers,
    load_documents,
    should_annotate_lines,
    source_files,
)
from .progress import ProgressCallback

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.yaml"

# Manifest keys whose change invalidates every stored embedding, so the DB can
# only be rebuilt from scratch. Everything else (which files exist and what
# they contain) is reconciled file-by-file.
#
# ``file_converters`` is in here because it decides what text the chunks are
# made of (a converted PDF's text instead of the source DOCX's), so a change to
# the conversion setup makes every stored chunk suspect. ``attempted``/``failed``
# are bookkeeping for the file diff and deliberately excluded: they are derived
# from the walk, not configuration.
_REBUILD_KEYS = ("chunk_size", "chunk_overlap", "embedding_model", "file_converters")


@dataclass(frozen=True)
class IndexPlan:
    """What must happen to bring the DB in line with the current sources.

    ``full_rebuild`` means every stored embedding is invalid (no manifest, or a
    chunking/embedding-model/conversion change), so the DB dir is wiped and
    rebuilt. Otherwise the three file lists describe an incremental update.

    ``retry`` names files that are known to have failed to load and whose
    content has not changed since. They are not part of ``has_changes`` — a
    corpus with one unreadable file is not "stale" — but a caller that wants to
    keep trying (the GUI's explicit Refresh) reloads them too.
    """

    full_rebuild: bool = False
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    retry: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """True if the DB is out of date in any way."""
        return bool(self.full_rebuild or self.added or self.modified or self.removed)

    @property
    def reindexed(self) -> list[str]:
        """Files that must be loaded and embedded (whether or not they succeed)."""
        return self.added + self.modified

    def reload_paths(self, include_retry: bool = False) -> list[str]:
        """The files an update should read: changed ones, plus failed ones on request."""
        return self.reindexed + (self.retry if include_retry else [])


@dataclass
class IndexReport:
    """Per-file outcome of one index run, for a front end to display.

    Everything here is derived after the run from three sources: the walk the
    loaders performed (which files exist, and which extensions were skipped),
    the load results (what failed and why), and the DB itself (how many chunks
    each file contributed).

    ``indexed`` is what *this run* embedded — empty when the DB was already
    current, which is the answer to "what did pressing Refresh do?". ``stored``
    is the whole corpus now in the DB, so a front end can still show the
    contents of a corpus it did not just rebuild. A file that appears in
    ``stored`` with no chunks is reported as failed rather than silently counted
    as indexed.
    """

    full_rebuild: bool = False
    changed: bool = False
    indexed: list[str] = field(default_factory=list)
    stored: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    failed: list[FileFailure] = field(default_factory=list)
    skipped: list[FileSkip] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    chunks: dict[str, int] = field(default_factory=dict)
    orphans: list[str] = field(default_factory=list)

    @property
    def indexed_count(self) -> int:
        return len(self.indexed)

    @property
    def skipped_unsupported(self) -> list[FileSkip]:
        return [entry for entry in self.skipped if entry.reason == "unsupported"]

    @property
    def summary(self) -> str:
        """One-line summary of the run, e.g. ``3 indexed, 1 failed, 2 skipped``."""
        parts = [f"{len(self.indexed)} indexed"]
        if self.unchanged:
            parts.append(f"{len(self.unchanged)} unchanged")
        if self.removed:
            parts.append(f"{len(self.removed)} deleted")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped")
        if self.orphans:
            parts.append(f"{len(self.orphans)} orphaned")
        if not self.changed:
            parts.append(f"nothing to do ({len(self.stored)} file(s) up to date)")
        return ", ".join(parts)

    def detail_lines(self) -> list[str]:
        """One line per file that needs explaining, for a status panel."""
        lines: list[str] = []
        if self.full_rebuild:
            lines.append("Full rebuild: chunking, embedding, or conversion changed.")
        for path in self.indexed[:20]:
            lines.append(
                f"indexed   {Path(path).name} ({self.chunks.get(path, 0)} chunks)"
            )
        for path in self.unchanged[:20]:
            lines.append(f"unchanged {Path(path).name}")
        for path in self.removed[:20]:
            lines.append(f"removed   {Path(path).name}")
        for failure in self.failed[:20]:
            lines.append(f"failed    {Path(failure.path).name}: {failure.error}")
        for entry in self.skipped[:20]:
            lines.append(f"skipped   {Path(entry.path).name} ({entry.reason})")
        for path in self.orphans[:20]:
            lines.append(f"orphaned  {Path(path).name} (in DB, not on disk)")
        return lines

    def as_dict(self) -> dict:
        """JSON-ready form of the report."""
        return {
            "full_rebuild": self.full_rebuild,
            "changed": self.changed,
            "summary": self.summary,
            "counts": {
                "indexed": len(self.indexed),
                "stored": len(self.stored),
                "unchanged": len(self.unchanged),
                "removed": len(self.removed),
                "failed": len(self.failed),
                "skipped": len(self.skipped),
                "orphans": len(self.orphans),
                "chunks": sum(self.chunks.values()),
            },
            "indexed": [
                {"path": path, "chunks": self.chunks.get(path, 0)}
                for path in self.indexed
            ],
            "stored": [
                {"path": path, "chunks": self.chunks.get(path, 0)}
                for path in self.stored
            ],
            "unchanged": list(self.unchanged),
            "removed": list(self.removed),
            "failed": [{"path": f.path, "error": f.error} for f in self.failed],
            "skipped": [
                {"path": s.path, "reason": s.reason, "detail": s.detail}
                for s in self.skipped
            ],
            "orphans": list(self.orphans),
        }


def get_embeddings(model_name: str) -> OllamaEmbeddings:
    """Initialize and return OllamaEmbeddings."""
    return OllamaEmbeddings(model=model_name)


def get_vectorstore(db_directory: str, embedding_model: str) -> Chroma:
    """Initialize and return the Chroma vector store."""
    embeddings = get_embeddings(embedding_model)
    return Chroma(persist_directory=db_directory, embedding_function=embeddings)


def close_vectorstore(vectorstore: Chroma) -> None:
    """Release the client's underlying DB connection.

    Chroma holds an open handle to ``chroma.sqlite3``. Deleting the DB
    dir while that handle is open leaves a stale connection to a removed
    inode, and the next write fails with "attempt to write a readonly
    database". Call this before wiping/rebuilding the DB from source.
    """
    client = getattr(vectorstore, "_client", None)
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _split_documents(
    docs: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split loaded documents into overlapping chunks, annotating line numbers."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits: list[Document] = []
    for doc in docs:
        doc_splits = text_splitter.split_documents([doc])
        if should_annotate_lines(doc):
            annotate_line_numbers(doc_splits, doc.page_content)
        splits.extend(doc_splits)
    return splits


def _embed_in_batches(
    splits: list[Document],
    vectorstore: Chroma,
    embed_batch_size: int,
    progress: ProgressCallback | None = None,
) -> None:
    """Embed ``splits`` into Chroma in batches of at most ``embed_batch_size`` chunks.

    The number of batches is derived dynamically from the total chunk count, so
    the setting stays sensible regardless of dataset size.

    ``progress`` reports the batches on the same single status line the ingest
    step uses; the tqdm bar is suppressed then, since two live progress
    displays would fight over the same terminal line.
    """
    total_splits = len(splits)
    n_batches = math.ceil(total_splits / embed_batch_size)
    batch_sizes = [
        min(embed_batch_size, total_splits - i * embed_batch_size)
        for i in range(n_batches)
    ]

    logger.info(
        "Embedding chunks into Chroma (%d batches of <= %d)...",
        n_batches,
        embed_batch_size,
    )

    remaining = list(splits)
    for index, size in enumerate(
        tqdm(
            batch_sizes,
            total=n_batches,
            desc="Indexing Batches",
            disable=progress is not None or not sys.stderr.isatty(),
        ),
        start=1,
    ):
        if progress is not None:
            progress(f"[{index}/{n_batches}] embedding chunks ...")
        batch = remaining[:size]
        remaining = remaining[size:]
        vectorstore.add_documents(batch)


def _load_into_splits(
    paths: list[str],
    chunk_size: int,
    chunk_overlap: int,
    progress: ProgressCallback | None,
    report: LoadReport | None,
    loader=None,
    on_missing: str = "raise",
) -> list[Document]:
    """Load ``paths`` and split them, recording what loaded and what failed.

    The returned splits carry the same ``source`` metadata the loaders stamped,
    which is what :func:`chunk_counts` reads back to attribute stored chunks to
    files. Paths that fail to load are recorded in ``report`` and contribute
    nothing.

    A file that loaded but produced no chunk is recorded in
    ``report.chunkless``: it has nothing to retrieve, so the run reports it as a
    failure rather than counting an unsearchable document as indexed.
    """
    docs = load_documents(
        paths,
        progress=progress,
        on_missing=on_missing,  # type: ignore[arg-type]
        report=report,
        **({"loader": loader} if loader is not None else {}),
    )
    splits = _split_documents(docs, chunk_size, chunk_overlap)
    if report is not None:
        produced = {
            str(doc.metadata["source"]) for doc in splits if doc.metadata.get("source")
        }
        report.chunkless.extend(
            path
            for path in dict.fromkeys(report.attempted_paths())
            if path not in produced
        )
    return splits


def chunk_counts(vectorstore: Chroma) -> dict[str, int]:
    """Count the chunks currently stored per source file.

    This is what lets a refresh report say which files actually made it into the
    DB (and, by omission, which ones loaded but produced no text) instead of
    trusting the load step's own claim of success.
    """
    counts: dict[str, int] = {}
    for chunk in _collection_chunks(vectorstore):
        source = chunk.metadata.get("source")
        if source:
            counts[str(source)] = counts.get(str(source), 0) + 1
    return counts


def create_index(
    sources: list[str],
    vectorstore: Chroma,
    chunk_size: int,
    chunk_overlap: int,
    embed_batch_size: int,
    db_directory: str,
    progress: ProgressCallback | None = None,
    report: LoadReport | None = None,
    loader=None,
    on_missing: str = "raise",
) -> None:
    """Build the index from scratch: load, split, and embed every source file.

    The full-build counterpart to ``update_index``, which applies only the
    changed files to an index that already exists.

    Each entry in ``sources`` may point to a single supported file (e.g.
    .txt/.md/.pdf/.docx/.pptx/.html/.png) or to a directory containing multiple
    supported files.

    Also builds a BM25 index over the same chunks and persists it to
    ``<db_directory>/bm25_index/`` for hybrid retrieval.

    ``progress`` receives one status line per file read and per embedded batch.
    ``report`` collects which files loaded, failed, or were skipped, and
    ``loader`` overrides how one path becomes documents (the DOCX/PPTX
    conversion path reads a cached PDF instead of the source file).
    ``on_missing`` decides whether a disappeared source entry is an error.
    """
    splits = _load_into_splits(
        sources, chunk_size, chunk_overlap, progress, report, loader, on_missing
    )

    if not splits:
        logger.warning("No documents found to index.")
        return

    logger.info("Split text into %d chunks.", len(splits))
    _embed_in_batches(splits, vectorstore, embed_batch_size, progress)
    save_bm25_index(splits, db_directory)


# Chroma binds one SQL variable per returned row, and SQLite caps a statement
# at 32766 of them, so an unpaged get() over a large collection fails with
# "too many SQL variables". Read the corpus back one page at a time instead.
_COLLECTION_PAGE_SIZE = 10000


def _collection_chunks(vectorstore: Chroma) -> list[Document]:
    """Return every chunk currently stored in Chroma as a ``Document``.

    Reading the stored text back (rather than re-splitting the corpus) is what
    lets the BM25 index be rebuilt after an incremental update without
    touching files that did not change.
    """
    chunks: list[Document] = []
    offset = 0
    while True:
        stored = vectorstore._collection.get(
            include=["documents", "metadatas"],
            limit=_COLLECTION_PAGE_SIZE,
            offset=offset,
        )
        documents = stored.get("documents") or []
        metadatas = stored.get("metadatas") or []
        chunks.extend(
            Document(page_content=text, metadata=dict(metadata or {}))
            for text, metadata in zip(documents, metadatas)
        )
        if len(documents) < _COLLECTION_PAGE_SIZE:
            return chunks
        offset += len(documents)


def _delete_chunks_for_files(vectorstore: Chroma, files: list[str]) -> None:
    """Delete every stored chunk whose ``source`` metadata is one of ``files``."""
    if not files:
        return
    vectorstore._collection.delete(where={"source": {"$in": files}})


def update_index(
    vectorstore: Chroma,
    plan: IndexPlan,
    chunk_size: int,
    chunk_overlap: int,
    embed_batch_size: int,
    db_directory: str,
    progress: ProgressCallback | None = None,
    report: LoadReport | None = None,
    reload: list[str] | None = None,
    loader=None,
) -> None:
    """Apply ``plan`` to an existing DB without re-embedding untouched files.

    Chunks belonging to removed or modified files are deleted from Chroma,
    then added and modified files are re-loaded, split, and embedded. The BM25
    index has no incremental update path, so it is rebuilt from the chunks now
    stored in Chroma — cheap, since that requires no embedding calls.

    ``reload`` overrides which files are re-read, defaulting to the plan's
    changed files. A caller passes ``plan.reload_paths(include_retry=True)`` to
    also retry files that failed on an earlier run. ``report`` collects the
    per-file outcome (see :func:`create_index`).
    """
    stale = plan.removed + plan.modified
    if stale:
        logger.info("Removing chunks for %d changed/deleted file(s).", len(stale))
        _delete_chunks_for_files(vectorstore, stale)

    reindexed = plan.reindexed if reload is None else reload
    if reindexed:
        logger.info("Indexing %d new/changed file(s).", len(reindexed))
        # A file fingerprinted earlier this run may already be gone; skipping
        # it beats aborting the update, and the next run records the deletion.
        splits = _load_into_splits(
            reindexed,
            chunk_size,
            chunk_overlap,
            progress,
            report,
            loader,
            on_missing="skip",
        )
        if splits:
            _embed_in_batches(splits, vectorstore, embed_batch_size, progress)
        else:
            logger.warning("No content found in the new/changed files.")

    save_bm25_index(_collection_chunks(vectorstore), db_directory)


# Read in 1 MiB blocks rather than slurping whole files: the corpus can
# include large PDFs/images, and every file is hashed on each pipeline start.
# (hashlib.file_digest would replace this loop, but it needs Python 3.11.)
_HASH_BLOCK_SIZE = HASH_BLOCK_SIZE


def _hash_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's contents."""
    return hash_file(path)


def file_fingerprints(sources: list[str], on_missing: str = "raise") -> dict[str, str]:
    """Return a ``{file path: content hash}`` map of the source file set.

    Hashing per file (rather than folding everything into one digest) is what
    makes incremental indexing possible: comparing this map against the one in
    the manifest names exactly which files were added, modified, or deleted.
    Cheap enough to run on every pipeline start.

    The file set comes from :func:`raggy.loaders.source_files`, the same walk
    the loaders use, so these keys are exactly the ``source`` metadata values
    stored on the chunks they fingerprint. ``on_missing`` is passed through to
    that walk (``"skip"`` tolerates a source that disappeared).
    """
    fingerprints: dict[str, str] = {}
    for file_path in source_files(sources, on_missing=on_missing):  # type: ignore[arg-type]
        try:
            fingerprints[str(file_path)] = hash_file(file_path)
        except OSError:
            continue
    return fingerprints


def _load_manifest(db_directory: str) -> dict | None:
    """Read the build manifest (if any) from the DB directory."""
    manifest_path = Path(db_directory) / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    return yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}


def read_manifest(db_directory: str) -> dict | None:
    """Public form of :func:`_load_manifest`, for front ends that report on it."""
    return _load_manifest(db_directory)


def _write_manifest(db_directory: str, index_cfg: dict) -> None:
    """Persist the index-affecting config so later runs can detect drift."""
    Path(db_directory).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(db_directory) / MANIFEST_FILENAME
    manifest_path.write_text(yaml.safe_dump(index_cfg), encoding="utf-8")


def _reset_db_directory(db_directory: str) -> None:
    """Physically delete all contents of the DB directory.

    ``reset_collection()`` only removes records but leaves stale segment /
    version files behind in the DB dir, so repeated rebuilds accumulate
    garbage. A full wipe of the directory is the clean way to rebuild.
    """
    root = Path(db_directory)
    if not root.exists():
        return
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def build_index_config(
    sources: list[str],
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    fingerprints: dict[str, str] | None = None,
    report: LoadReport | None = None,
    file_converters: dict[str, str] | None = None,
) -> dict:
    """Build the index-affecting config (including fresh per-file fingerprints).

    The returned dict is compared against the persisted ``manifest.yaml`` to
    decide whether the DB is stale and, if so, which files need re-indexing.

    Three keys describe the file set, and the difference between them is what
    makes failures visible instead of silently permanent:

    - ``files`` — the files whose chunks are in the DB (they loaded, and
      produced text);
    - ``failed`` — the files a load pass read and could not use, with the
      reason (the loader's error, or "no text came out of it");
    - ``attempted`` — every walkable file seen, whether it was usable or not.
      This is what the next run diffs against, so an unreadable file stays
      unreadable-but-known rather than looking like a brand-new file on every
      run, while a *changed* unreadable file is still retried.

    ``fingerprints`` may be passed to reuse a walk already performed this run;
    ``report`` (from the load pass that just ran) is what names the failed and
    chunkless files. ``file_converters`` records the DOCX/PPTX → PDF converters
    in play, since a change there changes the text every chunk is made of.
    """
    fingerprints = (
        file_fingerprints(sources) if fingerprints is None else dict(fingerprints)
    )
    unusable: set[str] = set()
    if report is not None:
        unusable = {failure.path for failure in report.failed} | set(report.chunkless)
        unusable &= set(fingerprints)

    return {
        "sources": list(sources),
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "embedding_model": embedding_model,
        "file_converters": dict(file_converters or {}),
        "files": {
            path: digest
            for path, digest in fingerprints.items()
            if path not in unusable
        },
        # Fingerprints, not messages: this map is diffed on the next run, and the
        # process that reads it has no use for last run's error text (the report
        # carries that).
        "failed": {path: fingerprints[path] for path in unusable},
        "attempted": fingerprints,
    }


def plan_index_update(
    db_directory: str, index_cfg: dict, retry_failed: bool = False
) -> IndexPlan:
    """Diff the current index config against the stored manifest.

    A missing manifest, a manifest without per-file fingerprints (written by an
    older version), or a change to chunking/embedding/conversion settings forces
    a full rebuild. Everything else is reduced to the set of files that were
    added, modified, or deleted since the last build.

    Files counted as failed by the previous build are not treated as changes (an
    unreadable file does not make the DB stale), but ``retry_failed`` lists them
    under ``plan.retry`` so an explicit refresh can try them again.
    """
    stored = _load_manifest(db_directory)
    if stored is None:
        return IndexPlan(full_rebuild=True)

    if any(_rebuild_key_changed(stored, key, index_cfg) for key in _REBUILD_KEYS):
        return IndexPlan(full_rebuild=True)

    stored_files = _stored_fingerprints(stored)
    if stored_files is None:
        return IndexPlan(full_rebuild=True)

    stored_failed = stored.get("failed")
    if not isinstance(stored_failed, dict):
        stored_failed = {}

    current_files = index_cfg["files"]
    attempted = index_cfg.get("attempted") or {}

    # Files the last build tried and could not use. They are absent from the
    # stored ``files`` map, so without this they would look like new files on
    # every single run. ``failed`` records the file's content hash at the time,
    # so a file is a *known* failure only while its bytes are unchanged; editing
    # it clears the flag and the file rejoins the normal diff.
    known_failed = {
        path for path, digest in stored_failed.items() if attempted.get(path) == digest
    }

    return IndexPlan(
        added=sorted(set(current_files) - set(stored_files) - known_failed),
        modified=sorted(
            path
            for path, digest in current_files.items()
            if path not in known_failed
            and path in stored_files
            and stored_files[path] != digest
        ),
        removed=sorted(set(stored_files) - set(attempted) - known_failed),
        retry=sorted(known_failed),
    )


def _rebuild_key_changed(stored: dict, key: str, index_cfg: dict) -> bool:
    """True if a rebuild-worthy key differs between the manifest and now.

    A manifest written before a key existed has no value for it. That means
    "not configured" rather than "changed" for ``file_converters``, whose
    default is the empty mapping — upgrading raggy must not rebuild every
    user's DB for a feature they are not using.
    """
    absent_default: dict | None = {} if key == "file_converters" else None
    return stored.get(key, absent_default) != index_cfg[key]


def _stored_fingerprints(stored: dict) -> dict[str, str] | None:
    """The file map of a stored manifest, or None if it is unusable.

    Manifests written before ``failed``/``attempted`` existed only have
    ``files``, which is exactly the set of loadable files — the same thing the
    newer key holds — so they diff correctly without a rebuild.
    """
    for key in ("attempted", "files"):
        value = stored.get(key)
        if isinstance(value, dict):
            return value
    return None


def db_needs_rebuild(db_directory: str, index_cfg: dict) -> bool:
    """Return True if the stored manifest no longer matches the current sources.

    Covers both kinds of staleness (full rebuild and incremental update); use
    ``plan_index_update`` when the distinction matters.
    """
    return plan_index_update(db_directory, index_cfg).has_changes


@dataclass(frozen=True)
class IndexOutcome:
    """What one :func:`initialize_db` call decided to do, and what it wrote.

    Returned alongside the store so a caller that has to *report* on the run
    (the GUI's Refresh) can describe it without guessing: whether every
    embedding was thrown away, which files the plan moved, and the manifest
    config that was actually persisted — all computed before the manifest was
    overwritten with the new state.
    """

    vectorstore: Chroma
    plan: IndexPlan
    manifest: dict
    collection_was_empty: bool = False

    @property
    def full_rebuild(self) -> bool:
        """True if this run re-embedded the whole corpus from source."""
        return self.plan.full_rebuild or self.collection_was_empty


def initialize_db(
    db_directory: str,
    embedding_model: str,
    sources: list[str],
    chunk_size: int,
    chunk_overlap: int,
    embed_batch_size: int,
    progress: ProgressCallback | None = None,
    report: LoadReport | None = None,
    loader=None,
    file_converters: dict[str, str] | None = None,
    retry_failed: bool = False,
    on_missing: str = "raise",
) -> IndexOutcome:
    """
    Initializes the Chroma database.

    If the database is empty, or ``chunk_size``/``chunk_overlap``/
    ``embedding_model`` no longer match the persisted manifest.yaml, the
    DB directory is wiped and the docs are re-indexed from scratch. If
    only the source files changed, the DB is updated incrementally: just the
    added and modified files are embedded and the removed ones dropped.

    ``progress`` receives a status line for each step of that work (see
    :func:`create_index`); it is never called when the DB is already current.
    ``report`` collects which files loaded, failed, or were skipped — including
    files that are re-read for a changed manifest. ``retry_failed`` additionally
    re-reads files that failed on an earlier run even though they have not
    changed, which is what the GUI's explicit Refresh does. ``on_missing``
    decides whether a source entry that no longer exists is an error.
    ``loader`` overrides how one path becomes documents, and ``file_converters``
    records the DOCX/PPTX → PDF setup in play (both are what the conversion path
    supplies, and both are part of the manifest).

    Returns an :class:`IndexOutcome`; ``outcome.vectorstore`` is the open DB.
    """
    # Fingerprints are computed once here and reused for the manifest written at
    # the end of the run, so a refresh hashes the corpus once, not twice. Nothing
    # has to be *read* to plan: which files cannot be used is already recorded in
    # the manifest's ``failed`` map, so an unreadable file never looks new and
    # never has to be re-read just to be classified.
    fingerprints = file_fingerprints(sources, on_missing=on_missing)

    # Config that determines the index content; only these drive re-indexing.
    # ``file_converters`` is part of the comparison config (not just what gets
    # recorded): a converter appearing or disappearing changes the text the
    # chunks are made of, which is exactly a full-rebuild condition.
    index_cfg = build_index_config(
        sources,
        chunk_size,
        chunk_overlap,
        embedding_model,
        fingerprints,
        file_converters=file_converters,
    )
    plan = plan_index_update(db_directory, index_cfg, retry_failed=retry_failed)

    if plan.full_rebuild:
        logger.info(
            "Detected an index config change (chunk_size, chunk_overlap, "
            "embedding_model, or file_converters); rebuilding DB from source "
            "documents."
        )
        # Wipe the DB dir BEFORE opening any Chroma connection. Deleting
        # the sqlite files while a client holds an open handle leaves a stale
        # connection to a removed inode, which fails on the next write
        # ("attempt to write a readonly database").
        _reset_db_directory(db_directory)

    vectorstore = get_vectorstore(db_directory, embedding_model)
    collection_count = vectorstore._collection.count()

    if plan.full_rebuild or collection_count == 0:
        create_index(
            sources=sources,
            vectorstore=vectorstore,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embed_batch_size=embed_batch_size,
            db_directory=db_directory,
            progress=progress,
            report=report,
            loader=loader,
            on_missing=on_missing,
        )
    elif plan.has_changes or (retry_failed and plan.retry):
        reload_paths = plan.reload_paths(include_retry=retry_failed)
        logger.info(
            "Detected source changes (%d added, %d modified, %d deleted, "
            "%d retried); updating DB incrementally.",
            len(plan.added),
            len(plan.modified),
            len(plan.removed),
            len(plan.retry) if retry_failed else 0,
        )
        update_index(
            vectorstore=vectorstore,
            plan=plan,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embed_batch_size=embed_batch_size,
            db_directory=db_directory,
            progress=progress,
            report=report,
            reload=reload_paths,
            loader=loader,
        )
    else:
        logger.info("Loaded existing vector DB from '%s'.", db_directory)
        return IndexOutcome(
            vectorstore=vectorstore,
            plan=plan,
            manifest=_load_manifest(db_directory) or {},
        )

    # The manifest is written from what actually happened, so a file that failed
    # to load is recorded as failed (and retried next time) instead of being
    # written down as indexed.
    manifest = build_index_config(
        sources,
        chunk_size,
        chunk_overlap,
        embedding_model,
        fingerprints,
        report=report,
        file_converters=file_converters,
    )
    _write_manifest(db_directory, manifest)
    logger.info("Successfully wrote DB to '%s'.", db_directory)
    return IndexOutcome(
        vectorstore=vectorstore,
        plan=plan,
        manifest=manifest,
        collection_was_empty=collection_count == 0,
    )
