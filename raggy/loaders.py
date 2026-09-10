import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from langchain_community.document_loaders import (
    BSHTMLLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from .hashes import hash_file
from .progress import ProgressCallback

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".pdf",
    ".docx",
    ".pptx",
    ".html",
    ".htm",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

# The file types whose chunks get start_line/end_line annotations. Only
# formats whose loader returns the file verbatim qualify; see
# should_annotate_lines for why the rest carry no location metadata.
LINE_ANNOTATED_EXTENSIONS = {".txt", ".md", ".markdown"}

DEFAULT_OCR_DPI = 150


@dataclass(frozen=True)
class FileSkip:
    """A file the walk found but that raggy cannot use.

    ``reason`` is ``"unsupported"`` (an extension raggy has no loader for) or
    ``"missing"`` (a source entry that no longer exists). Kept as data rather
    than a log line so a front end can report what was left out instead of
    making the user read the log.
    """

    path: str
    reason: Literal["unsupported", "missing"]
    detail: str = ""


@dataclass(frozen=True)
class FileFailure:
    """A supported file that was read and failed to load."""

    path: str
    error: str


@dataclass
class LoadReport:
    """What a load pass did, per file.

    ``skipped`` carries the files the walk rejected (see :class:`FileSkip`),
    and ``failed`` the supported files whose loader raised. Callers that do not
    pass a report keep the previous behaviour: skips and failures are logged
    only.

    ``chunkless`` is filled in one step later, by the indexer: a file can load
    perfectly and still be worth reporting as a failure, because its text was
    empty (a blank DOCX) or nothing survived chunking. Such a file has no chunks
    to retrieve, so counting it as indexed would be a lie the corpus pays for.
    """

    documents: list[Document] = field(default_factory=list)
    skipped: list[FileSkip] = field(default_factory=list)
    failed: list[FileFailure] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    chunkless: list[str] = field(default_factory=list)

    @property
    def loaded_paths(self) -> list[str]:
        """The ``source`` of every file that produced at least one document."""
        seen: dict[str, None] = {}
        for doc in self.documents:
            source = doc.metadata.get("source")
            if source:
                seen.setdefault(str(source), None)
        return list(seen)

    def attempted_paths(self) -> list[str]:
        """Every supported file the walk handed to a loader, in walk order."""
        return list(self.paths)


def _record_unsupported(skipped: list[Path], report: LoadReport | None) -> None:
    """Mirror the walk's skipped files into ``report``, keyed by path."""
    if report is None:
        return
    known = {entry.path for entry in report.skipped}
    for path in skipped:
        if str(path) not in known:
            report.skipped.append(
                FileSkip(
                    path=str(path), reason="unsupported", detail=path.suffix.lower()
                )
            )


def _record_missing(missing: list[Path], report: LoadReport | None) -> None:
    """Mirror source entries that no longer exist into ``report``."""
    if report is None:
        return
    known = {entry.path for entry in report.skipped}
    for path in missing:
        if str(path) not in known:
            report.skipped.append(FileSkip(path=str(path), reason="missing"))


def file_failures(report: LoadReport) -> tuple[dict[str, str], dict[str, str]]:
    """Split a load report into ``(attempted, failed)`` fingerprint-shaped maps.

    Both maps are ``{file path: sha256}`` — for every file the walk handed to a
    loader, and for the subset that contributed nothing. They are fingerprints
    rather than messages because the manifest diffs them on the next run; the
    *reason* each file failed is what :func:`failure_reasons` returns. Files that
    vanished between the walk and here are dropped rather than recorded: they are
    a race, not a property of the corpus.
    """
    suspect = {failure.path for failure in report.failed} | set(report.chunkless)

    attempted: dict[str, str] = {}
    for path in set(report.attempted_paths()) | suspect:
        try:
            attempted[path] = hash_file(path)
        except OSError:
            continue

    failed = {path: attempted[path] for path in suspect if path in attempted}
    return attempted, failed


def failure_reasons(report: LoadReport) -> dict[str, str]:
    """Why each failed file contributed nothing, keyed by path.

    A file fails either because its loader raised or because nothing survived
    extraction and chunking (a blank DOCX, an image with no recognizable text).
    Both mean the same thing to a user looking at the report, so both are
    reported here; the distinction between them is only in the message.
    """
    reasons = {failure.path: failure.error for failure in report.failed}
    for path in report.chunkless:
        reasons.setdefault(path, "no text could be extracted from this file")
    return reasons


_ocr_engine = None


def _get_ocr_engine():
    """Return a lazily-initialized shared RapidOCR engine instance."""
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr import RapidOCR

        # RapidOCR logs every model load at INFO; keep it out of the CLI output.
        _ocr_engine = RapidOCR(params={"Global.log_level": "error"})
    return _ocr_engine


def _ocr_image_bytes(image_bytes: bytes) -> str:
    """Run OCR on raw image bytes and return the recognized text."""
    result = _get_ocr_engine()(image_bytes)
    if not result.txts:
        return ""
    return "\n".join(text for text in result.txts if text)


def _load_image(path: Path, text_cache=None) -> list[Document]:
    """OCR a single image file into one Document.

    ``text_cache`` is the :class:`raggy.render.ExtractedTextCache` of the corpus
    being indexed, when there is one. OCR is the most expensive extraction raggy
    does per byte and its result is wanted twice — once to be embedded, and again
    to be shown beside the image — so it is cached while it is in hand and read
    back from there afterwards. Without a cache (a caller that passes none) this
    is exactly the OCR pass it always was.
    """
    text = _extracted_text(path, text_cache)
    return [Document(page_content=text, metadata={"source": str(path)})]


def _extracted_text(path: Path, text_cache=None) -> str:
    """The text of an image file: from ``text_cache`` if it has it, else OCR.

    The single place OCR is run and cached, so every caller — the indexer and
    whatever displays the text later — goes through the same cache.
    """
    if text_cache is not None:
        cached = text_cache.read(path)
        if cached is not None:
            return cached
    text = _ocr_image_bytes(path.read_bytes())
    if text_cache is not None:
        text_cache.store(path, text)
    return text


def _load_ocr_pdf(path: Path) -> list[Document]:
    """OCR an image-only PDF by rendering each page and running OCR per page."""
    import pymupdf

    documents: list[Document] = []
    with pymupdf.open(str(path)) as pdf:
        for page_number in range(pdf.page_count):
            page = pdf[page_number]
            pixmap = page.get_pixmap(dpi=DEFAULT_OCR_DPI)
            text = _ocr_image_bytes(pixmap.tobytes("png"))
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": str(path), "page": page_number + 1},
                )
            )
    return documents


def _collect_text_frame(frame, text_parts: list[str]) -> None:
    """Append the text of every non-empty paragraph in a text frame."""
    text = "\n".join(
        paragraph.text for paragraph in frame.paragraphs if paragraph.text.strip()
    )
    if text.strip():
        text_parts.append(text)


def _collect_shape(shape, text_parts: list[str]) -> None:
    """Recursively collect text from a shape, its text frames, and its tables."""
    if getattr(shape, "has_text_frame", False):
        _collect_text_frame(shape.text_frame, text_parts)
    elif getattr(shape, "has_table", False):
        for row in shape.table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                text_parts.append(" | ".join(cells))
    elif getattr(shape, "shape_type", None) == 6:  # MSO_SHAPE_TYPE.GROUP
        for child in shape.shapes:
            _collect_shape(child, text_parts)


def _load_pptx(path: Path) -> list[Document]:
    """Extract text from every slide of a PowerPoint deck, one Document per slide."""
    from pptx import Presentation

    presentation = Presentation(str(path))
    documents: list[Document] = []

    for index, slide in enumerate(presentation.slides, start=1):
        text_parts: list[str] = []

        for shape in slide.shapes:
            _collect_shape(shape, text_parts)

        if text_parts:
            documents.append(
                Document(
                    page_content="\n\n".join(text_parts),
                    metadata={"source": str(path), "page": index},
                )
            )

    return documents


def source_kind(path: Path) -> str:
    """Classify a source file for the viewer: how its content should render.

    ``"pdf"`` for native PDFs, ``"image"`` for OCR'd images, ``"text"`` for the
    plain-text formats that carry line numbers, ``"html"`` for web pages, and
    ``"office"`` for the DOCX/PPTX pair that the GUI converts to PDF for
    display. Derived from the extension alone so it can be stamped onto chunks
    at index time and read back at display time.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in LINE_ANNOTATED_EXTENSIONS:
        return "text"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix in {".docx", ".pptx"}:
        return "office"
    return "text"


def _finish_loaded(path: Path, documents: list[Document]) -> list[Document]:
    """Stamp the viewer hints every chunk of a loaded file carries.

    ``source`` is the path the user knows (also the manifest key), and
    ``source_kind`` tells the GUI which viewer to open. Both are written here,
    once per loaded file, so chunk metadata stays consistent across formats —
    including the conversion path, whose loader loads a cached PDF but must
    still attribute the chunks to the original document.
    """
    kind = source_kind(path)
    for doc in documents:
        doc.metadata["source"] = str(path)
        doc.metadata["source_kind"] = kind
    return documents


def _load_file(path: Path, text_cache=None) -> list[Document]:
    """Load documents from a single supported file based on its extension.

    Callers reach this only through :func:`source_files`, which is what decides
    a file is supported, so the extension is dispatched on here but never
    re-checked. Every returned document has its ``source``/``source_kind``
    metadata normalized by :func:`_finish_loaded`.

    ``text_cache`` is the optional :class:`raggy.render.ExtractedTextCache` for
    the corpus being indexed; it is passed to the image path, the one loader
    whose output a viewer needs again later (see :func:`_load_image`).
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        documents = loader.load()
        for doc in documents:
            if "page" in doc.metadata:
                doc.metadata["page"] += 1
        if not any((doc.page_content or "").strip() for doc in documents):
            logger.info("No extractable text in '%s'; falling back to OCR...", path)
            documents = _load_ocr_pdf(path)
        return _finish_loaded(path, documents)

    if suffix in IMAGE_EXTENSIONS:
        return _finish_loaded(path, _load_image(path, text_cache))

    if suffix == ".docx":
        return _finish_loaded(path, Docx2txtLoader(str(path)).load())

    if suffix == ".pptx":
        return _finish_loaded(path, _load_pptx(path))

    if suffix in {".html", ".htm"}:
        return _finish_loaded(
            path, BSHTMLLoader(str(path), open_encoding="utf-8").load()
        )

    return _finish_loaded(path, TextLoader(str(path), encoding="utf-8").load())


def annotate_line_numbers(splits: list[Document], content: str) -> None:
    """Annotate text splits with ``start_line``/``end_line`` line ranges.

    The positions are derived from where each chunk's text appears in the
    ``content`` string (1-indexed line numbers). Because ``chunk_overlap``
    causes adjacent chunks to share text, a search from ``cursor`` may not
    find a chunk against its true start; the first-occurrence fallback still
    yields approximate (but useful) line attribution.
    """
    cursor = 0
    chunk_size_tolerance = max(0, len(content))
    for split in splits:
        text = split.page_content
        start = content.find(text, cursor)
        if start == -1:
            start = content.find(text, 0, cursor + chunk_size_tolerance)
        if start == -1:
            start = content.find(text)

        if start == -1:
            continue
        split.metadata["start_line"] = content.count("\n", 0, start) + 1
        if text.endswith("\n"):
            newlines_in_text = text.count("\n") - 1
        else:
            newlines_in_text = text.count("\n")
        split.metadata["end_line"] = split.metadata["start_line"] + newlines_in_text
        cursor = start + len(text)


def should_annotate_lines(doc: Document) -> bool:
    """Return True only for text-based files that get line-number annotations.

    Line numbers are counted in the loaded ``page_content``, so they describe
    the file itself only for formats whose loader returns it verbatim — the
    plain-text ones, read by ``TextLoader``. HTML is excluded for exactly this
    reason: ``BSHTMLLoader`` yields the extracted text, which keeps the
    newlines inside text nodes but drops those inside tags and comments, so
    the count drifts further from the real line the deeper into the file a
    chunk sits.

    PDFs and PowerPoint decks carry a ``page`` key set by their loaders. DOCX
    has no native page boundaries (and Word's pagination can't be reproduced
    reliably), and image files have no meaningful lines, so both carry no
    location metadata and are skipped here.
    """
    suffix = Path(doc.metadata.get("source", "")).suffix.lower()
    return suffix in LINE_ANNOTATED_EXTENSIONS


def source_label(doc) -> str:
    """Build a human-readable source location for a retrieved document."""
    meta = doc.metadata
    parts = [Path(meta.get("source", "unknown")).name]

    if "page" in meta:
        parts.append(f"page {meta['page']}")
    if "start_line" in meta:
        parts.append(f"lines {meta['start_line']}-{meta['end_line']}")

    return ", ".join(parts)


def _walk_source(root: Path, skipped: list[Path]) -> Iterator[Path]:
    """Yield the supported files one source entry contributes, in load order.

    Paths are canonicalized (:meth:`Path.resolve`) because they become the
    corpus's identity in three places at once — the manifest's fingerprint keys,
    each chunk's ``source`` metadata, and the ``?path=`` a GUI sends back when it
    opens a cited document. Anything less than one canonical spelling on this
    machine (Windows in particular, where an 8.3 ``ADMINI~1`` path and its long
    form are the same file) makes those three disagree about a file that has not
    changed, which reads as churn.
    """
    if root.is_dir():
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                skipped.append(_canonical(file_path))
                continue
            yield _canonical(file_path)
    elif root.suffix.lower() in SUPPORTED_EXTENSIONS:
        yield _canonical(root)
    else:
        skipped.append(_canonical(root))


def _canonical(path: Path) -> Path:
    """The single spelling of ``path`` used everywhere a file is identified."""
    try:
        return path.resolve()
    except OSError:  # pragma: no cover - unresolvable path (broken link, perms)
        return path


def source_files(
    sources: Sequence[str | Path],
    skipped: list[Path] | None = None,
    on_missing: Literal["raise", "skip"] = "raise",
    report: LoadReport | None = None,
) -> list[Path]:
    """Return every supported file under ``sources``, in the order it is indexed.

    This is the corpus's single walk: the loaders below read the files it names
    and :func:`raggy.indexing.file_fingerprints` hashes them, so the paths
    recorded in the manifest are by construction the ones the loaders write
    into each chunk's ``source`` metadata.

    Each entry may be a single file or a directory, which is walked
    recursively. Unsupported files are recorded in ``skipped`` rather than
    raising, and a directory holding none simply contributes nothing. A file
    reachable through more than one entry is returned once, at its first
    position. ``on_missing`` decides what an entry that no longer exists means
    (see :func:`load_documents`); ``report`` additionally records the skipped
    and missing entries as data.
    """
    if on_missing not in ("raise", "skip"):
        raise ValueError(f"on_missing must be 'raise' or 'skip', got {on_missing!r}")

    paths: list[Path] = []
    missing: list[Path] = []
    for source in sources:
        path = Path(source)
        (paths if path.exists() else missing).append(path)
    if missing and on_missing == "raise":
        raise FileNotFoundError(
            "Source document(s) not found at: "
            + ", ".join(str(path) for path in missing)
        )
    for path in missing:
        logger.warning("Skipping '%s': file no longer exists.", path)
    _record_missing(missing, report)

    # Source entries are canonicalized too: they are recorded in the manifest and
    # compared on every run, so the short (``ADMINI~1``) and long spellings of one
    # directory must not look like a configuration change.
    paths = [_canonical(path) for path in paths]

    files: list[Path] = []
    seen: set[Path] = set()

    def claim(path: Path) -> bool:
        """True the first time a path is reached, under any of its spellings."""
        resolved = path.resolve()
        if resolved in seen:
            return False
        seen.add(resolved)
        return True

    for root in paths:
        unsupported: list[Path] = []
        for file_path in _walk_source(root, unsupported):
            if claim(file_path):
                files.append(file_path)
        if skipped is not None:
            skipped.extend(path for path in unsupported if claim(path))
    return files


def _load_each(
    paths: list[Path],
    progress: ProgressCallback | None,
    report: LoadReport | None = None,
    loader: Callable[[Path], list[Document]] = _load_file,
) -> list[Document]:
    """Load every path in turn, reporting each one and skipping what fails.

    A file that cannot be read is logged and passed over rather than aborting
    the run: one unreadable file should not cost an otherwise good corpus. With
    a ``report``, the failure is also recorded there (path + error message) so a
    front end can show which files were left out instead of only logging them.

    ``loader`` reads one path into documents; the conversion path passes its
    own so DOCX/PPTX are read from their cached PDF while keeping their
    original ``source`` path.
    """
    total = len(paths)
    documents: list[Document] = []
    for index, path in enumerate(paths, start=1):
        if progress is not None:
            progress(f"[{index}/{total}] ingesting {path.name} ...")
        if report is not None:
            report.paths.append(str(path))
        try:
            documents.extend(loader(path))
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load '%s': %s", path, e)
            if report is not None:
                report.failed.append(
                    FileFailure(path=str(path), error=str(e) or repr(e))
                )
    return documents


def _report_unsupported(skipped: list[Path], loaded_any: bool) -> None:
    """Log a single message summarizing ignored unsupported files.

    Raised to WARNING when nothing loaded at all: the sources then hold nothing
    readable — most likely a directory of file types raggy does not support —
    and the CLI shows WARNING but not INFO.
    """
    if not skipped:
        return
    logger.log(
        logging.INFO if loaded_any else logging.WARNING,
        "Detected %d unsupported file(s) that won't be used; ignoring them. "
        "Supported file types: %s.",
        len(skipped),
        ", ".join(sorted(SUPPORTED_EXTENSIONS)),
    )


def load_documents(
    sources: Sequence[str | Path],
    progress: ProgressCallback | None = None,
    on_missing: Literal["raise", "skip"] = "raise",
    report: LoadReport | None = None,
    loader: Callable[[Path], list[Document]] = _load_file,
) -> list[Document]:
    """Load documents from files and/or directories.

    Each entry may be a single supported file or a directory containing them;
    see :func:`source_files` for how the two are walked. Unsupported file types
    are ignored and reported once, and a file that fails to read is logged and
    skipped rather than aborting the run. ``progress`` receives one status line
    per file, counted across all entries.

    ``on_missing`` decides what an entry that no longer exists means. The
    default ``"raise"`` suits the configured sources, where a missing path is a
    config error worth stopping for. Incremental indexing passes ``"skip"``:
    its file list was fingerprinted earlier in the same run, so a file deleted
    since then is expected, and dropping it beats aborting the whole update —
    the next run reconciles it as a deletion.

    ``report`` (optional) collects the per-file outcome the log lines carry:
    unsupported/missing entries in ``report.skipped`` and load failures in
    ``report.failed``. ``loader`` overrides how one path becomes documents (the
    DOCX/PPTX conversion path reads the cached PDF instead of the source file).
    """
    skipped: list[Path] = []
    files = source_files(sources, skipped, on_missing=on_missing, report=report)
    _record_unsupported(skipped, report)
    documents = _load_each(files, progress, report, loader)
    _report_unsupported(skipped, bool(documents))
    if report is not None:
        report.documents = documents
    return documents
