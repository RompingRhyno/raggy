"""What the viewer needs to display a cited chunk's source document.

The GUI's main pane renders whatever a citation points at, and the format
decides both the viewer and how a jump works:

- **pdf** — native PDFs and converted DOCX/PPTX. The viewer opens the PDF and
  jumps to the chunk's page, then highlights the chunk's own text on that page.
  This is the case the whole conversion pipeline exists to produce.
- **text** — ``.txt``/``.md``/``.markdown``. Chunks carry line ranges, so a jump
  means scrolling to a line and highlighting the chunk text there.
- **image** — standalone OCR'd images. There is no text layer to search, so the
  viewer shows the image next to the raw OCR text of the chunk (plan feature 4,
  option (a)); no interactive jump.
- **html** — rendered markup, with the chunk text highlighted in place.

Everything here is derived from chunk metadata plus the render cache, so no
document is re-read or re-converted at display time.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from .loaders import IMAGE_EXTENSIONS, source_kind
from .render import RenderCache

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
HTML_EXTENSIONS = {".html", ".htm"}


@dataclass(frozen=True)
class DocumentSpec:
    """How to display one source document, and where to jump inside it."""

    source: str
    name: str
    kind: str
    content_url: str
    page: int | None = None
    start_line: int | None = None
    end_line: int | None = None
    search: str | None = None
    highlightable: bool = True
    preview_url: str | None = None
    note: str = ""
    ocr_text: str = ""
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        payload = {
            "source": self.source,
            "name": self.name,
            "kind": self.kind,
            "content_url": self.content_url,
            "page": self.page,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "search": self.search,
            "highlightable": self.highlightable,
            "preview_url": self.preview_url,
            "note": self.note,
            "ocr_text": self.ocr_text,
            "metadata": dict(self.metadata),
        }
        return payload


def build_spec(
    doc: Document,
    db_directory: str,
    content_url_for,
    corpus_id: str | None = None,
) -> DocumentSpec:
    """Describe how to show ``doc``'s source file, with its citation position.

    ``content_url_for(path, corpus_id)`` maps a filesystem path to the URL the
    browser should fetch it from; the server supplies it so this module stays
    free of HTTP details.
    """
    meta = doc.metadata or {}
    source = str(meta.get("source", ""))
    if not source:
        raise ValueError("chunk has no 'source' metadata")

    path = Path(source)
    kind = str(meta.get("source_kind") or source_kind(path))
    search = _search_text(doc)

    if kind == "image" or path.suffix.lower() in IMAGE_EXTENSIONS:
        return DocumentSpec(
            source=source,
            name=path.name,
            kind="image",
            content_url=content_url_for(source, corpus_id),
            search=None,
            highlightable=False,
            note=(
                "Image sources have no text layer to search, so this chunk's "
                "OCR text is shown beside the image instead of a highlight."
            ),
            ocr_text=doc.page_content,
            metadata=_public_metadata(meta),
        )

    # ``source_kind`` is the loader's classification ("office" for DOCX/PPTX, or
    # the extension when a converted file records its own kind); a chunk with no
    # kind at all is still a PDF if that is what its extension says.
    if kind in {"pdf", "office", "docx", "pptx"} or path.suffix.lower() == ".pdf":
        return DocumentSpec(
            source=source,
            name=path.name,
            kind="pdf",
            content_url=_pdf_url(
                meta, source, db_directory, content_url_for, corpus_id
            ),
            page=_int_or_none(meta.get("page")),
            search=search,
            note=_conversion_note(path, meta),
            metadata=_public_metadata(meta),
        )

    if path.suffix.lower() in HTML_EXTENSIONS:
        return DocumentSpec(
            source=source,
            name=path.name,
            kind="html",
            content_url=content_url_for(source, corpus_id),
            search=search,
            note="",
            metadata=_public_metadata(meta),
        )

    # Everything else that reached the index is a plain-text format whose line
    # numbers were annotated at index time.
    return DocumentSpec(
        source=source,
        name=path.name,
        kind="text",
        content_url=content_url_for(source, corpus_id),
        start_line=_int_or_none(meta.get("start_line")),
        end_line=_int_or_none(meta.get("end_line")),
        search=search,
        note="",
        metadata=_public_metadata(meta),
    )


def _pdf_url(meta, source, db_directory, content_url_for, corpus_id) -> str:
    """The URL of the PDF to display: the source itself, or its cached render.

    ``rendered`` is stamped on the chunks at index time by the conversion
    loader, so the common path needs no lookup. Files indexed before conversion
    existed (or with the converter absent) fall back to the cache by hash, and
    finally to the source itself — if the source already is a PDF that is
    exactly right, and if it is a DOCX the viewer will report that it cannot
    display it rather than showing something wrong.
    """
    rendered = meta.get("rendered")
    if rendered and Path(rendered).exists():
        return content_url_for(str(rendered), corpus_id, cache_file=True)

    try:
        cached = RenderCache(db_directory).cached_path(source)
    except OSError:
        cached = None
    if cached is not None:
        return content_url_for(str(cached), corpus_id, cache_file=True)

    return content_url_for(source, corpus_id)


def _conversion_note(path: Path, meta: dict) -> str:
    """A short explanation when a converted document is shown as a PDF."""
    if meta.get("converted"):
        return (
            f"{path.name} is displayed as a PDF rendered from the original "
            f"{path.suffix.lstrip('.').upper()} at index time."
        )
    return ""


# How much of a chunk to use as a search string. This is not a limit of PDF,
# pdf.js, the chunker or the API: it is a match-robustness budget. The viewer
# looks for the search as one contiguous run of words (spaces allowed to differ),
# so the longer the run, the more chances that one word inside it was hyphenated,
# split across a column break, or reordered by the layout engine — and a single
# miss loses the whole highlight. Roughly one printed line is both distinctive
# enough to identify the passage and short enough to survive extraction.
MAX_SEARCH_CHARS = 180

# Below this, a line is too generic to identify a passage ("Introduction"), so
# the first substantial line is preferred over a short first one.
MIN_SEARCH_CHARS = 24


def _search_text(doc: Document) -> str | None:
    """The chunk's own text, trimmed for use as a search string.

    A whole chunk is a poor search string (see :data:`MAX_SEARCH_CHARS`), so this
    takes the first substantial line of the chunk and clamps it to the budget —
    **on a word boundary**, because a string cut mid-word can never match: the
    viewer matches whole words, and the tail "incorporat" appears nowhere in any
    document.

    Falls back to the whole chunk (also clamped, also on a word boundary) when a
    chunk is one unbroken line, which is the common shape for HTML and for
    converted Office documents.
    """
    for line in (doc.page_content or "").splitlines():
        candidate = " ".join(line.split())
        if len(candidate) >= MIN_SEARCH_CHARS:
            return _clamp_to_words(candidate)
    return _clamp_to_words(" ".join((doc.page_content or "").split())) or None


def _clamp_to_words(text: str) -> str:
    """``text`` within :data:`MAX_SEARCH_CHARS`, cut at the last whole word.

    If a single word is longer than the whole budget, the hard cut stands: there
    is no word boundary to round to, and returning nothing would lose the
    highlight entirely.
    """
    if len(text) <= MAX_SEARCH_CHARS:
        return text
    head = text[:MAX_SEARCH_CHARS]
    boundary = head.rfind(" ")
    return head[:boundary] if boundary > 0 else head


def _int_or_none(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _public_metadata(meta: dict) -> dict:
    """Chunk metadata for display: the location keys, JSON-safe."""
    keys = ("page", "start_line", "end_line", "source_kind", "relevance_score")
    return {key: meta[key] for key in keys if key in meta}
