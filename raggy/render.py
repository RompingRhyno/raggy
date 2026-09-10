"""DOCX/PPTX → PDF pre-rendering, and the OCR text of images, cached per source
file content hash.

Two derived artefacts live here, for the same reason: both are expensive to
produce from a source file, both are needed again after indexing, and both
depend on nothing but that file's bytes.

- **Rendered Office documents.** Rendering DOCX/PPTX to PDF during indexing buys
  two things at once: one viewer for every paged format, and location metadata
  that means the same thing everywhere — a page number in a converted DOCX is a
  page number the PDF viewer can actually jump to, which Word's unreliable
  pagination cannot give us.
- **Extracted image text.** OCR runs during indexing so the text can be embedded.
  The viewer then wants to show that same text beside the image, and re-reading
  it out of the vector store does not work: an image's text is split into chunks
  like any other, and the splitter drops separator characters between windows, so
  stitching the chunks back does not reliably reproduce what was read. Keeping
  the text here is what stops the OCR from running a second time — which on an
  image-heavy corpus is seconds of work on every listing (see
  :class:`ExtractedTextCache`).

Both caches live under ``<db_directory>/render_cache/``, keyed by the source
file's SHA-256 — the same hash the manifest already computes to detect changes.
That is the whole invalidation story: a changed source has a different hash, so
it is derived again; an unchanged one is served from the cache. Sharing the
directory means :meth:`RenderCache.prune` clears both when a source leaves the
corpus. A full rebuild wipes the DB directory, and the caches with it.

LibreOffice headless is a single-process application: concurrent invocations
either queue behind a running instance or corrupt each other's output. Every
conversion therefore goes through one lock and one private ``-env:UserInstallation``
profile, and each has a timeout so a hung ``soffice`` cannot wedge indexing.
"""

import logging
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from .hashes import hash_file
from .loaders import _load_file

logger = logging.getLogger(__name__)

RENDER_CACHE_DIRNAME = "render_cache"

# The file a source's extracted text is cached in, inside its content-hash
# directory. Named for what it holds rather than for the one producer of it: any
# format whose text is expensive to extract and needed again later can use it.
EXTRACTED_TEXT_FILENAME = "extracted.txt"

# Formats that are rendered to PDF before being indexed. Kept explicit: this is
# the set whose *display* is a PDF, so it is also the set whose chunk metadata
# points the viewer at the cached file.
CONVERTIBLE_EXTENSIONS = (".docx", ".pptx")

DEFAULT_TIMEOUT_SECONDS = 180


class ConversionError(RuntimeError):
    """A source document could not be rendered to PDF."""


# Where packaged installs put the binary without adding it to PATH: the Windows
# installer and the macOS app bundle both do exactly that, and asking a user to
# edit PATH to use the GUI would be a poor trade for one string.
COMMON_INSTALL_PATHS: dict[str, tuple[str, ...]] = {
    "win32": (
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        r"C:\Program Files\LibreOffice 7\program\soffice.exe",
    ),
    "darwin": ("/Applications/LibreOffice.app/Contents/MacOS/soffice",),
}


def find_executable(command: str) -> str | None:
    """Locate ``command``, falling back to the well-known install locations."""
    found = shutil.which(command)
    if found:
        return found
    for candidate in COMMON_INSTALL_PATHS.get(sys.platform, ()):
        if Path(candidate).is_file():
            return candidate
    return None


@dataclass(frozen=True)
class DocumentConverter:
    """Something that can turn one document into a PDF.

    ``name`` goes into the manifest (as ``file_converters``) so that changing
    the converter — installing LibreOffice, or swapping it out — invalidates
    the chunks built from the old one instead of silently mixing the two.
    ``command`` is the resolved path where one was found, or the bare name to
    look up at conversion time.
    """

    name: str
    command: str
    extensions: tuple[str, ...]

    def convert(self, source: Path, dest_dir: Path, timeout: int) -> Path:
        """Render ``source`` into ``dest_dir`` and return the produced PDF."""
        executable = find_executable(self.command) or (
            self.command if Path(self.command).is_file() else None
        )
        if executable is None:
            raise ConversionError(f"{self.name} not found (looked for {self.command})")

        dest_dir.mkdir(parents=True, exist_ok=True)
        before = set(dest_dir.glob("*.pdf"))

        # A private profile per invocation: soffice refuses to run two instances
        # against one profile, and a stale lock in the shared profile is the
        # usual cause of a headless conversion hanging forever.
        with tempfile.TemporaryDirectory(prefix="raggy-soffice-") as profile:
            command = [
                executable,
                f"-env:UserInstallation={Path(profile).resolve().as_uri()}",
                "--headless",
                "--norestore",
                "--invisible",
                "--convert-to",
                "pdf",
                "--outdir",
                str(dest_dir),
                str(source),
            ]
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as e:
                raise ConversionError(f"conversion timed out after {timeout}s") from e
            except OSError as e:
                raise ConversionError(f"could not run {self.command}: {e}") from e

        produced = sorted(set(dest_dir.glob("*.pdf")) - before)
        if not produced:
            produced = [path for path in sorted(dest_dir.glob("*.pdf"))]
        if not produced:
            detail = (result.stderr or result.stdout or "").strip().splitlines()
            raise ConversionError(
                "no PDF was produced" + (f": {detail[-1]}" if detail else "")
            )
        return produced[0]


def find_libreoffice() -> DocumentConverter | None:
    """Return a LibreOffice converter if soffice is installed, else None.

    Looked up on PATH first, then in the locations the Windows installer and the
    macOS bundle use — neither of which puts ``soffice`` on PATH by default.
    """
    for candidate in ("soffice", "libreoffice"):
        executable = find_executable(candidate)
        if executable:
            return DocumentConverter(
                name="libreoffice",
                command=executable,
                extensions=CONVERTIBLE_EXTENSIONS,
            )
    return None


def converter_map() -> dict[str, str]:
    """The ``{extension: converter name}`` map recorded in the manifest.

    Empty when no converter is available, which is what tells the indexer to
    fall back to reading DOCX/PPTX natively (and what makes a later install of
    LibreOffice count as a change worth re-indexing).
    """
    converter = find_libreoffice()
    if converter is None:
        return {}
    return {extension: converter.name for extension in converter.extensions}


def _entry_dir(db_directory: str, source: Path | str) -> Path:
    """The content-hash directory holding every derived artefact for ``source``.

    One directory per source file's SHA-256, shared by the caches below, so the
    artefacts derived from a file stay together and a single ``prune`` clears
    them all.
    """
    return Path(db_directory) / RENDER_CACHE_DIRNAME / hash_file(source)


class RenderCache:
    """Content-addressed cache of source documents rendered to PDF.

    ``<db_directory>/render_cache/<sha256>/<stem>.pdf``. The hash comes from
    the source file, so two different files never collide and an edited file
    never reuses a stale render.
    """

    def __init__(self, db_directory: str, timeout: int = DEFAULT_TIMEOUT_SECONDS):
        self.root = Path(db_directory) / RENDER_CACHE_DIRNAME
        self.timeout = timeout
        self._lock = threading.Lock()
        self._converter = find_libreoffice()

    @property
    def available(self) -> bool:
        """True if a converter was found on PATH."""
        return self._converter is not None

    def supports(self, path: Path | str) -> bool:
        """True if this cache can render ``path`` (converter present, extension known)."""
        if self._converter is None:
            return False
        return Path(path).suffix.lower() in self._converter.extensions

    def _entry_dir(self, source: Path) -> Path:
        return _entry_dir(self.root.parent, source)

    def get(self, source: Path | str, force: bool = False) -> Path:
        """Return the cached PDF for ``source``, rendering it if needed.

        Raises :class:`ConversionError` if no converter is available or the
        conversion fails.
        """
        if self._converter is None:
            raise ConversionError("no document converter is installed")
        path = Path(source)
        entry = self._entry_dir(path)
        existing = sorted(entry.glob("*.pdf"))
        if existing and not force:
            return existing[0]

        # Serialized: LibreOffice headless cannot run conversions in parallel.
        with self._lock:
            existing = sorted(entry.glob("*.pdf"))
            if existing and not force:
                return existing[0]
            logger.info("Rendering '%s' to PDF for indexing...", path.name)
            with tempfile.TemporaryDirectory(prefix="raggy-render-") as staging:
                produced = self._converter.convert(path, Path(staging), self.timeout)
                entry.mkdir(parents=True, exist_ok=True)
                target = entry / f"{path.stem}.pdf"
                shutil.move(str(produced), str(target))
        return target

    def cached_path(self, source: Path | str) -> Path | None:
        """Return the cached PDF for ``source`` if it exists, without rendering."""
        path = Path(source)
        try:
            entry = self._entry_dir(path)
        except OSError:
            return None
        existing = sorted(entry.glob("*.pdf"))
        return existing[0] if existing else None

    def prune(self, keep_sources: list[str]) -> int:
        """Drop cache entries whose source files are no longer in the corpus.

        Only entries whose source path is known to be gone are removed, so a
        cache entry for a file that merely moved is discarded too — it can
        always be rebuilt, and keeping it would grow the cache forever. Both
        caches live in the same per-source directory, so this clears a file's
        rendered PDF and its extracted text together.
        """
        if not self.root.exists():
            return 0
        wanted = set()
        for source in keep_sources:
            path = Path(source)
            if path.exists():
                try:
                    wanted.add(hash_file(path))
                except OSError:
                    continue
        removed = 0
        for entry in self.root.iterdir():
            if entry.is_dir() and entry.name not in wanted:
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        return removed


class ExtractedTextCache:
    """Content-addressed cache of the text extracted from a source file.

    ``<db_directory>/render_cache/<sha256>/extracted.txt`` — the same hash-keyed
    directory as :class:`RenderCache`, which is what makes one ``prune`` clear
    both. Today only images use it (see :func:`raggy.loaders._load_image`), but
    nothing here is image-specific.

    This exists because the obvious alternative — reading the text back out of
    the vector store, where indexing already put it — is not sound. An image's
    OCR output is split into chunks like any other document, and at small chunk
    sizes the splitter drops separator characters at the window boundaries, so
    concatenating the chunks does not reliably reproduce the text that was read.
    A cache that is written once, while the text is in hand, has no such problem.
    """

    def __init__(self, db_directory: str):
        self.root = Path(db_directory) / RENDER_CACHE_DIRNAME

    def path(self, source: Path | str) -> Path | None:
        """Where ``source``'s text is (or would be) cached, or None if unhashable."""
        try:
            return _entry_dir(self.root.parent, source) / EXTRACTED_TEXT_FILENAME
        except OSError:
            return None

    def read(self, source: Path | str) -> str | None:
        """The cached text for ``source``, or None if it was never cached.

        An empty string is a real answer — "OCR found nothing" — and is cached
        as such, so a text-less image costs one OCR pass ever, not one per
        listing. Only a missing file returns None, which tells the caller to
        extract the text itself.
        """
        path = self.path(source)
        if path is None:
            return None
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # A cached file that cannot be read is a cache miss, never an error:
            # the caller can always produce the text again.
            return None

    def store(self, source: Path | str, text: str) -> None:
        """Cache ``text`` as the extracted text of ``source``.

        Best-effort: a cache that cannot be written costs a re-extraction later,
        which is exactly what happened before this cache existed, so it must not
        be allowed to fail an indexing run.
        """
        path = self.path(source)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError as e:
            logger.warning("Could not cache extracted text for '%s': %s", source, e)


def make_text_caching_loader(
    db_directory: str, loader: Callable | None = None
) -> Callable[[Path], list[Document]]:
    """Wrap a file loader so the text it extracts is cached as it is extracted.

    Every indexing run should go through this, whichever entry point started it:
    the cache is what stops the OCR behind an image from running a second time
    when a viewer asks for the text (see :class:`ExtractedTextCache`), and it can
    only be filled by whoever reads the file first.

    ``loader`` is the run's own loader — a conversion loader for DOCX/PPTX, or
    None for the native one. Whichever it is, the cache is bound to it here,
    where the corpus being indexed is known, rather than passed down through
    every call site that might read a file.
    """

    text_cache = ExtractedTextCache(db_directory)

    def load(path: Path) -> list[Document]:
        if loader is None:
            return _load_file(path, text_cache=text_cache)
        try:
            return loader(path, text_cache=text_cache)
        except TypeError:
            # A loader of the caller's own (or a test double): its signature does
            # not take a cache, and it reads files itself.
            return loader(path)

    return load


def make_conversion_loader(
    cache: RenderCache,
    fallback: Callable[[Path], list[Document]] = _load_file,
) -> Callable[[Path], list[Document]]:
    """Build a loader that reads convertible files from their cached PDF.

    The returned callable has :func:`raggy.loaders._load_file`'s signature, so
    it drops straight into ``load_documents``. Convertible extensions go through
    the cache (chunked from the rendered PDF, so citation pages match the
    viewer); everything else — and any file whose conversion fails — is read
    natively by ``fallback``, with the failure logged. Metadata is rewritten so
    the chunks still name the *source* document: the DB is keyed by source path,
    and the cached PDF is an implementation detail of display.

    Like :func:`raggy.loaders._load_file`, it takes an optional ``text_cache``
    and passes it to the fallback, so the one path that extracts something
    expensive (an image's OCR) fills the cache whichever loader reaches it.
    """

    def load(path: Path, text_cache=None) -> list[Document]:
        if not cache.supports(path):
            return _via_fallback(path, text_cache)

        try:
            rendered = cache.get(path)
        except ConversionError as e:
            logger.warning(
                "Could not convert '%s' (%s); indexing its native text instead.",
                path.name,
                e,
            )
            return _via_fallback(path, text_cache)

        documents = _load_file(rendered)
        for doc in documents:
            doc.metadata["source"] = str(path)
            doc.metadata["source_kind"] = path.suffix.lower().lstrip(".")
            doc.metadata["rendered"] = str(rendered)
            doc.metadata["converted"] = True
        return documents

    def _via_fallback(path: Path, text_cache) -> list[Document]:
        if text_cache is None:
            return fallback(path)
        return fallback(path, text_cache=text_cache)

    return load
