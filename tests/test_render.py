"""DOCX/PPTX → PDF rendering: content-addressed cache, serialized conversion.

LibreOffice is not required. The tests substitute a converter whose "conversion"
is a scripted file write, which is what makes the interesting behaviour —
cache keying, serialization, fallback — testable without a word processor.
"""

import subprocess
import threading
import time
from pathlib import Path

import pymupdf
import pytest

from raggy import render
from raggy.loaders import _load_file
from raggy.render import (
    ConversionError,
    DocumentConverter,
    RenderCache,
    converter_map,
    find_libreoffice,
    make_conversion_loader,
)


def write_pdf(path: Path, text: str = "RENDERED CONTENT") -> Path:
    """Write a one-page PDF whose text is ``text``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 100), text)
    document.save(str(path))
    return path


@pytest.fixture
def docx(tmp_path):
    """A file that only *looks* like a DOCX; the fake converter never reads it."""
    path = tmp_path / "report.docx"
    path.write_bytes(b"pretend this is a docx")
    return path


class FakeConverter(DocumentConverter):
    """A ``DocumentConverter`` whose conversion is a scripted PDF write.

    Subclassing rather than patching an instance: the real converter is frozen
    (its ``name`` goes into the manifest, so it should not be mutable), and this
    keeps the same public shape without shelling out to a word processor.
    """

    def __init__(
        self, name="fake-office", extensions=(".docx", ".pptx"), body="RENDERED"
    ):
        super().__init__(name=name, command="fake", extensions=extensions)
        self.body = body

    def convert(self, source: Path, dest_dir: Path, timeout: int) -> Path:
        return write_pdf(
            Path(dest_dir) / f"{source.stem}.pdf", f"{self.body} {source.stem}"
        )


class SlowConverter(DocumentConverter):
    """A converter that records how many conversions run at the same time."""

    def __init__(self, observe, delay=0.05):
        super().__init__(name="slow", command="fake", extensions=(".docx",))
        self.observe = observe
        self.delay = delay

    def convert(self, source: Path, dest_dir: Path, timeout: int) -> Path:
        self.observe()
        time.sleep(self.delay)
        return write_pdf(Path(dest_dir) / f"{source.stem}.pdf")


class BoomConverter(DocumentConverter):
    """A converter that always fails, to exercise the native-text fallback."""

    def __init__(self):
        super().__init__(name="boom", command="fake", extensions=(".docx",))

    def convert(self, source: Path, dest_dir: Path, timeout: int) -> Path:
        raise ConversionError("conversion exploded")


class TestFindConverter:
    def test_no_converter_when_soffice_is_absent(self, monkeypatch):
        monkeypatch.setattr(render, "find_executable", lambda command: None)

        assert render.find_executable("definitely-not-here") is None
        assert find_libreoffice() is None
        assert converter_map() == {}

    def test_found_converter_maps_both_office_extensions(self, monkeypatch):
        monkeypatch.setattr(
            render, "find_executable", lambda command: f"/usr/bin/{command}"
        )

        converter = find_libreoffice()

        assert converter is not None
        assert converter.name == "libreoffice"
        assert converter.command == "/usr/bin/soffice"
        assert converter_map() == {".docx": "libreoffice", ".pptx": "libreoffice"}

    def test_install_paths_are_used_when_soffice_is_not_on_path(self, monkeypatch):
        """The Windows installer and the macOS bundle do not touch PATH."""
        monkeypatch.setattr(render.shutil, "which", lambda command: None)
        monkeypatch.setattr(render.sys, "platform", "win32")
        monkeypatch.setattr(
            render.Path, "is_file", lambda self: str(self).endswith("soffice.exe")
        )

        found = render.find_executable("soffice")

        assert found is not None and found.endswith("soffice.exe")


class TestRenderCache:
    def test_converts_once_and_serves_the_cache_afterwards(self, docx, tmp_path):
        cache = RenderCache(str(tmp_path / "db"))
        cache._converter = FakeConverter()

        first = cache.get(docx)
        mtime = first.stat().st_mtime_ns
        second = cache.get(docx)

        assert first == second
        assert first.stat().st_mtime_ns == mtime
        assert "RENDERED report" in pymupdf.open(str(first))[0].get_text()

    def test_edited_source_gets_a_new_render(self, docx, tmp_path):
        """The cache key is the content hash, so this is the whole invalidation story."""
        cache = RenderCache(str(tmp_path / "db"))
        cache._converter = FakeConverter()

        first = cache.get(docx)
        docx.write_bytes(b"something else entirely")
        second = cache.get(docx)

        assert first != second
        assert first.exists()  # the old render is not deleted, just no longer used

    def test_cached_path_does_not_convert(self, docx, tmp_path):
        cache = RenderCache(str(tmp_path / "db"))
        cache._converter = FakeConverter()

        assert cache.cached_path(docx) is None
        assert cache.get(docx).exists()
        assert cache.cached_path(docx) == cache.get(docx)

    def test_unsupported_extension_is_not_converted(self, tmp_path):
        cache = RenderCache(str(tmp_path / "db"))
        cache._converter = FakeConverter()
        notes = tmp_path / "notes.txt"
        notes.write_text("plain text", encoding="utf-8")

        assert cache.supports(notes) is False
        assert cache.supports(tmp_path / "deck.pptx") is True

    def test_conversion_is_serialized(self, tmp_path):
        """LibreOffice headless must never be run twice at once."""
        cache = RenderCache(str(tmp_path / "db"))
        overlap = {"max": 0, "active": 0}
        guard = threading.Lock()

        def observe() -> None:
            with guard:
                overlap["active"] += 1
                overlap["max"] = max(overlap["max"], overlap["active"])
            time.sleep(0.02)
            with guard:
                overlap["active"] -= 1

        cache._converter = SlowConverter(observe, delay=0)

        sources = []
        for index in range(4):
            path = tmp_path / f"doc{index}.docx"
            path.write_bytes(f"document {index}".encode())
            sources.append(path)

        threads = [
            threading.Thread(target=cache.get, args=(source,)) for source in sources
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert overlap["max"] == 1

    def test_missing_converter_raises_a_clear_error(self, docx, tmp_path):
        cache = RenderCache(str(tmp_path / "db"))
        cache._converter = None

        with pytest.raises(ConversionError, match="no document converter"):
            cache.get(docx)

    def test_prune_drops_entries_for_files_that_are_gone(self, docx, tmp_path):
        cache = RenderCache(str(tmp_path / "db"))
        cache._converter = FakeConverter()
        cache.get(docx)

        removed = cache.prune([str(docx)])
        assert removed == 0

        docx.unlink()
        assert cache.prune([]) == 1


class TestDocumentConverter:
    def test_missing_executable_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(render, "find_executable", lambda command: None)
        converter = DocumentConverter(name="x", command="x", extensions=(".docx",))

        with pytest.raises(ConversionError, match="not found"):
            converter.convert(tmp_path / "a.docx", tmp_path, timeout=1)

    def test_no_output_is_reported_with_the_converter_message(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(render.shutil, "which", lambda command: "soffice")
        monkeypatch.setattr(
            render.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "no filter found"),
        )
        converter = DocumentConverter(
            name="x", command="soffice", extensions=(".docx",)
        )

        with pytest.raises(ConversionError, match="no PDF was produced"):
            converter.convert(tmp_path / "a.docx", tmp_path, timeout=1)

    def test_timeout_is_reported(self, tmp_path, monkeypatch):
        monkeypatch.setattr(render.shutil, "which", lambda command: "soffice")

        def timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="soffice", timeout=1)

        monkeypatch.setattr(render.subprocess, "run", timeout)
        converter = DocumentConverter(
            name="x", command="soffice", extensions=(".docx",)
        )

        with pytest.raises(ConversionError, match="timed out"):
            converter.convert(tmp_path / "a.docx", tmp_path, timeout=1)

    def test_private_profile_is_passed_to_soffice(self, tmp_path, monkeypatch):
        """A shared profile is what makes concurrent headless runs hang."""
        monkeypatch.setattr(render.shutil, "which", lambda command: "soffice")
        seen = {}

        def run(command, **kwargs):
            seen["command"] = command
            write_pdf(Path(command[command.index("--outdir") + 1]) / "a.pdf")
            return subprocess.CompletedProcess(command, 0, "", "")

        monkeypatch.setattr(render.subprocess, "run", run)
        converter = DocumentConverter(
            name="x", command="soffice", extensions=(".docx",)
        )

        converter.convert(tmp_path / "a.docx", tmp_path, timeout=5)

        profile = next(arg for arg in seen["command"] if arg.startswith("-env:"))
        assert profile.startswith("-env:UserInstallation=file:///")
        assert "--headless" in seen["command"]


class TestConversionLoader:
    def test_converted_document_is_attributed_to_the_source(self, docx, tmp_path):
        cache = RenderCache(str(tmp_path / "db"))
        cache._converter = FakeConverter()
        loader = make_conversion_loader(cache, fallback=lambda path: _load_file(path))

        documents = loader(docx)

        assert len(documents) == 1
        # The DB is keyed by the file the user knows, not by the cached render.
        assert documents[0].metadata["source"] == str(docx)
        assert documents[0].metadata["source_kind"] == "docx"
        assert documents[0].metadata["converted"] is True
        assert Path(documents[0].metadata["rendered"]).exists()
        assert "RENDERED report" in documents[0].page_content

    def test_failed_conversion_falls_back_to_native_text(self, tmp_path, monkeypatch):
        """A DOCX that cannot be rendered must still be indexed, not dropped."""
        path = tmp_path / "notes.docx"
        path.write_bytes(b"not really a docx")
        cache = RenderCache(str(tmp_path / "db"))
        cache._converter = FakeConverter()

        calls = {"fallback": 0}

        def fallback(source: Path):
            calls["fallback"] += 1
            from langchain_core.documents import Document

            return [
                Document(page_content="NATIVE TEXT", metadata={"source": str(source)})
            ]

        cache._converter = BoomConverter()
        loader = make_conversion_loader(cache, fallback=fallback)

        documents = loader(path)

        assert calls["fallback"] == 1
        assert documents[0].page_content == "NATIVE TEXT"
        assert documents[0].metadata["source"] == str(path)

    def test_non_convertible_files_go_straight_to_the_fallback(self, tmp_path):
        cache = RenderCache(str(tmp_path / "db"))
        cache._converter = FakeConverter()
        seen = []

        def fallback(source: Path):
            seen.append(source)
            from langchain_core.documents import Document

            return [Document(page_content="x", metadata={"source": str(source)})]

        loader = make_conversion_loader(cache, fallback=fallback)
        notes = tmp_path / "notes.txt"

        loader(notes)

        assert seen == [notes]

    def test_loader_output_is_indexable_by_the_normal_path(self, docx, tmp_path):
        """The conversion loader is a drop-in for _load_file."""
        cache = RenderCache(str(tmp_path / "db"))
        cache._converter = FakeConverter()
        loader = make_conversion_loader(cache)

        documents = loader(docx)

        assert [doc.metadata["page"] for doc in documents] == [1]
        assert all(doc.metadata["source"] == str(docx) for doc in documents)
