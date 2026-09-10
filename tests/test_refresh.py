"""An explicit index run: what it reports, and what it does to the DB.

These tests run against a real Chroma collection (with a fake embedding
function, so no Ollama) because the questions they answer are about stored
state, not about the shape of a call: which files' vectors survive a refresh,
what the manifest records about failures, and what the report says happened.
"""

from pathlib import Path

import pytest
import yaml
from langchain_core.embeddings import Embeddings

from raggy import indexing
from raggy.indexing import chunk_counts, read_manifest
from raggy.refresh import refresh_index


class FakeEmbeddings(Embeddings):
    """Deterministic 3-dimension embeddings: enough for Chroma, no model needed."""

    def embed_documents(self, texts):
        return [[float(len(t) % 7), float(sum(map(ord, t)) % 13), 1.0] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@pytest.fixture
def no_ollama(monkeypatch):
    monkeypatch.setattr(indexing, "get_embeddings", lambda model_name: FakeEmbeddings())


@pytest.fixture
def corpus(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    # Distinct contents on purpose: Chroma derives a chunk's id from its text,
    # so three files made of the same repeated phrase would deduplicate into one
    # stored source and tell us nothing about per-file bookkeeping.
    (docs / "keep.txt").write_text("KEEP the first file " * 20, encoding="utf-8")
    (docs / "edit.txt").write_text("EDIT the second file " * 20, encoding="utf-8")
    (docs / "gone.txt").write_text("GONE the third file " * 20, encoding="utf-8")
    return docs


def run(docs_dir, db_dir, **kwargs):
    defaults = {
        "db_directory": str(db_dir),
        "embedding_model": "fake",
        "sources": [str(docs_dir)],
        # Inside the shipped default overlap (100): a corpus config may not set
        # an overlap at or above its own chunk size.
        "chunk_size": 120,
        "chunk_overlap": 20,
        "embed_batch_size": 10,
        "convert": False,
        # Pin the conversion setup: these tests must not depend on whether
        # LibreOffice happens to be installed on the machine running them.
        "file_converters": {},
    }
    defaults.update(kwargs)
    return refresh_index(**defaults)


def rerun(docs_dir, db_dir, store, **kwargs):
    """Run a refresh while releasing the previous store first.

    Chroma keeps an open handle on its files, and on Windows that locks them
    against the rebuild's directory wipe. The GUI releases the handle the same
    way before every refresh (see ``CorpusRuntime.refresh``).
    """
    indexing.close_vectorstore(store)
    return run(docs_dir, db_dir, **kwargs)


def test_first_run_indexes_every_supported_file(no_ollama, corpus, tmp_path):
    store, report = run(corpus, tmp_path / "db")

    assert report.full_rebuild is True
    assert report.changed is True
    assert report.skipped == []
    assert {entry["path"] for entry in report.as_dict()["indexed"]} == {
        str(path) for path in sorted(corpus.iterdir())
    }
    assert report.indexed_count == 3
    # Chunk counts are read back from the DB, not assumed from the load step.
    assert all(count > 0 for count in report.chunks.values())
    assert sum(report.chunks.values()) == sum(chunk_counts(store).values())

    manifest = read_manifest(str(tmp_path / "db"))
    assert set(manifest["files"]) == set(report.chunks)
    assert manifest["attempted"] == manifest["files"]
    assert manifest["failed"] == {}


def test_second_run_is_a_no_op(no_ollama, corpus, tmp_path):
    db = tmp_path / "db"
    store, _ = run(corpus, db)

    _, report = rerun(corpus, db, store)

    assert report.full_rebuild is False
    assert report.changed is False
    assert report.indexed == []
    assert sorted(report.unchanged) == [str(path) for path in sorted(corpus.iterdir())]


def test_edited_file_is_reindexed_and_others_are_left_alone(
    no_ollama, corpus, tmp_path
):
    db = tmp_path / "db"
    store, first = run(corpus, db)
    before = read_manifest(str(db))["files"]

    (corpus / "edit.txt").write_text("CHANGED the second file " * 20, encoding="utf-8")
    _, report = rerun(corpus, db, store)

    assert report.changed is True
    assert report.indexed == [str(corpus / "edit.txt")]
    assert sorted(report.unchanged) == [
        str(corpus / "gone.txt"),
        str(corpus / "keep.txt"),
    ]
    # Untouched files are not re-embedded: their fingerprints are unchanged.
    after = read_manifest(str(db))["files"]
    assert after[str(corpus / "keep.txt")] == before[str(corpus / "keep.txt")]
    assert after[str(corpus / "edit.txt")] != before[str(corpus / "edit.txt")]
    assert (
        first.chunks[str(corpus / "keep.txt")]
        == report.chunks[str(corpus / "keep.txt")]
    )


def test_deleted_file_is_pruned_from_the_db_and_the_manifest(
    no_ollama, corpus, tmp_path
):
    """Plan verification task 1: deletion must remove vectors, not orphan them."""
    db = tmp_path / "db"
    store, _ = run(corpus, db)
    assert str(corpus / "gone.txt") in chunk_counts(store)

    (corpus / "gone.txt").unlink()
    store, report = rerun(corpus, db, store)

    assert report.removed == [str(corpus / "gone.txt")]
    assert str(corpus / "gone.txt") not in chunk_counts(store)
    indexing.close_vectorstore(store)
    manifest = read_manifest(str(db))
    assert str(corpus / "gone.txt") not in manifest["files"]
    assert str(corpus / "gone.txt") not in manifest["attempted"]
    # Nothing in the DB belongs to a file that is not on disk.
    assert report.orphans == []


def test_added_file_is_indexed_without_touching_the_rest(no_ollama, corpus, tmp_path):
    db = tmp_path / "db"
    store, _ = run(corpus, db)

    (corpus / "new.md").write_text("NEW the fourth file " * 20, encoding="utf-8")
    _, report = rerun(corpus, db, store)

    assert report.indexed == [str(corpus / "new.md")]
    assert len(report.unchanged) == 3
    assert report.removed == []


def test_corrupt_file_is_reported_as_failed_not_skipped(no_ollama, corpus, tmp_path):
    """Plan verification task 2: a corrupt-but-correctly-named file is a failure."""
    (corpus / "broken.pdf").write_bytes(b"%PDF-1.4\nthis is not a real pdf\n")

    _, report = run(corpus, tmp_path / "db")

    failed = {entry.path: entry.error for entry in report.failed}
    assert str(corpus / "broken.pdf") in failed
    assert failed[str(corpus / "broken.pdf")]
    assert report.skipped == []
    # The failure is recorded in the manifest so the next run can retry it
    # instead of treating it as a brand-new file it has never seen.
    manifest = read_manifest(str(tmp_path / "db"))
    assert str(corpus / "broken.pdf") in manifest["failed"]
    assert str(corpus / "broken.pdf") not in manifest["files"]
    assert str(corpus / "broken.pdf") in manifest["attempted"]


def test_repaired_file_is_picked_up_after_being_reported_as_failed(
    no_ollama, corpus, tmp_path
):
    db = tmp_path / "db"
    broken = corpus / "broken.pdf"
    broken.write_bytes(b"not a pdf at all")
    store, first = run(corpus, db)
    assert [entry.path for entry in first.failed] == [str(broken)]

    # A real PDF, written where the broken one was.
    import pymupdf

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 100), "REPAIRED CONTENT")
    document.save(str(broken))

    store, report = rerun(corpus, db, store)

    assert report.failed == []
    assert str(broken) in report.chunks
    assert str(broken) in chunk_counts(store)
    assert str(broken) in read_manifest(str(db))["files"]
    indexing.close_vectorstore(store)


def test_unchanged_failed_file_is_retried_rather_than_counted_as_new(
    no_ollama, corpus, tmp_path
):
    """A file that loads but yields no text stays a known failure, not a new file."""
    db = tmp_path / "db"
    blank = corpus / "blank.txt"
    blank.write_text("   \n\n  \n", encoding="utf-8")

    store, first = run(corpus, db)
    assert [entry.path for entry in first.failed] == [str(blank)]

    # Same bytes, so the corpus has not changed; an explicit refresh still
    # re-reads the file (retry_failed defaults to True) and reports the same
    # failure, rather than treating it as a brand-new file it has never seen.
    second_store, second = rerun(corpus, db, store)

    assert second.changed is False
    assert [entry.path for entry in second.failed] == [str(blank)]
    assert str(blank) not in second.indexed
    indexing.close_vectorstore(second_store)


def test_unsupported_extension_is_skipped_not_failed(no_ollama, corpus, tmp_path):
    (corpus / "notes.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    _, report = run(corpus, tmp_path / "db")

    skipped = report.skipped_unsupported
    assert [entry.path for entry in skipped] == [str(corpus / "notes.csv")]
    assert skipped[0].reason == "unsupported"
    assert skipped[0].detail == ".csv"
    assert report.failed == []
    # Unsupported files are not part of the corpus at all.
    assert (
        str(corpus / "notes.csv")
        not in read_manifest(str(tmp_path / "db"))["attempted"]
    )


def test_missing_source_is_an_error_unless_allowed(no_ollama, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("A " * 40, encoding="utf-8")
    gone = tmp_path / "gone"

    with pytest.raises(FileNotFoundError, match="not found"):
        run(gone, tmp_path / "db")

    # A source that disappeared since the last run is reported, not fatal, when
    # the caller opts in — which is what a Refresh does.
    _, report = run(gone, tmp_path / "db", allow_missing=True, retry_failed=False)

    assert report.indexed == []
    assert [entry.path for entry in report.skipped] == [str(gone)]
    assert report.skipped[0].reason == "missing"


def test_report_summary_and_dict_are_renderable(no_ollama, corpus, tmp_path):
    (corpus / "notes.csv").write_text("a,b\n", encoding="utf-8")
    (corpus / "broken.pdf").write_bytes(b"nope")

    _, report = run(corpus, tmp_path / "db")

    payload = report.as_dict()
    assert payload["counts"]["indexed"] == 3
    assert payload["counts"]["failed"] == 1
    assert payload["counts"]["skipped"] == 1
    assert "3 indexed" in report.summary
    assert "1 failed" in report.summary
    assert any("broken.pdf" in line for line in report.detail_lines())


def test_changing_chunking_forces_a_full_rebuild(no_ollama, corpus, tmp_path):
    db = tmp_path / "db"
    store, _ = run(corpus, db)

    _, report = rerun(corpus, db, store, chunk_size=64)

    assert report.full_rebuild is True
    assert report.indexed_count == 3
    assert report.unchanged == []


def test_conversion_setup_is_recorded_and_changing_it_rebuilds(
    no_ollama, corpus, tmp_path
):
    """A converter appearing/disappearing changes the text every chunk is made of."""
    db = tmp_path / "db"
    store, _ = run(corpus, db, file_converters={})
    assert read_manifest(str(db))["file_converters"] == {}

    _, report = rerun(
        corpus,
        db,
        store,
        file_converters={".docx": "libreoffice", ".pptx": "libreoffice"},
    )

    assert report.full_rebuild is True
    assert read_manifest(str(db))["file_converters"] == {
        ".docx": "libreoffice",
        ".pptx": "libreoffice",
    }


def test_manifest_written_before_conversion_existed_does_not_force_a_rebuild(
    no_ollama, corpus, tmp_path
):
    """Upgrading raggy must not rebuild every existing DB."""
    db = tmp_path / "db"
    store, _ = run(corpus, db)
    manifest = read_manifest(str(db))
    manifest.pop("file_converters")
    manifest.pop("attempted")
    manifest.pop("failed")
    (db / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")

    _, report = rerun(corpus, db, store)

    assert report.full_rebuild is False
    assert report.changed is False


def test_legacy_manifest_without_fingerprints_still_rebuilds(
    no_ollama, corpus, tmp_path
):
    db = tmp_path / "db"
    db.mkdir(parents=True, exist_ok=True)
    (db / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "sources": [str(corpus)],
                "chunk_size": 120,
                "chunk_overlap": 20,
                "embedding_model": "fake",
                "content_hash": "deadbeef",
            }
        ),
        encoding="utf-8",
    )

    _, report = run(corpus, db)

    assert report.full_rebuild is True


def test_chunk_metadata_carries_the_source_kind_for_the_viewer(
    no_ollama, corpus, tmp_path
):
    store, _ = run(corpus, tmp_path / "db")

    stored = indexing._collection_chunks(store)

    assert stored
    assert {chunk.metadata["source_kind"] for chunk in stored} == {"text"}
    assert all(chunk.metadata["source"] for chunk in stored)


def test_office_documents_are_indexed_from_a_rendered_pdf(
    no_ollama, tmp_path, monkeypatch
):
    """With a converter available, DOCX chunks describe the PDF the viewer opens.

    Rendering happens at index time (see :mod:`raggy.render`); this drives that
    path end to end with a scripted converter, which is the only way to exercise
    it on a machine without LibreOffice installed.
    """
    from raggy.render import RenderCache, make_conversion_loader
    from tests.test_render import FakeConverter

    docs = tmp_path / "docs"
    docs.mkdir()
    report_docx = docs / "report.docx"
    report_docx.write_bytes(b"pretend docx")
    (docs / "notes.txt").write_text("PLAIN TEXT " * 20, encoding="utf-8")
    db = tmp_path / "db"
    converter = FakeConverter("fake-office")
    monkeypatch.setattr("raggy.render.find_libreoffice", lambda: converter)
    cache = RenderCache(str(db))
    cache._converter = converter

    store, report = run(
        docs,
        db,
        convert=True,
        file_converters={".docx": "fake-office", ".pptx": "fake-office"},
        loader=make_conversion_loader(cache),
    )

    assert report.failed == []
    assert str(report_docx) in report.chunks
    # The conversion setup is recorded, so adding or removing a converter is a
    # rebuild-worthy change rather than a silent mix of the two text sources.
    manifest = read_manifest(str(db))
    assert manifest["file_converters"] == {
        ".docx": "fake-office",
        ".pptx": "fake-office",
    }

    docx_chunks = [
        chunk
        for chunk in indexing._collection_chunks(store)
        if chunk.metadata["source"] == str(report_docx)
    ]
    assert docx_chunks
    for chunk in docx_chunks:
        assert chunk.metadata["converted"] is True
        assert chunk.metadata["source_kind"] == "docx"
        # The cached PDF is what the viewer must open, and it is what the chunks
        # were read from — so its page numbers match what is displayed.
        rendered = Path(chunk.metadata["rendered"])
        assert rendered.exists()
        assert "RENDERED report" in chunk.page_content
