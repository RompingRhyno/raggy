"""What the viewer is told to display for a cited chunk.

The GUI's jump-to-citation depends entirely on this mapping: the ``kind`` picks
the viewer, ``page``/``start_line`` says where to jump, and ``search`` is the
chunk's own text used for the in-page highlight. A mistake here shows up as a
citation that opens the wrong file or highlights nothing.
"""

import pymupdf
import pytest
from langchain_core.documents import Document

from raggy.documents import build_spec


def url_for(path, corpus_id=None, cache_file=False):
    """Stand-in for the server's URL builder."""
    flag = "&cache=1" if cache_file else ""
    return f"/api/corpora/{corpus_id or 'c'}/file?path={path}{flag}"


def test_text_chunk_carries_its_line_range_and_a_search_string(tmp_path):
    source = tmp_path / "notes.md"
    body = "First line of the note, long enough to search for\nsecond line\n"
    source.write_text(body, encoding="utf-8")
    doc = Document(
        page_content="First line of the note, long enough to search for",
        metadata={
            "source": str(source),
            "source_kind": "text",
            "start_line": 1,
            "end_line": 1,
        },
    )

    spec = build_spec(doc, str(tmp_path / "db"), url_for, "c")

    assert spec.kind == "text"
    assert spec.name == "notes.md"
    assert spec.start_line == 1
    assert spec.end_line == 1
    assert spec.page is None
    assert spec.search.startswith("First line of the note")
    assert spec.highlightable is True
    assert spec.content_url.endswith("notes.md")


def test_search_string_is_trimmed_to_one_substantial_line(tmp_path):
    source = tmp_path / "big.txt"
    source.write_text("x", encoding="utf-8")
    doc = Document(
        page_content="short\nA much longer line that identifies the passage\nmore",
        metadata={"source": str(source), "source_kind": "text"},
    )

    spec = build_spec(doc, str(tmp_path / "db"), url_for)

    assert spec.search == "A much longer line that identifies the passage"


def test_search_string_falls_back_to_the_whole_chunk_when_lines_are_short(tmp_path):
    source = tmp_path / "tiny.txt"
    source.write_text("x", encoding="utf-8")
    doc = Document(
        page_content="one\ntwo\nthree",
        metadata={"source": str(source), "source_kind": "text"},
    )

    spec = build_spec(doc, str(tmp_path / "db"), url_for)

    assert spec.search == "one two three"


def test_pdf_chunk_points_at_the_page_and_the_original_file(tmp_path):
    source = tmp_path / "paper.pdf"
    doc = Document(
        page_content="A distinctive sentence from page four of the paper",
        metadata={"source": str(source), "source_kind": "pdf", "page": 4},
    )

    spec = build_spec(doc, str(tmp_path / "db"), url_for, "papers")

    assert spec.kind == "pdf"
    assert spec.page == 4
    assert spec.highlightable is True
    assert "paper.pdf" in spec.content_url
    assert "cache=1" not in spec.content_url


def test_converted_office_chunk_points_at_the_cached_render(tmp_path):
    """The viewer must show the PDF the page numbers were computed from."""
    source = tmp_path / "report.docx"
    source.write_bytes(b"docx bytes")
    rendered = tmp_path / "render_cache" / "abc" / "report.pdf"
    rendered.parent.mkdir(parents=True)
    rendered.write_bytes(b"%PDF-1.4")
    doc = Document(
        page_content="Text from the converted report",
        metadata={
            "source": str(source),
            "source_kind": "docx",
            "page": 2,
            "converted": True,
            "rendered": str(rendered),
        },
    )

    spec = build_spec(doc, str(tmp_path / "db"), url_for, "papers")

    assert spec.kind == "pdf"
    assert spec.page == 2
    assert "cache=1" in spec.content_url
    assert "report.pdf" in spec.content_url
    assert "report.docx" in spec.note


def test_converted_chunk_without_a_rendered_hint_falls_back_to_the_cache(tmp_path):
    """Chunks indexed before conversion existed still find their render."""
    source = tmp_path / "old.docx"
    source.write_bytes(b"docx bytes")
    db = tmp_path / "db"
    from raggy.render import RenderCache
    from tests.test_render import FakeConverter

    cache = RenderCache(str(db))
    cache._converter = FakeConverter()
    rendered = cache.get(source)
    doc = Document(
        page_content="Text from a converted report",
        metadata={"source": str(source), "source_kind": "docx", "page": 1},
    )

    spec = build_spec(doc, str(db), url_for)

    assert rendered.exists()
    assert spec.kind == "pdf"
    assert "cache=1" in spec.content_url


def test_image_chunk_disables_jumping_and_carries_the_ocr_text(tmp_path):
    source = tmp_path / "scan.png"
    source.write_bytes(b"png bytes")
    doc = Document(
        page_content="HELLO WORLD from OCR",
        metadata={"source": str(source), "source_kind": "image"},
    )

    spec = build_spec(doc, str(tmp_path / "db"), url_for)

    assert spec.kind == "image"
    assert spec.highlightable is False
    assert spec.search is None
    assert spec.ocr_text == "HELLO WORLD from OCR"
    assert "no text layer" in spec.note


def test_image_detected_from_the_extension_even_without_kind_metadata(tmp_path):
    source = tmp_path / "photo.jpeg"
    source.write_bytes(b"jpeg bytes")
    doc = Document(page_content="", metadata={"source": str(source)})

    spec = build_spec(doc, str(tmp_path / "db"), url_for)

    assert spec.kind == "image"


def test_html_chunk_is_flagged_for_the_markup_viewer(tmp_path):
    source = tmp_path / "page.html"
    source.write_text("<html><body>Hello</body></html>", encoding="utf-8")
    doc = Document(
        page_content="Hello from the page, in a sentence long enough to highlight",
        metadata={"source": str(source), "source_kind": "html"},
    )

    spec = build_spec(doc, str(tmp_path / "db"), url_for)

    assert spec.kind == "html"
    assert spec.highlightable is True
    assert spec.search


def test_missing_source_metadata_is_an_error():
    doc = Document(page_content="x", metadata={})

    with pytest.raises(ValueError, match="source"):
        build_spec(doc, "db", url_for)


def test_spec_is_json_ready(tmp_path):
    import json

    source = tmp_path / "a.pdf"
    source.write_bytes(b"%PDF-1.4")
    doc = Document(
        page_content="A line long enough to be used as the search string",
        metadata={
            "source": str(source),
            "source_kind": "pdf",
            "page": 3,
            "relevance_score": 0.5,
        },
    )

    payload = build_spec(doc, str(tmp_path / "db"), url_for).as_dict()

    json.dumps(payload)
    assert payload["page"] == 3
    assert payload["metadata"]["relevance_score"] == 0.5


def test_a_real_pdf_flows_from_loading_to_a_jump_target(tmp_path):
    """End to end for the common case: index a PDF, then ask where to jump."""
    from raggy.loaders import load_documents

    source = tmp_path / "real.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 100), "The quick brown fox jumps over the lazy dog")
    document.save(str(source))

    loaded = load_documents([str(source)])
    spec = build_spec(loaded[0], str(tmp_path / "db"), url_for)

    assert spec.kind == "pdf"
    assert spec.page == 1
    assert "quick brown fox" in spec.search
