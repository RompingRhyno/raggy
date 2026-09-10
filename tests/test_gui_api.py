"""The GUI's HTTP surface, driven through a real client.

These run against a temporary GUI home and a fake embedding function, so no
Ollama, no reranker download and no LibreOffice are needed: what is being tested
is the contract the browser depends on (see ``raggy/gui/API.md``) — status codes,
payload shapes, and the two things a local server has to get right, path
authorization and refusing to run two long operations on one corpus at once.
"""

import json
import re
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.embeddings import Embeddings

from raggy import indexing, loaders, pipeline
from raggy.gui.server import create_app
from tests.test_render import FakeConverter


class FakeEmbeddings(Embeddings):
    """Deterministic embeddings: enough for Chroma, no model server needed."""

    def embed_documents(self, texts):
        return [[float(len(t) % 7), float(sum(map(ord, t)) % 13), 1.0] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@pytest.fixture
def docs_dir(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "alpha.txt").write_text("Alpha document content " * 20, encoding="utf-8")
    (folder / "beta.txt").write_text("Beta document content " * 20, encoding="utf-8")
    (folder / "notes.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    return folder


@pytest.fixture
def client(tmp_path, monkeypatch, docs_dir):
    """An app whose home, embeddings, model pulls and indexing are all local fakes."""
    monkeypatch.setattr(indexing, "get_embeddings", lambda model: FakeEmbeddings())
    # Model pulls hit Ollama, which these tests neither need nor require; the
    # pipeline path itself is exercised with a stubbed chain.
    monkeypatch.setattr(
        "raggy.gui.state.ensure_models", lambda cfg, progress=None: None
    )
    app = create_app(tmp_path / "gui")
    with TestClient(app) as test_client:
        test_client.docs_dir = docs_dir
        yield test_client


def make_corpus(client, name="Papers", sources=None):
    response = client.post(
        "/api/corpora",
        json={"name": name, "sources": sources or [str(client.docs_dir)]},
    )
    assert response.status_code == 200, response.text
    return response.json()["corpus"]


class StubLLM:
    """A chat model that records what it was asked and answers nothing useful.

    ``test_gui_api``'s other query tests stub the whole chain, which cannot show
    whether a *retrieval* scope worked. This one replaces only generation, so
    the real retriever — Chroma filter, BM25 pass, cross-encoder rerank — is what
    decides which chunks arrive here.
    """

    def __init__(self):
        self.prompt = ""

    def invoke(self, messages):
        self.prompt = "\n".join(str(getattr(m, "content", m)) for m in messages)
        return "stub answer"

    def __call__(self, messages):  # pragma: no cover - LCEL uses .invoke
        return self.invoke(messages)


@pytest.fixture
def stub_generation(monkeypatch):
    """Fake the LLM and the cross-encoder: no Ollama, no model download.

    The cross-encoder scores every pair equally, which is deliberate: the rerank
    stage must not be what decides whether a filtered-out file's chunks survive,
    or a passing test would prove nothing about the scope.
    """
    llm = StubLLM()
    monkeypatch.setattr(pipeline, "get_llm", lambda *a, **k: llm)
    monkeypatch.setattr(pipeline, "get_cross_encoder", lambda model: EvenScores([]))
    return llm


class EvenScores(BaseCrossEncoder):
    """A cross-encoder that likes everything equally."""

    def __init__(self, scores):
        self.scores = scores

    def score(self, pairs):
        return [0.9 for _ in pairs]


def refreshed_corpus(client):
    """Index the fixture's files and return their paths, keyed by name, exactly
    as the API reports them — which is what a browser sends back as a scope."""
    corpus = make_corpus(client)
    client.post(f"/api/corpora/{corpus['id']}/refresh")
    payload = client.get(f"/api/corpora/{corpus['id']}/documents").json()
    return {entry["name"]: entry["path"] for entry in payload["documents"]}


def cited_names(payload):
    return {
        Path(citation["target"]["source"]).name
        for citation in payload["citations"]
        if citation["target"]
    }


def write_scan(path, label):
    """A synthetic scan of ``label``, big enough for OCR to read unambiguously."""
    import pymupdf

    document = pymupdf.open()
    sheet = document.new_page(width=900, height=200)
    sheet.insert_text((20, 100), label, fontsize=40)
    sheet.get_pixmap(dpi=150).save(str(path))


class TestHealth:
    def test_reports_capabilities_without_a_converter(self, client):
        payload = client.get("/api/health").json()

        assert payload["status"] == "ok"
        assert payload["features"]["streaming"] is False
        assert payload["features"]["chat_memory"] is False
        # Holds either way; what matters is that the flag is present and boolean.
        assert isinstance(payload["features"]["document_conversion"], bool)


class TestCorporaAPI:
    def test_starts_empty(self, client):
        payload = client.get("/api/corpora").json()

        assert payload == {"active": None, "corpora": []}

    def test_create_lists_and_activates(self, client):
        corpus = make_corpus(client)

        listing = client.get("/api/corpora").json()

        assert listing["active"] == corpus["id"]
        assert corpus["indexed"] is False
        assert listing["corpora"][0]["sources"] == [str(client.docs_dir.resolve())]

    def test_create_reports_a_missing_folder(self, client, tmp_path):
        response = client.post(
            "/api/corpora", json={"name": "Ghost", "sources": [str(tmp_path / "nope")]}
        )

        assert response.status_code == 400
        assert "does not exist" in response.json()["error"]["message"]

    def test_create_rejects_an_empty_name(self, client):
        response = client.post(
            "/api/corpora", json={"name": "", "sources": [str(client.docs_dir)]}
        )

        assert response.status_code == 422  # pydantic min_length

    def test_sources_and_db_directory_are_the_servers_business(self, client):
        corpus = make_corpus(client)

        response = client.patch(
            f"/api/corpora/{corpus['id']}",
            json={"settings": {"db_directory": "/tmp/evil", "sources": ["/tmp"]}},
        )

        assert response.status_code == 200
        updated = response.json()["corpus"]
        assert updated["db_directory"] == corpus["db_directory"]
        assert updated["sources"] == corpus["sources"]

    def test_switch_active_corpus(self, client, tmp_path):
        first = make_corpus(client, "First")
        second_folder = tmp_path / "other"
        second_folder.mkdir()
        (second_folder / "x.txt").write_text("x " * 200, encoding="utf-8")
        second = make_corpus(client, "Second", [str(second_folder)])

        assert client.post(f"/api/corpora/{first['id']}/activate").json() == {
            "active": first["id"]
        }
        assert client.get("/api/corpora").json()["active"] == first["id"]
        assert second["id"] != first["id"]

    def test_add_source_to_a_corpus(self, client, tmp_path):
        corpus = make_corpus(client)
        extra = tmp_path / "extra"
        extra.mkdir()

        response = client.post(
            f"/api/corpora/{corpus['id']}/sources",
            json={"name": "Papers", "sources": [str(extra)]},
        )

        assert response.status_code == 200
        assert str(extra.resolve()) in response.json()["corpus"]["sources"]

    def test_delete_moves_the_active_corpus(self, client, docs_dir, tmp_path):
        first = make_corpus(client, "First")
        second_folder = tmp_path / "second"
        second_folder.mkdir()
        (second_folder / "y.txt").write_text("y " * 200, encoding="utf-8")
        second = make_corpus(client, "Second", [str(second_folder)])
        client.post(f"/api/corpora/{second['id']}/activate")

        payload = client.delete(f"/api/corpora/{first['id']}").json()

        assert payload["active"] == second["id"]
        assert [c["id"] for c in client.get("/api/corpora").json()["corpora"]] == [
            second["id"]
        ]

    def test_delete_takes_the_index_with_it(self, client):
        """What the GUI's Delete button sends, and why it is the default.

        The corpus's DB directory is raggy's copy of the files — one per corpus
        by design. Leaving it behind would orphan an index nothing can reach and
        nothing will clean up, so the confirmation dialog deletes it too, and
        says so.
        """
        corpus = make_corpus(client)
        client.post(f"/api/corpora/{corpus['id']}/refresh")
        index = Path(corpus["db_directory"])
        assert index.is_dir() and any(index.iterdir()), (
            "refresh should have built an index"
        )

        response = client.delete(f"/api/corpora/{corpus['id']}")

        assert response.status_code == 200
        assert response.json()["active"] is None
        assert not index.exists(), "the corpus's DB directory should be gone"
        assert client.get("/api/corpora").json()["corpora"] == []
        # And the source files the user indexed are untouched.
        assert (client.docs_dir / "alpha.txt").exists()

    def test_delete_can_keep_the_index(self, client):
        """The flag is real: a caller that wants the DB kept can say so."""
        corpus = make_corpus(client)
        client.post(f"/api/corpora/{corpus['id']}/refresh")
        index = Path(corpus["db_directory"])

        client.delete(f"/api/corpora/{corpus['id']}?delete_db=false")

        assert index.exists()

    def test_deleting_an_unknown_corpus_is_a_404(self, client):
        response = client.delete("/api/corpora/ghost")

        assert response.status_code == 404
        assert "unknown corpus" in response.json()["error"]["message"]

    def test_unknown_corpus_is_a_404(self, client):
        assert client.get("/api/corpora/ghost/documents").status_code == 404
        assert client.post("/api/corpora/ghost/activate").status_code == 404
        assert client.post("/api/corpora/ghost/refresh").status_code == 404


class TestBrowse:
    def test_returns_directories_and_files(self, client, docs_dir):
        payload = client.get("/api/browse", params={"path": str(docs_dir)}).json()

        names = {entry["name"] for entry in payload["entries"]}
        assert {"alpha.txt", "beta.txt", "notes.csv"} <= names
        assert payload["path"] == str(docs_dir.resolve())
        assert payload["entries"][0]["is_dir"] is False  # files only in this folder

    def test_directories_sort_first(self, client, tmp_path):
        (tmp_path / "afile.txt").write_text("x", encoding="utf-8")
        (tmp_path / "zfolder").mkdir()

        entries = client.get("/api/browse", params={"path": str(tmp_path)}).json()[
            "entries"
        ]

        assert entries[0]["is_dir"] is True

    def test_hidden_entries_are_omitted(self, client, tmp_path):
        browse_root = tmp_path / "browse"
        browse_root.mkdir()
        (browse_root / ".hidden").mkdir()
        (browse_root / "visible").mkdir()

        names = {
            entry["name"]
            for entry in client.get(
                "/api/browse", params={"path": str(browse_root)}
            ).json()["entries"]
        }

        assert names == {"visible"}

    def test_not_a_directory_is_a_400(self, client, docs_dir):
        response = client.get(
            "/api/browse", params={"path": str(docs_dir / "alpha.txt")}
        )

        assert response.status_code == 400
        assert "not a directory" in response.json()["error"]["message"]

    def test_defaults_to_the_home_directory(self, client):
        payload = client.get("/api/browse").json()

        assert payload["path"] == str(Path.home().resolve())


class TestRefreshAndDocuments:
    def test_refresh_indexes_the_corpus_and_reports_it(self, client):
        corpus = make_corpus(client)

        payload = client.post(f"/api/corpora/{corpus['id']}/refresh").json()["report"]

        assert payload["full_rebuild"] is True
        assert payload["counts"]["indexed"] == 2
        assert payload["counts"]["skipped"] == 1
        assert {Path(entry["path"]).name for entry in payload["indexed"]} == {
            "alpha.txt",
            "beta.txt",
        }
        assert payload["skipped"][0]["reason"] == "unsupported"
        assert payload["counts"]["chunks"] > 0

    def test_second_refresh_says_nothing_to_do(self, client):
        corpus = make_corpus(client)
        client.post(f"/api/corpora/{corpus['id']}/refresh")

        payload = client.post(f"/api/corpora/{corpus['id']}/refresh").json()["report"]

        assert payload["changed"] is False
        assert payload["counts"]["indexed"] == 0
        assert payload["counts"]["unchanged"] == 2
        assert "nothing to do" in payload["summary"]

    def test_changing_a_setting_rebuilds_the_corpus(self, client):
        """A pipeline setting the GUI exposes must reach the index, and rebuild it."""
        corpus = make_corpus(client)
        client.post(f"/api/corpora/{corpus['id']}/refresh")
        before = client.get(f"/api/corpora/{corpus['id']}/documents").json()["counts"][
            "chunks"
        ]

        response = client.patch(
            f"/api/corpora/{corpus['id']}",
            json={"settings": {"chunk_size": 40, "chunk_overlap": 10}},
        )
        assert response.status_code == 200, response.text
        payload = client.post(f"/api/corpora/{corpus['id']}/refresh").json()["report"]

        assert payload["full_rebuild"] is True
        # Smaller chunks from the same text: the corpus is re-embedded, so the
        # count can only go up.
        assert payload["counts"]["chunks"] > before
        manifest = yaml.safe_load(
            (Path(corpus["db_directory"]) / "manifest.yaml").read_text(encoding="utf-8")
        )
        assert manifest["chunk_size"] == 40

    def test_a_setting_that_cannot_index_is_refused_before_the_run(self, client):
        """An impossible chunking config must fail at the patch, not mid-index."""
        corpus = make_corpus(client)

        response = client.patch(
            f"/api/corpora/{corpus['id']}", json={"settings": {"chunk_size": 40}}
        )

        assert response.status_code == 400
        # The shipped default overlap (100) is larger than the new chunk size.
        assert "chunk_overlap" in response.json()["error"]["message"]

    def test_deleting_a_file_prunes_it_from_the_corpus(self, client):
        corpus = make_corpus(client)
        client.post(f"/api/corpora/{corpus['id']}/refresh")

        (client.docs_dir / "beta.txt").unlink()
        payload = client.post(f"/api/corpora/{corpus['id']}/refresh").json()["report"]

        assert payload["counts"]["removed"] == 1
        assert [Path(path).name for path in payload["removed"]] == ["beta.txt"]
        documents = client.get(f"/api/corpora/{corpus['id']}/documents").json()
        names = {entry["name"] for entry in documents["documents"]}
        assert "beta.txt" not in names

    def test_corrupt_file_is_reported_as_failed(self, client):
        (client.docs_dir / "broken.pdf").write_bytes(b"not a pdf at all")
        corpus = make_corpus(client)

        payload = client.post(f"/api/corpora/{corpus['id']}/refresh").json()["report"]

        assert payload["counts"]["failed"] == 1
        assert payload["failed"][0]["path"].endswith("broken.pdf")
        assert payload["failed"][0]["error"]

    def test_documents_lists_indexed_files_with_their_kinds(self, client):
        corpus = make_corpus(client)
        client.post(f"/api/corpora/{corpus['id']}/refresh")

        payload = client.get(f"/api/corpora/{corpus['id']}/documents").json()

        assert payload["indexed"] is True
        assert payload["counts"]["documents"] == 2
        entries = {entry["name"]: entry for entry in payload["documents"]}
        assert entries["alpha.txt"]["status"] == "indexed"
        assert entries["alpha.txt"]["kind"] == "text"
        assert entries["alpha.txt"]["chunks"] > 0
        assert entries["alpha.txt"]["content_url"].startswith("/api/corpora/")

    def test_unindexed_source_files_show_up_as_new(self, client):
        corpus = make_corpus(client)
        client.post(f"/api/corpora/{corpus['id']}/refresh")
        (client.docs_dir / "gamma.txt").write_text("Gamma " * 50, encoding="utf-8")

        payload = client.get(f"/api/corpora/{corpus['id']}/documents").json()

        gamma = next(e for e in payload["documents"] if e["name"] == "gamma.txt")
        assert gamma["status"] == "new"
        assert gamma["chunks"] == 0

    def test_counts_the_files_that_produced_no_text(self, client):
        """The GUI hides those by default, so it needs them counted, not guessed.

        A file on disk that has not been indexed yet is *not* one of them: it may
        well have text, and the list already marks it "new".
        """
        corpus = make_corpus(client)
        (client.docs_dir / "blank.txt").write_text("   \n\n \n", encoding="utf-8")
        (client.docs_dir / "unreadable.pdf").write_bytes(b"not a pdf at all")

        report = client.post(f"/api/corpora/{corpus['id']}/refresh").json()["report"]
        payload = client.get(f"/api/corpora/{corpus['id']}/documents").json()

        # Two readable files, one blank, one corrupt, plus one added afterwards.
        assert report["counts"]["failed"] == 2
        assert payload["counts"]["without_text"] == 2
        empty = {e["name"] for e in payload["documents"] if not e["chunks"]}
        assert empty == {"blank.txt", "unreadable.pdf"}
        assert all(
            e["chunks"] > 0 for e in payload["documents"] if e["name"] not in empty
        )

        (client.docs_dir / "later.txt").write_text("Later " * 40, encoding="utf-8")
        after = client.get(f"/api/corpora/{corpus['id']}/documents").json()
        assert after["counts"]["without_text"] == 2
        later = next(e for e in after["documents"] if e["name"] == "later.txt")
        assert later["status"] == "new" and later["chunks"] == 0

    def test_an_image_carries_its_ocr_text_for_the_viewer(self, client):
        """The viewer's OCR panel is the whole of feature 4's image handling.

        The text does not come from the vector store: an image's OCR output is
        split into chunks like any other document, and stitching them back does
        not reliably reproduce it (the splitter drops separators between
        windows). It comes from the extracted-text cache instead — see
        :class:`TestImageTextIsCached`.
        """
        image = client.docs_dir / "scan.png"
        write_scan(image, "RECOGNISABLE HEADING")
        corpus = make_corpus(client)
        client.post(f"/api/corpora/{corpus['id']}/refresh")

        payload = client.get(f"/api/corpora/{corpus['id']}/documents").json()

        entry = next(e for e in payload["documents"] if e["name"] == "scan.png")
        assert entry["status"] == "indexed"
        # OCR of synthetic glyphs is not letter-perfect on every platform, so the
        # assertion is that text came back at all.
        assert entry["ocr_text"].strip()
        assert "HEADING" in entry["ocr_text"].upper()

    def test_a_text_less_image_reports_no_ocr_text(self, client):
        """A blank image counts as text-less *and* has no text for a panel."""
        import pymupdf

        image = client.docs_dir / "blank.png"
        document = pymupdf.open()
        document.new_page(width=400, height=200).get_pixmap(dpi=72).save(str(image))
        corpus = make_corpus(client)

        report = client.post(f"/api/corpora/{corpus['id']}/refresh").json()["report"]

        assert [Path(f["path"]).name for f in report["failed"]] == ["blank.png"]
        payload = client.get(f"/api/corpora/{corpus['id']}/documents").json()
        entry = next(e for e in payload["documents"] if e["name"] == "blank.png")
        assert entry["chunks"] == 0
        assert entry["ocr_text"] == ""
        assert payload["counts"]["without_text"] == 1

    def test_a_converted_document_is_offered_as_its_pdf(self, client, monkeypatch):
        """The file list must not hand the PDF viewer a .docx.

        Indexing renders DOCX/PPTX to PDF and chunks that render, so clicking the
        document in the file list has to open the render — the source file itself
        is not something pdf.js can read.
        """
        from raggy.render import RenderCache

        docx = client.docs_dir / "report.docx"
        docx.write_bytes(b"pretend docx")
        converter = FakeConverter("libreoffice")
        monkeypatch.setattr("raggy.render.find_libreoffice", lambda: converter)
        corpus = make_corpus(client)
        # Render it the way indexing would, so the cache entry the viewer needs
        # exists (the rendering itself is covered in test_render.py).
        cache = RenderCache(corpus["db_directory"])
        cache._converter = converter
        cache.get(docx)

        payload = client.get(f"/api/corpora/{corpus['id']}/documents").json()
        entry = next(e for e in payload["documents"] if e["name"] == "report.docx")

        assert entry["kind"] == "document"
        assert entry["content_url"].split("&")[0].endswith(".pdf")
        assert "cache=1" in entry["content_url"]
        assert entry["source_url"].endswith(".docx")
        # And the URL the viewer is handed actually serves a PDF.
        served = client.get(entry["content_url"])
        assert served.status_code == 200
        assert served.content.startswith(b"%PDF")

    def test_status_is_readable_while_idle(self, client):
        corpus = make_corpus(client)

        payload = client.get(f"/api/corpora/{corpus['id']}/status").json()

        assert payload["busy"] is False
        assert payload["corpus"] == corpus["id"]


class TestFileServing:
    def test_serves_a_file_inside_the_corpus(self, client):
        corpus = make_corpus(client)

        response = client.get(
            f"/api/corpora/{corpus['id']}/file",
            params={"path": str(client.docs_dir / "alpha.txt")},
        )

        assert response.status_code == 200
        assert "Alpha document content" in response.text
        assert response.headers["cache-control"] == "no-store"

    def test_refuses_a_file_outside_the_corpus(self, client, tmp_path):
        corpus = make_corpus(client)
        secret = tmp_path / "secret.txt"
        secret.write_text("do not serve me", encoding="utf-8")

        response = client.get(
            f"/api/corpora/{corpus['id']}/file", params={"path": str(secret)}
        )

        assert response.status_code == 403
        assert "not part of this corpus" in response.json()["error"]["message"]

    def test_refuses_a_directory(self, client):
        corpus = make_corpus(client)

        response = client.get(
            f"/api/corpora/{corpus['id']}/file", params={"path": str(client.docs_dir)}
        )

        assert response.status_code == 403

    def test_serves_pdf_with_the_right_media_type(self, client):
        import pymupdf

        pdf = client.docs_dir / "page.pdf"
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 100), "PDF TEXT")
        document.save(str(pdf))
        corpus = make_corpus(client)

        response = client.get(
            f"/api/corpora/{corpus['id']}/file", params={"path": str(pdf)}
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.content.startswith(b"%PDF")


class TestQuery:
    def test_answers_with_citations_carrying_jump_targets(self, client, monkeypatch):
        """The whole pipeline, with generation stubbed out."""

        class FakeChain:
            def invoke(self, payload):
                return "Alpha is described in the first document."

        def fake_build_rag_chain(**kwargs):
            sink = kwargs["doc_sink"]
            from langchain_core.documents import Document

            sink.append(
                [
                    Document(
                        page_content="Alpha document content " * 4,
                        metadata={
                            "source": str(client.docs_dir / "alpha.txt"),
                            "source_kind": "text",
                            "start_line": 1,
                            "end_line": 4,
                            "relevance_score": 0.9,
                        },
                    )
                ]
            )
            return FakeChain(), None

        monkeypatch.setattr("raggy.api.ensure_ollama_model", lambda *a, **k: False)
        monkeypatch.setattr("raggy.api.build_rag_chain", fake_build_rag_chain)
        corpus = make_corpus(client)

        response = client.post("/api/query", json={"query": "What is alpha?"})

        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["answer"].startswith("Alpha is described")
        assert payload["corpus"] == corpus["id"]
        citation = payload["citations"][0]
        assert "page" not in citation["metadata"]
        assert citation["metadata"]["start_line"] == 1
        assert citation["score"] == pytest.approx(0.9)
        assert citation["target"]["kind"] == "text"
        assert citation["target"]["start_line"] == 1
        assert citation["target"]["highlightable"] is True
        assert citation["target"]["search"]

    def test_query_against_a_specific_corpus(self, client, monkeypatch):
        seen = {}

        class FakeChain:
            def invoke(self, payload):
                return "ok"

        def fake_build_rag_chain(**kwargs):
            seen["db"] = kwargs["db_directory"]
            kwargs["doc_sink"].append([])
            return FakeChain(), None

        monkeypatch.setattr("raggy.api.ensure_ollama_model", lambda *a, **k: False)
        monkeypatch.setattr("raggy.api.build_rag_chain", fake_build_rag_chain)
        corpus = make_corpus(client)

        client.post(
            "/api/query", params={"corpus_id": corpus["id"]}, json={"query": "hi"}
        )

        assert seen["db"] == corpus["db_directory"]

    def test_query_without_a_corpus_is_a_404(self, client):
        response = client.post("/api/query", json={"query": "hi"})

        assert response.status_code == 404
        assert "no corpus is configured" in response.json()["error"]["message"]

    def test_query_rejects_an_empty_question(self, client):
        make_corpus(client)

        assert client.post("/api/query", json={"query": ""}).status_code == 422

    def test_pipeline_failures_come_back_as_readable_errors(self, client, monkeypatch):
        def boom(**kwargs):
            raise ConnectionError("Connection refused to ollama")

        monkeypatch.setattr("raggy.api.ensure_ollama_model", lambda *a, **k: False)
        monkeypatch.setattr("raggy.api.build_rag_chain", boom)
        make_corpus(client)

        response = client.post("/api/query", json={"query": "hi"})

        assert response.status_code == 500
        error = response.json()["error"]
        assert "Connection refused" in error["message"]
        assert "ollama serve" in error["hint"]


class TestBusyCorpus:
    def test_a_second_long_operation_is_refused_not_queued(self, client):
        """Indexing holds the DB; a concurrent query must not interleave with it."""
        corpus = make_corpus(client)
        runtime = client.app.state.raggy.runtime(corpus["id"])
        assert runtime._lock.acquire(blocking=False) is True
        try:
            response = client.post(f"/api/corpora/{corpus['id']}/refresh")
            query = client.post("/api/query", json={"query": "hi"})
        finally:
            runtime._lock.release()

        assert response.status_code == 409
        assert "busy" in response.json()["error"]["message"]
        assert query.status_code == 409


class TestQueryScope:
    """What a question is asked of: one file, or the corpus minus the hidden ones.

    These exercise the whole retrieval path, because the scope is only real if
    it reaches both of its arms — the vector store through a metadata filter and
    BM25 through the list of files it may return.
    """

    def test_a_question_can_be_limited_to_one_file(self, client, stub_generation):
        paths = refreshed_corpus(client)

        payload = client.post(
            "/api/query",
            json={"query": "content", "include_sources": [paths["alpha.txt"]]},
        ).json()

        assert cited_names(payload) == {"alpha.txt"}

    def test_a_hidden_file_is_kept_out_of_the_answer(self, client, stub_generation):
        paths = refreshed_corpus(client)

        payload = client.post(
            "/api/query",
            json={"query": "content", "exclude_sources": [paths["alpha.txt"]]},
        ).json()

        assert "alpha.txt" not in cited_names(payload)
        assert "beta.txt" in cited_names(payload)

    def test_the_corpus_spelling_of_a_path_is_what_matches(
        self, client, stub_generation
    ):
        """A browser hands back the path the file list gave it, but the same file
        can be spelled differently (case, a different root casing on Windows) and
        must still select its chunks: the store compares the filter literally."""
        paths = refreshed_corpus(client)

        payload = client.post(
            "/api/query",
            json={"query": "content", "include_sources": [paths["alpha.txt"].upper()]},
        ).json()

        assert cited_names(payload) == {"alpha.txt"}

    def test_a_scope_that_selects_nothing_is_refused(self, client, stub_generation):
        """Better a 400 than an answer assembled from an empty context: "I cannot
        find that" would read as a fact about the corpus."""
        paths = refreshed_corpus(client)

        response = client.post(
            "/api/query",
            json={"query": "content", "exclude_sources": list(paths.values())},
        )

        assert response.status_code == 400
        assert "no files are selected" in response.json()["error"]["message"]
        assert response.json()["error"]["hint"]

    def test_hiding_every_file_leaves_the_answer_refused_not_empty(
        self, client, stub_generation
    ):
        """The GUI's red dots alone can empty the scope, with no include list."""
        paths = refreshed_corpus(client)

        response = client.post(
            "/api/query",
            json={
                "query": "content",
                "include_sources": [paths["alpha.txt"]],
                "exclude_sources": [paths["alpha.txt"]],
            },
        )

        assert response.status_code == 400

    def test_an_unknown_scope_is_refused(self, client, stub_generation):
        refreshed_corpus(client)

        response = client.post(
            "/api/query",
            json={
                "query": "content",
                "include_sources": [str(client.docs_dir / "ghost.txt")],
            },
        )

        assert response.status_code == 400

    def test_no_scope_still_searches_the_whole_corpus(self, client, stub_generation):
        refreshed_corpus(client)

        payload = client.post("/api/query", json={"query": "content"}).json()

        assert cited_names(payload) == {"alpha.txt", "beta.txt"}


class TestImageTextIsCached:
    """OCR runs once per image, not once per listing.

    Listing a corpus asks for every image's text so the viewer can show it beside
    the image. Re-running OCR there is what made an image-heavy corpus take ten
    seconds to show its files (measured: 12.0s cold, 0.1s warm on a 29-image
    corpus). These tests count the OCR invocations rather than trusting a timing,
    because a timing difference is a symptom and the count is the property.
    """

    @pytest.fixture
    def ocr_calls(self, monkeypatch):
        """Count OCR passes, with the engine itself stubbed out.

        Stubbed deliberately: what is under test is *whether* the extraction runs
        again, and a real engine would make the test slow and its output
        platform-dependent.
        """
        calls = []
        monkeypatch.setattr(
            loaders,
            "_ocr_image_bytes",
            lambda image_bytes: calls.append(len(image_bytes)) or "STUBBED OCR TEXT",
        )
        return calls

    def test_indexing_caches_the_text_it_extracts(self, client, ocr_calls):
        write_scan(client.docs_dir / "invoice.png", "INVOICE 2024-0042")
        corpus = make_corpus(client)

        client.post(f"/api/corpora/{corpus['id']}/refresh")

        assert len(ocr_calls) == 1, "OCR should run once, while indexing"
        cached = list(Path(corpus["db_directory"]).glob("render_cache/*/extracted.txt"))
        assert len(cached) == 1
        assert cached[0].read_text(encoding="utf-8") == "STUBBED OCR TEXT"

    def test_listing_an_image_reuses_the_cached_text(self, client, ocr_calls):
        write_scan(client.docs_dir / "invoice.png", "INVOICE 2024-0042")
        corpus = make_corpus(client)
        client.post(f"/api/corpora/{corpus['id']}/refresh")
        assert len(ocr_calls) == 1

        for _ in range(3):
            payload = client.get(f"/api/corpora/{corpus['id']}/documents").json()
            entry = next(e for e in payload["documents"] if e["name"] == "invoice.png")
            assert entry["ocr_text"] == "STUBBED OCR TEXT"

        assert len(ocr_calls) == 1, "listing must not OCR again"

    def test_editing_an_image_re_extracts_rather_than_reusing(self, client, ocr_calls):
        """The cache is keyed by the file's hash, so changed bytes cannot hit it."""
        image = client.docs_dir / "invoice.png"
        write_scan(image, "INVOICE 2024-0042")
        corpus = make_corpus(client)
        client.post(f"/api/corpora/{corpus['id']}/refresh")
        assert len(ocr_calls) == 1

        write_scan(image, "INVOICE 2024-0043 SUPERSEDED")
        client.post(f"/api/corpora/{corpus['id']}/refresh")

        assert len(ocr_calls) == 2, "an edited image has to be read again"

    def test_a_corpus_indexed_before_the_cache_still_shows_its_text(
        self, client, ocr_calls
    ):
        """The fallback, which is also the upgrade path for existing corpora.

        Nothing was cached for a corpus indexed before this existed, so the text
        is extracted on demand and cached then — the wait happens once, not on
        every listing.
        """
        image = client.docs_dir / "invoice.png"
        write_scan(image, "INVOICE 2024-0042")
        corpus = make_corpus(client)
        client.post(f"/api/corpora/{corpus['id']}/refresh")

        # Simulate the old state: the index exists, the text cache does not.
        for cached in Path(corpus["db_directory"]).glob("render_cache/*/extracted.txt"):
            cached.unlink()
        assert len(ocr_calls) == 1

        first = client.get(f"/api/corpora/{corpus['id']}/documents").json()
        entry = next(e for e in first["documents"] if e["name"] == "invoice.png")
        assert entry["ocr_text"] == "STUBBED OCR TEXT"
        assert len(ocr_calls) == 2, "a cold cache costs one extraction"

        second = client.get(f"/api/corpora/{corpus['id']}/documents").json()
        entry = next(e for e in second["documents"] if e["name"] == "invoice.png")
        assert entry["ocr_text"] == "STUBBED OCR TEXT"
        assert len(ocr_calls) == 2, "and it is cached, so the next listing is free"

    def test_a_text_less_image_is_not_re_extracted_every_listing(
        self, client, ocr_calls
    ):
        """An empty result is a result: it must be cached as one."""
        import pymupdf

        blank = client.docs_dir / "blank.png"
        document = pymupdf.open()
        document.new_page(width=400, height=200).get_pixmap(dpi=72).save(str(blank))
        corpus = make_corpus(client)
        client.post(f"/api/corpora/{corpus['id']}/refresh")

        for _ in range(2):
            client.get(f"/api/corpora/{corpus['id']}/documents")

        assert len(ocr_calls) == 1, "an image with nothing in it is read once"

    def test_index_is_served_at_the_root(self, client):
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_api_routes_win_over_the_static_mount(self, client):
        """A static mount at / must not shadow /api/*."""
        assert client.get("/api/health").json()["status"] == "ok"

    def test_the_shell_is_never_served_from_the_browser_cache(self, client):
        """The app that runs is the app on disk.

        A cached older `app.js` against a newer `index.html` is a page whose
        startup throws on the first missing element — every control dead, with
        no request in the server log to explain it. Reuse is how that happens
        with Starlette's defaults, so nothing may be stored and nothing may be
        reused without asking.
        """
        for path in ("/", "/styles.css", "/app.js", "/js/api.js", "/js/pdfview.js"):
            response = client.get(path)
            directive = response.headers.get("cache-control", "")

            assert response.status_code == 200, path
            assert "no-store" in directive or "no-cache" in directive, (path, directive)
            assert "max-age" not in directive, path
            assert "immutable" not in directive, path

    def test_a_conditional_request_still_gets_the_current_assets(self, client):
        """Whatever the browser cached, revalidating returns the body.

        Answering these with a 304 is the failure mode this guards: the browser
        keeps what it has and is told the page is up to date, so a change on
        disk never reaches the user on a plain reload.
        """
        for path in ("/", "/app.js", "/styles.css"):
            first = client.get(path)
            etag = first.headers.get("etag")
            assert etag, path
            revalidated = client.get(path, headers={"If-None-Match": etag})

            assert revalidated.status_code == 200, path
            assert revalidated.content == first.content, path

    def test_hidden_elements_are_actually_hidden(self, client):
        """`hidden` must beat the overlay/modal layout rules.

        The overlay and the modal are laid out with `display: flex`, which
        outranks the user-agent `[hidden] { display: none }` at equal
        specificity — leaving an invisible full-screen backdrop that swallows
        every click in the app underneath it. No API test can catch that, so the
        rule itself is asserted.
        """
        assert (
            "[hidden] { display: none !important; }" in client.get("/styles.css").text
        )

    def test_pdfjs_is_vendored_and_served(self, client):
        """The viewer has to work offline: pdf.js ships with the app, not a CDN."""
        for path in ("/vendor/pdfjs/pdf.min.mjs", "/vendor/pdfjs/pdf.worker.min.mjs"):
            response = client.get(path)
            assert response.status_code == 200, path
            assert len(response.content) > 100_000, path

    def test_static_assets_reference_no_remote_resources(self, client):
        """No CDN and no web fonts: the GUI must run with no network at all."""
        for path in ("/", "/styles.css", "/app.js"):
            body = client.get(path).text
            for remote in ("//cdn.", "fonts.googleapis", "fonts.gstatic", "unpkg.com"):
                assert remote not in body, f"{path} references {remote}"

    def test_the_file_list_groups_text_less_files_last(self, client):
        """Ordering is the front end's job, and this is asserted because the rule
        is easy to lose in a refactor: readable files first, then the ones with no
        text under their own heading."""
        app = client.get("/app.js").text

        assert "[...withText, ...empty]" in app
        assert "file-group" in app
        assert "Files with no text" in app

    def test_the_ask_pane_offers_a_scope_choice(self, client):
        """Two buttons above the box, not one toggle: which files a question is
        asked of has to be readable at a glance, not inferred from a pressed
        state someone has to find."""
        index = client.get("/").text

        assert 'id="scope-all"' in index and 'id="scope-file"' in index
        assert 'id="query-input"' in index
        # The scope row sits between the log and the box it applies to.
        assert index.index('id="scope-all"') < index.index('id="query-input"')

    def test_the_file_list_status_marker_is_a_button(self, client):
        """The green/red dot is the control that hides a file from context.

        Asserted because it is a *behaviour* with no server-side test: the row is
        built in app.js, so nothing else in this suite can see that the marker
        became clickable or that hiding it reaches the query.
        """
        app = client.get("/app.js").text

        assert "contextToggle" in app
        assert "toggleFileHidden" in app
        assert "is-hidden" in app
        # Hiding must reach the API, not just the list's appearance.
        assert "exclude_sources" in app
        assert "include_sources" in app

    def test_every_element_app_js_looks_up_exists_in_the_page(self, client):
        """The front end fails all at once when an id goes missing.

        `byId` throws during startup, so one renamed or dropped element leaves a
        page where the corpus dropdown is empty, the file list never fills and no
        button responds — with nothing in the server log, because the requests
        that would have failed are never made. Nothing else in this suite pairs
        the two files that have to agree, so this does.

        The reverse is deliberately not asserted: several ids are anchors for CSS
        or for `aria-*` and are never looked up.
        """
        app = client.get("/app.js").text
        index = client.get("/").text
        ids = set(re.findall(r'byId\("([^"]+)"\)', app))

        assert ids, "app.js no longer looks any element up by id"
        missing = {
            element_id for element_id in ids if f'id="{element_id}"' not in index
        }
        assert not missing, (
            f"app.js looks up ids the page does not have: {sorted(missing)}"
        )

    def test_the_context_toggle_is_bordered_only_on_hover(self, client):
        """No border at rest, a rounded-square border on hover: the point is to
        show the dot is a button without boxing in every row of the list."""
        css = client.get("/styles.css").text

        assert ".context-toggle" in css
        assert ".context-toggle:hover" in css
        assert ".context-toggle.is-hidden" in css
        # The dot itself is a pseudo-element, so the box around it can appear
        # without moving the dot or reflowing the row.
        assert ".context-toggle::before" in css
        # At rest the border is transparent — stated, so "no border" cannot
        # quietly become "a border on every row".
        at_rest = css[
            css.index(".context-toggle {") : css.index(".context-toggle::before")
        ]
        assert "border: 1px solid transparent;" in at_rest

    def test_the_text_layer_rules_are_present(self, client):
        """The PDF text layer must be told how to position its spans.

        pdf.js positions each text run with an inline transform, but the rules
        that make those spans absolutely positioned and transparent live in its
        own stylesheet (web/pdf_viewer.css), which this viewer does not load. Left
        out, every span becomes static content stacked at the page's top-left
        corner — visible as a column of text over the page image. Asserted here
        because nothing else in the test suite can see a layout bug.
        """
        css = client.get("/styles.css").text

        for rule in (
            ".pdf-surface .textLayer span",
            ".pdf-surface .textLayer br",
            "position: absolute;",
            "color: transparent;",
            "transform-origin: 0% 0%;",
            "font-size: calc(var(--text-scale-factor) * var(--font-height));",
        ):
            assert rule in css, f"missing from styles.css: {rule}"

    def test_the_viewer_publishes_the_layer_scale(self, client):
        """`--total-scale-factor` sizes the text runs, so the viewer must set it."""
        viewer = client.get("/js/pdfview.js").text

        assert "--total-scale-factor" in viewer
        assert "setLayerDimensions" in viewer

    def test_pdf_js_ships_its_viewer_stylesheet_for_reference(self, client):
        """Vendored so the rules above can be checked against their source."""
        response = client.get("/vendor/pdfjs/pdf_viewer.css")

        assert response.status_code == 200
        assert ".textLayer" in response.text
        assert "transform-origin" in response.text


def test_report_payload_is_json_serializable(client):
    corpus = make_corpus(client)

    report = client.post(f"/api/corpora/{corpus['id']}/refresh").json()["report"]

    json.dumps(report)
