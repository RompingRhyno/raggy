"""The GUI's front end, run for real against the server.

A page whose every control is dead is what this catches. Nothing else in the
suite can see that layer: an endpoint test proves the API answers, and the static
tests prove the files agree with each other, but neither runs `app.js` — a throw
on its first line (a `const` read before it is initialized, an element looked up
before it exists) leaves the page rendered and inert, with a clean server log and
every other test passing.

So the real module is fetched from a real server and driven over a fake DOM in
Node (see ``frontend_smoke.mjs``, which lists the checks it makes). Only the
embedding function and the chat model are faked: the retriever, the index and the
HTTP surface are the real ones.
"""

import json
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from raggy import indexing, pipeline
from raggy.gui.server import create_app
from tests.test_gui_api import EvenScores, FakeEmbeddings, StubLLM

SMOKE_SCRIPT = Path(__file__).resolve().parent / "js" / "frontend_smoke.mjs"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _wait_until_up(port: int, deadline: float) -> bool:
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


@pytest.fixture
def served_app(tmp_path, monkeypatch):
    """A real server on a loopback port, indexing is local and offline."""
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "user"))
    # The dialog opens on the home directory, so it has to exist: a home that
    # does not is why the dialog used to open with an empty name field, and the
    # fallback that covers it is tested separately.
    (tmp_path / "user").mkdir()
    monkeypatch.setattr(indexing, "get_embeddings", lambda model: FakeEmbeddings())
    monkeypatch.setattr(pipeline, "get_llm", lambda *a, **k: StubLLM())
    monkeypatch.setattr(pipeline, "get_cross_encoder", lambda model: EvenScores([]))
    monkeypatch.setattr(
        "raggy.gui.state.ensure_models", lambda cfg, progress=None: None
    )
    # The BM25 index is built from whatever the fake embeddings produced, and the
    # reranker never runs: no model is downloaded and nothing is reached over the
    # network.

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "alpha.txt").write_text("Alpha document content " * 30, encoding="utf-8")
    (docs / "beta.txt").write_text("Beta document content " * 30, encoding="utf-8")

    app = create_app(tmp_path / "gui")
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    assert _wait_until_up(port, time.time() + 20), "the test server did not start"
    try:
        yield f"http://127.0.0.1:{port}", docs
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
def browser(tmp_path):
    """A real browser, or a skip: the front-end checks need a layout engine."""
    from tests.browser import HeadlessBrowser, find_chromium

    executable = find_chromium()
    if executable is None:
        pytest.skip("no Chromium-based browser is installed")
    instance = HeadlessBrowser(executable, tmp_path / "browser-profile", (1200, 800))
    try:
        yield instance
    finally:
        instance.close()


def _app_js(browser, base_url, expression, settle=1.5):
    """Evaluate ``expression`` on the page.

    A JSON string is parsed for convenience; an expression that already returns
    an object (``returnByValue`` gives structured data back) is returned as is.
    """
    browser.open(base_url, settle=settle)
    value = browser.evaluate(expression)
    return json.loads(value) if isinstance(value, str) else value


def _open_add_dialog(browser, base_url):
    """Open the add-corpus dialog and wait for its first browse to land.

    The dialog is populated by a request, so a fixed sleep either wastes time or
    races it; this waits for the folder name to appear in the field.
    """
    browser.open(base_url, settle=1.0)
    return browser.evaluate(
        """(async () => {
             document.getElementById('add-corpus-btn').click();
             const name = document.getElementById('corpus-name');
             for (let i = 0; i < 50 && !name.value; i += 1) {
               await new Promise((r) => setTimeout(r, 100));
             }
             return {
               name: name.value,
               path: document.getElementById('browse-path').textContent,
               sources: document.getElementById('corpus-sources').textContent,
             };
           })()"""
    )


def test_a_new_corpus_is_named_after_the_folder_being_browsed(served_app, browser):
    """The default name follows the folder, and stops following it once typed.

    The name used to be set once, on the first browse, so navigating on to the
    folder you actually wanted left the previous folder's name in the field.
    """
    base_url, _ = served_app
    data = _open_add_dialog(browser, base_url)

    assert data["name"] == "user", data  # the home directory the dialog opens on
    assert "user" in data["sources"], data
    assert "user" in data["path"], data


def test_typing_a_name_stops_the_folder_from_overwriting_it(served_app, browser):
    """The suggestion is a suggestion: the user's own name wins.

    Typing then browsing must not replace what was typed — the failure this
    guards is a name that follows the folder even after the user named it.
    """
    base_url, _ = served_app
    data = _app_js(
        browser,
        base_url,
        """(async () => {
             document.getElementById('add-corpus-btn').click();
             const name = document.getElementById('corpus-name');
             for (let i = 0; i < 50 && !name.value; i += 1) {
               await new Promise((r) => setTimeout(r, 100));
             }
             const suggested = name.value;
             name.value = 'My Papers';
             name.dispatchEvent(new Event('input'));
             document.getElementById('browse-home').click();
             await new Promise((r) => setTimeout(r, 1200));
             return { suggested: suggested, afterTyping: name.value };
           })()""",
        settle=1.0,
    )

    assert data["suggested"], "the dialog should have suggested the folder name"
    assert data["afterTyping"] == "My Papers", data


@pytest.fixture
def served_app_without_a_home(tmp_path, monkeypatch):
    """A server whose home directory does not exist.

    The dialog opens by asking the server for the home directory, so this is the
    case where that one browse fails and the dialog has to open somewhere else
    rather than sit empty.
    """
    home = tmp_path / "missing-home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(indexing, "get_embeddings", lambda model: FakeEmbeddings())
    monkeypatch.setattr(
        "raggy.gui.state.ensure_models", lambda cfg, progress=None: None
    )

    app = create_app(tmp_path / "gui")
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    assert _wait_until_up(port, time.time() + 20), "the test server did not start"
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_the_dialog_opens_somewhere_usable_when_the_home_directory_is_gone(
    served_app_without_a_home, browser
):
    """A dialog with an empty name field and no folders looks broken.

    The home directory is not a path the user chose, so failing to open it is not
    their problem to solve: the dialog opens at the filesystem root instead, with
    a suggestion in the name field and folders to click.
    """
    data = _open_add_dialog(browser, served_app_without_a_home)

    assert data["path"], "the dialog should have opened at some real directory"
    assert data["name"], f"the name field should have a suggestion: {data}"
    assert "index" in data["sources"], data


def test_the_page_works_when_it_is_actually_run(served_app):
    """Fetch the app from the server, evaluate it, and click its controls."""
    base_url, docs = served_app
    node = shutil.which("node")
    if node is None:  # pragma: no cover - a machine without Node
        pytest.skip("node is not installed, so the front end cannot be driven")

    # Build a corpus the page can show. Done over HTTP, the same way the page
    # would: nothing here talks to the app object directly.
    import httpx

    client = httpx.Client(base_url=base_url, timeout=120)
    corpus = client.post(
        "/api/corpora", json={"name": "Smoke", "sources": [str(docs)]}
    ).json()["corpus"]
    report = client.post(f"/api/corpora/{corpus['id']}/refresh").json()["report"]
    assert report["counts"]["indexed"] == 2, report

    result = subprocess.run(
        [node, str(SMOKE_SCRIPT), base_url],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,  # the return code is asserted below, with the output printed
    )
    print(result.stdout)
    if result.stderr.strip():
        print(result.stderr)

    assert result.returncode == 0, "the front end smoke test reported failures"
    assert "all checks passed" in result.stdout
    assert "app.js evaluates" in result.stdout
