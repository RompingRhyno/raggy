"""HTTP API and static host for the raggy GUI.

Design constraints worth stating, because they shape everything else here:

- **Local, single user.** The server binds to 127.0.0.1 and has no accounts or
  sessions. It is a window onto one user's own corpora.
- **No streaming, no chat memory.** Both are explicit non-goals of the build
  plan, so ``/api/query`` is a single request that returns a whole answer.
- **One long operation at a time.** Indexing holds an embedding model and a
  SQLite handle; a question holds the same handle for retrieval. Both take the
  per-corpus lock in :class:`raggy.gui.state.AppState`, so a refresh and a query
  queue instead of interleaving.

Every failure is returned as a JSON error object (``{"error": {"message": ...}}``)
with a message written for the person who clicked the button, not for a log
file — the GUI's job is to show it verbatim.
"""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..corpora import CorpusError, CorpusStore
from ..render import find_libreoffice
from .state import AppState, CorpusRuntime

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web"

MAX_QUERY_CHARS = 2000

# Letting the browser keep a copy of the app is a bad trade for a local
# single-user server: the files on disk *are* the running version, and a cached
# older app.js against a newer index.html is a page that throws on load and then
# looks entirely dead. "" is the suffix Starlette reports for the directory
# index, i.e. for "/".
STATIC_CACHE_CONTROL = {
    "": "no-store",  # "/" — the shell
    ".html": "no-store",  # any other page of the shell
    ".js": "no-cache",  # imported by URL, so a kept copy is a kept behaviour
    ".css": "no-cache",
}


class NoCacheStaticFiles(StaticFiles):
    """Static host that never lets the browser reuse what it already has.

    Starlette's default (`ETag` + `Last-Modified`, no `Cache-Control`) makes the
    browser revalidate and then reuse: after the app's files change on disk, the
    next visit is answered with a `304` and the browser runs the copy it kept —
    which is what left a user staring at a page whose every control was dead, with
    nothing in the server log to explain it. So no revalidation is offered at all
    (`is_not_modified` is always false), the response bodies are always current,
    and the directives above forbid reuse on top of that.

    Reloading the page is therefore enough to run the current version. That is
    worth the loopback bandwidth it costs.
    """

    def is_not_modified(self, response_headers, request_headers) -> bool:
        return False

    def file_response(self, full_path, stat_result, scope, status_code=200):
        response = super().file_response(full_path, stat_result, scope, status_code)
        path = Path(os.fspath(full_path))
        response.headers["Cache-Control"] = STATIC_CACHE_CONTROL.get(
            path.suffix, "no-cache"
        )
        return response


class CorpusCreate(BaseModel):
    """Body of ``POST /api/corpora``: a name and the folder(s) to index."""

    name: str = Field(min_length=1, max_length=120)
    sources: list[str] = Field(min_length=1)


class CorpusPatch(BaseModel):
    """Body of ``PATCH /api/corpora/{id}``: settings to merge in."""

    name: str | None = None
    settings: dict | None = None


class QueryBody(BaseModel):
    """Body of ``POST /api/query``: the question and the files it may be asked of.

    ``include_sources`` is the ask pane's "this file" scope (omitted for the
    whole corpus); ``exclude_sources`` is every file the user has hidden from
    context. A file named in both is excluded (see
    :func:`raggy.pipeline.source_filter`).
    """

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    include_sources: list[str] | None = None
    exclude_sources: list[str] | None = None


def _error(message: str, status: int = 400, **extra) -> JSONResponse:
    return JSONResponse({"error": {"message": message, **extra}}, status_code=status)


def _runtime_or_error(
    state: AppState, corpus_id: str | None
) -> CorpusRuntime | JSONResponse:
    """Resolve the requested (or active) corpus, or the error to return."""
    try:
        return state.runtime(corpus_id)
    except CorpusError as e:
        return _error(str(e), status=404)


def create_app(home: Path | None = None) -> FastAPI:
    """Build the GUI's FastAPI application."""
    store = CorpusStore(home)
    state = AppState(store)

    app = FastAPI(title="raggy GUI", version="0.1.0", docs_url="/api/docs")
    app.state.raggy = state

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):  # pragma: no cover
        logger.exception("Unhandled error for %s", request.url.path)
        return _error(f"{exc.__class__.__name__}: {exc}", status=500)

    # -- health & capabilities --------------------------------------------
    @app.get("/api/health")
    def health() -> dict:
        converter = find_libreoffice()
        return {
            "status": "ok",
            "home": str(store.home),
            "features": {
                "document_conversion": converter is not None,
                "converter": converter.name if converter else None,
                "streaming": False,
                "chat_memory": False,
            },
        }

    # -- corpora -----------------------------------------------------------
    @app.get("/api/corpora")
    def list_corpora() -> dict:
        active = store.active()
        return {
            "active": active.id if active else None,
            "corpora": [
                corpus.as_dict(active=active is not None and corpus.id == active.id)
                for corpus in store.list()
            ],
        }

    @app.post("/api/corpora")
    def create_corpus(body: CorpusCreate) -> dict:
        try:
            corpus = store.create(body.name, body.sources)
        except CorpusError as e:
            return _error(str(e))
        except FileNotFoundError as e:
            return _error(str(e))
        except Exception as e:  # noqa: BLE001 - validation errors must be shown, not raised
            return _error(f"could not create corpus: {e}")
        store.set_active(corpus.id)
        return {"corpus": corpus.as_dict(active=True)}

    @app.patch("/api/corpora/{corpus_id}")
    def patch_corpus(corpus_id: str, body: CorpusPatch) -> dict:
        try:
            corpus = store.get(corpus_id)
            if corpus is None:
                return _error(f"unknown corpus: {corpus_id}", status=404)
            settings = dict(body.settings or {})
            # A corpus's DB directory is what keeps its manifest separate from
            # every other corpus's, and its sources are changed through their own
            # endpoint. Neither is patchable here, whatever the body says.
            settings.pop("db_directory", None)
            settings.pop("sources", None)
            if body.name:
                settings["name"] = body.name
            if settings:
                corpus = store.update_settings(corpus_id, settings)
        except CorpusError as e:
            return _error(str(e), status=404)
        except Exception as e:  # noqa: BLE001
            return _error(str(e))
        state.drop(corpus_id)
        return {"corpus": corpus.as_dict(active=store.active_id == corpus_id)}

    @app.delete("/api/corpora/{corpus_id}")
    def delete_corpus(corpus_id: str, delete_db: bool = True) -> dict:
        try:
            state.drop(corpus_id)
            store.remove(corpus_id, delete_db=delete_db)
        except CorpusError as e:
            return _error(str(e), status=404)
        active = store.active()
        return {"active": active.id if active else None}

    @app.post("/api/corpora/{corpus_id}/activate")
    def activate_corpus(corpus_id: str) -> dict:
        try:
            state.activate(corpus_id)
        except CorpusError as e:
            return _error(str(e), status=404)
        active = store.active()
        return {"active": active.id if active else None}

    @app.post("/api/corpora/{corpus_id}/sources")
    def add_source(corpus_id: str, body: CorpusCreate) -> dict:
        try:
            corpus = store.add_source(corpus_id, body.sources[0])
        except CorpusError as e:
            return _error(str(e))
        state.drop(corpus_id)
        return {"corpus": corpus.as_dict(active=store.active_id == corpus_id)}

    # -- browse the local filesystem (for folder selection) ----------------
    @app.get("/api/browse")
    def browse(path: str | None = None, show_files: bool = True) -> dict:
        target = Path(path).expanduser() if path else Path.home()
        try:
            target = target.resolve()
        except OSError as e:
            return _error(f"cannot open {path}: {e}")
        if not target.is_dir():
            return _error(f"not a directory: {target}")

        entries = []
        try:
            children = sorted(
                target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
            )
        except PermissionError:
            return _error(f"permission denied: {target}", status=403)
        except OSError as e:
            return _error(f"cannot read {target}: {e}")

        for child in children:
            if child.name.startswith("."):
                continue
            is_dir = child.is_dir()
            if not is_dir and not show_files:
                continue
            entries.append({"name": child.name, "path": str(child), "is_dir": is_dir})

        parent = str(target.parent) if target.parent != target else None
        return {
            "path": str(target),
            "parent": parent,
            "home": str(Path.home()),
            "entries": entries[:2000],
        }

    # -- indexed contents & source files -----------------------------------
    @app.get("/api/corpora/{corpus_id}/documents")
    def list_documents(corpus_id: str) -> dict:
        runtime = _runtime_or_error(state, corpus_id)
        if isinstance(runtime, JSONResponse):
            return runtime
        try:
            return runtime.documents()
        except Exception as e:  # noqa: BLE001
            return _error(f"could not read the corpus: {e}", status=500)

    @app.get("/api/corpora/{corpus_id}/file")
    def get_file(corpus_id: str, path: str, cache: bool = False):
        runtime = _runtime_or_error(state, corpus_id)
        if isinstance(runtime, JSONResponse):
            return runtime
        target = runtime.authorize_path(path, cache_file=cache)
        if target is None:
            return _error("that file is not part of this corpus", status=403)
        return FileResponse(
            target,
            media_type=_media_type(target),
            filename=target.name,
            headers={"Cache-Control": "no-store"},
        )

    # -- indexing ----------------------------------------------------------
    @app.post("/api/corpora/{corpus_id}/refresh")
    def refresh(corpus_id: str) -> dict:
        runtime = _runtime_or_error(state, corpus_id)
        if isinstance(runtime, JSONResponse):
            return runtime
        try:
            report = runtime.refresh()
        except RuntimeError as e:
            return _error(str(e), status=409)
        except Exception as e:  # noqa: BLE001
            return _error(f"{e.__class__.__name__}: {e}", status=500)
        return {"report": report.as_dict()}

    @app.get("/api/corpora/{corpus_id}/status")
    def status(corpus_id: str) -> dict:
        try:
            return state.status(corpus_id)
        except CorpusError as e:
            return _error(str(e), status=404)

    # -- questions ---------------------------------------------------------
    @app.post("/api/query")
    def query(body: QueryBody, corpus_id: str | None = None) -> dict:
        runtime = _runtime_or_error(state, corpus_id)
        if isinstance(runtime, JSONResponse):
            return runtime
        try:
            return runtime.query(
                body.query,
                include_sources=body.include_sources,
                exclude_sources=body.exclude_sources,
            )
        except RuntimeError as e:
            return _error(str(e), status=409)
        except ValueError as e:
            # An empty retrieval scope, refused before any model runs: the
            # message says which control to change.
            return _error(str(e), hint="Unhide a file, or ask about all files.")
        except Exception as e:  # noqa: BLE001
            return _error(
                f"{e.__class__.__name__}: {e}",
                status=500,
                hint=_query_hint(e),
            )

    # Serve the UI itself. Mounted last so it cannot shadow an API route.
    if WEB_DIR.is_dir():
        app.mount(
            "/", NoCacheStaticFiles(directory=str(WEB_DIR), html=True), name="web"
        )
    else:  # pragma: no cover - the package always ships the directory
        logger.warning("GUI web assets missing at '%s'; API only.", WEB_DIR)

    return app


def _query_hint(exc: Exception) -> str | None:
    """An actionable next step for the failure shapes a GUI user can fix."""
    text = str(exc).lower()
    if "connection refused" in text or "failed to connect" in text:
        return "Ollama is not reachable; start it with `ollama serve`."
    if "api key" in text or "api_key" in text:
        return (
            "Set the provider's API key (OPENAI_API_KEY / ANTHROPIC_API_KEY / "
            "GEMINI_API_KEY) and restart the GUI."
        )
    if "not found" in text and "model" in text:
        return "The configured model is not available locally; pull it with `ollama pull <model>`."
    return None


_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/plain; charset=utf-8",
    ".markdown": "text/plain; charset=utf-8",
}


def _media_type(path: Path) -> str:
    return _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
