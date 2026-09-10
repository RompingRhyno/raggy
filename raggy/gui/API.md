# raggy GUI — HTTP API contract

This is the frozen interface between the GUI's backend (`raggy/gui/server.py`) and
its front end (`raggy/gui/web/`). The plan's non-goals apply here: no streaming
endpoint, no chat-memory parameter, no multi-user concerns.

Server: `raggy-gui` → `uvicorn` on `http://127.0.0.1:8765` (loopback only, single
local user). Static assets are mounted at `/`, so `/` serves `web/index.html`.
All API responses are JSON. File responses are raw bytes.

## Error shape

Every failure is HTTP 4xx/5xx with a JSON body:

```json
{ "error": { "message": "human readable sentence", "hint": "optional next step" } }
```

A 409 means the corpus is busy with another long operation (indexing vs querying);
the message says so. Show `message` verbatim, plus `hint` when present.

## Capabilities — `GET /api/health`

```json
{
  "status": "ok",
  "home": "C:\\Users\\me\\.raggy\\gui",
  "features": {
    "document_conversion": true,
    "converter": "libreoffice",
    "streaming": false,
    "chat_memory": false
  }
}
```

`document_conversion: false` means DOCX/PPTX are indexed from their native text
extraction instead of a rendered PDF. Surface that once, non-blockingly (a note in
the add-corpus / refresh UI), never as a blocker.

## Corpora

### `GET /api/corpora`

```json
{
  "active": "sample-docs",
  "corpora": [
    {
      "id": "sample-docs",
      "name": "Sample docs",
      "config_path": "...\\corpora\\sample-docs.yaml",
      "db_directory": "...\\corpora\\sample-docs.db",
      "sources": ["D:\\Dev\\raggy\\sample_docs"],
      "created": null,
      "active": true,
      "indexed": true
    }
  ]
}
```

`corpora` may be empty (`active: null`) — render an "add your first corpus" state.

### `POST /api/corpora` — add corpus by folder selection

```json
{ "name": "Papers", "sources": ["D:\\papers"] }
```

Returns `{ "corpus": { ...same shape as above, "active": true } }`, and the new
corpus becomes active. Errors: 400 with a message (missing name, missing folder,
folder does not exist, invalid settings).

### `POST /api/corpora/{id}/activate` → `{ "active": "papers" }`

### `PATCH /api/corpora/{id}`

```json
{ "name": "New name", "settings": { "retrieve_k": 50, "rerank_k": 5 } }
```

`settings` keys are the raggy config fields (`chunk_size`, `chunk_overlap`,
`embedding_model`, `llm_provider`, `llm_model`, `llm_temperature`, `retrieve_k`,
`hybrid_alpha`, `rerank_model`, `rerank_k`, `rerank_threshold`, `system_prompt`).
`sources` and `db_directory` are managed by the server and are not settable here.
Returns `{ "corpus": {...} }`. The next refresh re-indexes against the new settings.

### `POST /api/corpora/{id}/sources` → `{ "corpus": {...} }`

Body `{ "name": "...", "sources": ["D:\\more"] }` (only `sources[0]` is appended).
Adding a source does not delete anything; run a refresh afterwards.

### `DELETE /api/corpora/{id}?delete_db=true` → `{ "active": "next-id-or-null" }`

Removes the corpus config and, by default, its DB directory — raggy's own copy of
the files, one directory per corpus. Leaving it behind would orphan an index
nothing can reach and nothing will clean up, so `delete_db=true` is what the
GUI's Delete button sends and what its confirmation dialog says. Pass
`delete_db=false` to keep the index. The user's source files are never touched.

Deleting the active corpus moves `active` to the first one left, or to `null`.
The GUI must confirm before calling this: it is the one action in the app that
cannot be undone from the UI.

## Filesystem browsing (folder selection, not drag-and-drop)

### `GET /api/browse?path=<abs-path>&show_files=true`

`path` omitted → the user's home directory. Returns:

```json
{
  "path": "D:\\papers",
  "parent": "D:\\",
  "home": "C:\\Users\\me",
  "entries": [
    { "name": "2024", "path": "D:\\papers\\2024", "is_dir": true },
    { "name": "note.txt", "path": "D:\\papers\\note.txt", "is_dir": false }
  ]
}
```

Hidden entries are omitted. `parent` is `null` at a drive/root. Errors: 403
permission denied, 400 not a directory.

## Indexing

### `POST /api/corpora/{id}/refresh` — the Refresh button

Blocking: returns when the run is finished (models pulled, files read, chunks
embedded). While it runs, poll `GET /api/corpora/{id}/status` (~1s) for the
progress line. Serialize with a full-panel spinner/progress display.

```json
{
  "report": {
    "full_rebuild": false,
    "changed": true,
    "summary": "3 indexed, 12 unchanged, 1 failed, 2 skipped",
    "counts": { "indexed": 3, "unchanged": 12, "removed": 0, "failed": 1,
                "skipped": 2, "orphans": 0, "chunks": 84 },
    "indexed": [ { "path": "...\\a.pdf", "chunks": 12 } ],
    "unchanged": ["...\\b.txt"],
    "removed": [],
    "failed": [ { "path": "...\\broken.pdf", "error": "Stream has ended unexpectedly" } ],
    "skipped": [ { "path": "...\\notes.csv", "reason": "unsupported", "detail": ".csv" } ],
    "orphans": []
  }
}
```

Report semantics to render (this is the whole point of feature 1):

- **indexed** — files read and embedded now (chunk count each). Also true after a
  full rebuild, when every file is re-embedded.
- **unchanged** — supported files that were already up to date, so no work.
- **failed** — supported files that could not be processed. Show the error text;
  this is distinct from *skipped*.
- **skipped** — files whose extension raggy has no loader for (`reason:
  "unsupported"`), or a missing source entry (`reason: "missing"`).
- **removed** — files deleted from the sources folder; their vectors are pruned
  from the DB (verified behaviour).
- **orphans** — chunks in the DB with no file on disk (should normally be empty).

### `GET /api/corpora/{id}/status`

```json
{ "busy": true, "message": "[3/17] ingesting report.pdf ...",
  "elapsed_seconds": 12.4, "updates": 3, "corpus": "papers", "name": "Papers" }
```

`message` is a single line, ready to display. `busy` is true while a refresh or a
query holds the corpus.

## Indexed contents

### `GET /api/corpora/{id}/documents`

Backs the corpus file list in the UI.

```json
{
  "corpus": "papers",
  "name": "Papers",
  "db_directory": "...\\papers.db",
  "indexed": true,
  "counts": { "documents": 16, "chunks": 210, "failed": 1, "without_text": 1 },
  "documents": [
    {
      "path": "D:\\papers\\a.pdf",
      "name": "a.pdf",
      "kind": "pdf",
      "exists": true,
      "chunks": 12,
      "status": "indexed",
      "detail": "",
      "size": 521741,
      "content_url": "/api/corpora/papers/file?path=...",
      "source_url": "/api/corpora/papers/file?path=...a.pdf"
    }
  ]
}
```

`status` is `indexed`, `failed`, `new` (on disk, not indexed yet), `removed`.
`kind` is `pdf` | `document` | `presentation` | `text` | `web` | `image` | `other`.

Fields that exist for the file list's benefit:

- **`without_text`** counts the files to keep behind a toggle: no chunks and not
  `new`, i.e. unreadable or nothing extractable. `new` is excluded deliberately —
  such a file has not been read, so its text is unknown rather than absent. When
  asked for, those entries are shown **after** the readable ones, under a group
  heading.
- **`exists: false`** marks a manifest entry whose file was deleted since the last
  refresh. Such a row must not be openable: the file route answers 403.
- **`content_url`** is what a viewer opens. For a converted DOCX/PPTX it is the
  cached PDF, not the source file (pdf.js cannot read a `.docx`); `source_url` is
  the original.
- **`ocr_text`** (images only) is the text OCR read out of the file. An image is
  one document and is never split, so it has no chunk to carry that text; the
  value is empty when there was nothing to read, and the viewer then shows no OCR
  panel at all.

## Questions

### `POST /api/query?corpus_id=<id>` (body `{ "query": "...", "include_sources": [...], "exclude_sources": [...] }`)

`corpus_id` is optional and defaults to the active corpus. Blocking; no streaming.

`include_sources` / `exclude_sources` are optional and scope *retrieval* — which
files an answer may be built from. Both take absolute paths exactly as
`/documents` reports them (other spellings of the same path are folded and
matched). `include_sources` limits the question to those files (the ask pane's
"This file"); `exclude_sources` removes files from whatever that selected (the
files the user has hidden from context). A file named in both is **excluded**.
Omit both to search the whole corpus.

Both halves reach the retrieval arms, which express a scope differently: the
vector store takes a metadata filter, BM25 is told which files it may return.
400 with `"no files are selected for this question"` when the scope names no
file the corpus has — including the case where every file is hidden. A question
asked of nothing would otherwise come back as a confident "I cannot find that",
which reads as a fact about the corpus rather than about the selection.

```json
{
  "query": "What is TS-RAG?",
  "include_sources": ["D:\\papers\\ts_rag-pages-1.pdf"],
  "exclude_sources": ["D:\\papers\\survey-2023.pdf"]
}
```

Response:

```json
{
  "answer": "markdown text from the model",
  "query": "What is TS-RAG?",
  "corpus": "papers",
  "citations": [
    {
      "label": "ts_rag-pages-1.pdf, page 3",
      "snippet": "first ~280 chars, whitespace-collapsed",
      "text": "the full chunk text",
      "score": 0.83,
      "metadata": { "source": "D:\\...\\ts_rag-pages-1.pdf", "page": 3, "source_kind": "pdf" },
      "target": {
        "source": "D:\\...\\ts_rag-pages-1.pdf",
        "name": "ts_rag-pages-1.pdf",
        "kind": "pdf",
        "content_url": "/api/corpora/papers/file?path=...",
        "page": 3,
        "start_line": null,
        "end_line": null,
        "search": "the chunk's first substantial line, for in-page highlighting",
        "highlightable": true,
        "preview_url": null,
        "note": "",
        "ocr_text": ""
      }
    }
  ]
}
```

`score` may be `null` (rerank_threshold can drop scored chunks; unscored chunks are
kept). `target` may be `null` if the chunk's source cannot be resolved — render the
citation as plain text in that case.

`target.kind` decides the viewer (feature 4 / plan feature 3):

| kind | viewer | jump |
| --- | --- | --- |
| `pdf` | pdf.js page canvas + text layer | `page`, then highlight `search` in that page |
| `text` | `<pre>` of the file | scroll to `start_line`, highlight `search` |
| `html` | rendered markup | scroll to first `search` occurrence, highlight |
| `image` | `<img>` + raw OCR text panel | no jump; `ocr_text` is the chunk's OCR text, `highlightable: false` |

## Serving document content

### `GET /api/corpora/{id}/file?path=<abs-path>[&cache=1]`

Returns the file (PDF `application/pdf`, images, `text/plain`, `text/html`).
`cache=1` marks a path inside the corpus's render cache (converted DOCX/PPTX) —
pass `content_url` through unchanged; the flag is already embedded.

403 when the path is not part of the corpus (the server only serves files under
the corpus's sources or its own render cache). Responses are `Cache-Control:
no-store` — do not assume browser caching.

## Front-end layout requirements

- **Main pane = document viewer.** The retrieved/cited document is the primary
  content: PDF page view, text view, or image + OCR panel.
- **Side pane = chat.** Question box, answer, and the citation list. Clicking a
  citation switches the main pane to the cited file, jumps to the page (or line),
  and highlights the `search` text there.
- **Ask scope.** A row of buttons above the question box picks what the question
  is asked of — the whole corpus, or the file currently in the viewer (disabled
  while nothing is open) — and the choice is sent as `include_sources` on
  `/api/query`. It is a choice, not a toggle: both options stay visible so the
  current scope never has to be inferred.
- **Files hidden from context.** Each row of the file list carries its status as
  a clickable marker: green (indexed) means answers may cite the file, red means
  it is hidden. Hidden files are sent as `exclude_sources` on every question and
  the set is remembered per corpus in `localStorage`. The marker draws its
  rounded-square border only on hover (and on keyboard focus), so the list is not
  a grid of boxes; a 400 from `/api/query` is the backstop if the whole corpus
  ends up hidden. The hidden count is shown beside the scope buttons.
- **Toolbar:** corpus selector (dropdown), **Refresh** button with the progress
  line from `/status`, an **Add corpus** flow that picks a folder via
  `/api/browse` (no drag-and-drop), and a **Delete corpus** button for the active
  one.
- **Add corpus names itself.** The name field is pre-filled with the folder being
  browsed and follows it while the user navigates — so `D:\papers\2024` suggests
  "2024", not the home directory the dialog opened on. Typing a name stops the
  suggestion: it is a default, not a value that keeps being overwritten. The
  dialog also states which folders will actually be indexed, because a folder
  that is browsed but not ticked is still used.
- **Delete corpus is confirmed, and says what it removes.** The dialog names the
  corpus ("Remove *Papers* and its index?"), so it can be answered when several
  corpora exist, and it states that the files on disk are left alone. The safe
  choice (Cancel) holds the focus. Disabled while a refresh holds the corpus.
- After a refresh, show the report summary: indexed / unchanged / failed /
  skipped / removed counts, with per-file details expandable (especially failures
  with their error text).
