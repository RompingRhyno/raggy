# raggy

A lightweight CLI tool for Retrieval-Augmented Generation (RAG) over local documents built with LangChain, Chroma, and Ollama. Hybrid database (vector + BM25 index) and embedding generation run fully locally. Answer generation can run either via a local LLM or remotely using an API key. `raggy` supports most common document formats and handles images/scans automatically via OCR.

Usage example -- CLI returns an answer based on your documents, and citations along with their locations and relevance scores:

<img src="assets/cli_demo.png">

These are all currently supported file formats (all other formats are ignored):

| Type | Extensions |
| --- | --- |
| Documents | `.pdf`, `.docx`, `.pptx` |
| Text | `.txt`, `.md`, `.markdown` |
| Web | `.html`, `.htm` |
| Images (OCR) | `.png`, `.jpg`, `.jpeg`, `.bmp` |

## Prerequisites

Ollama is required for running the local embedding model (which feeds the on-disk vector DB), and also a local LLM (if needed). To install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh

# may need to start ollama after installation using the app or:
ollama
# or
ollama serve
```

If an API key will be used for accessing an LLM remotely, a standard environment variable needs to be set (one of the following):

```bash
export GEMINI_API_KEY=...
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

Python 3.10 or newer is required; installing `uv` is recommended:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

Clone the repo:

```bash
git clone https://github.com/paulknysh/raggy.git && cd raggy
```

Then install using:

```bash
# with uv
uv tool install -e .

# with pipx
pipx install -e .
```

For now, cloning + editable install is picked as a preferred installation method, as it allows you to experiment with the demo dataset, run the eval harness, and edit/debug code if needed. In the future, direct install via `uv tool install git+https ...`/`pipx install git+https ...` will be used instead.

## Usage (GUI)

The GUI puts the retrieved document in the main pane and the chat in a sidebar,
so an answer is something you check against its source rather than a wall of text
with citations underneath it. It runs as a local web server on the loopback
interface:

```bash
make gui
# or
raggy-gui
# or
uv run raggy-gui --port 8790
```

Then open http://127.0.0.1:8765. Everything is local: the server binds to
127.0.0.1, the vectors and the PDF viewer assets ship with the repo, and there is
no build step (plain HTML/CSS/ES modules).

### Corpora

Instead of one config file, the GUI holds any number of named **corpora**. Each
corpus is one config file plus **its own DB directory**, kept under
`~/.raggy/gui/corpora/` (override the location with `RAGGY_GUI_HOME`). Adding a
corpus means picking a folder or file in the folder browser and naming it — no
drag-and-drop:

- the folder's files are indexed on the first refresh;
- switching corpora switches both the DB and the chat;
- deleting a corpus removes its config and its DB, and never touches your files.

`db_directory` is deliberately not editable per corpus: two corpora sharing one
DB would diff their file sets against a single manifest and re-embed (or delete)
the wrong chunks.

### Refresh and what it reports

**Refresh** re-indexes the active corpus through the same incremental logic the
CLI uses (only added/changed files are re-embedded; deleting a file prunes its
vectors), then reports per file:

| Reported | Meaning |
| --- | --- |
| indexed | read and embedded by this run, with its chunk count |
| unchanged | already up to date, so no work |
| failed | supported file that could not be used, with the reason (a corrupt PDF, or a file with no extractable text) |
| skipped | unsupported extension, or a source that no longer exists |
| removed | deleted from the sources folder; its vectors are pruned |
| orphans | chunks in the DB with no file on disk (normally empty) |

Failures are remembered in the manifest, so a broken file is retried on the next
refresh instead of being re-read as if it were brand new — and a *fixed* file is
picked up immediately.

### Asking about one file, or with files hidden

Above the question box, two buttons choose what a question is asked of: the whole
corpus, or the file currently open in the viewer (disabled while nothing is
open). The choice is sent with the question, so an answer can only be built from
what it says — not from the whole corpus with the rest filtered out afterwards.

Each row of the file list carries its status as a clickable dot: green means the
file is in context, red means it is hidden from it. A hidden file is excluded
server-side from retrieval — its chunks are never retrieved by either the vector
or the lexical arm — and the set is remembered per corpus. Hiding is about what
answers may cite, not about what can be read: a hidden file still opens in the
viewer. The dot draws its border when the row is hovered, which is where the
"I am looking at this file" gesture actually happens.

### Document viewer and jump-to-citation

The main pane renders PDFs with a bundled pdf.js, text files with line numbers,
images with their OCR text, and HTML as markup. Clicking a citation:

- **PDF** (native, plus converted DOCX/PPTX) — opens the file at the cited page
  and highlights the chunk's own text on that page;
- **text/markdown** — scrolls to the cited line range and highlights the chunk;
- **HTML** — scrolls to the chunk text in the rendered page;
- **image** — shows the image next to the chunk's OCR text. Standalone images
  have no text layer to search, so there is no highlight to jump to; this is the
  documented v1 fallback for that one format. An image OCR read nothing out of
  gets no text panel at all, and is hidden from the file list with the other
  text-less files.

The PDF page is a canvas with pdf.js's text layer over it: one absolutely
positioned span per text run, transparent, sitting exactly on the rendered
glyphs, so text can be selected and searched while only the canvas is visible.
Those spans are positioned by an inline transform that pdf.js computes, but the
rules that make them absolute — and the scale variable that sizes them — are the
host's responsibility, since `TextLayer` is the API for building your own viewer.
They are reproduced in `raggy/gui/web/styles.css` from pdf.js's own
`web/pdf_viewer.css`, which is vendored alongside the library
(`python -m raggy.gui.vendor_pdfjs`) so the two can be compared.

### Office documents

With LibreOffice installed, DOCX and PPTX are rendered to PDF once per content
hash into `<db_directory>/render_cache/` and chunked from that PDF, so their page
numbers match what the viewer shows and one viewer covers every paged format.
Conversions are serialized (LibreOffice headless does not tolerate parallel
invocations) and cached against the same SHA-256 the manifest already uses, so
editing a document re-renders it and nothing else. Without LibreOffice those
formats fall back to their native text extraction and the GUI says so; installing
it later triggers a rebuild of the affected corpus.

An image's OCR text is cached in the same hash-keyed directory
(`<db_directory>/render_cache/<sha256>/extracted.txt`). Indexing OCRs an image to
embed it, and the viewer then wants to show that same text beside the image;
reading it back out of the vector store does not work, because an image's text is
split into chunks like any other document and the splitter drops separators at
the window boundaries. Caching it as it is read is what keeps OCR from running a
second time — otherwise every listing of an image-heavy corpus pays for a pass
over every image, and on a 29-image corpus that measured 12s against 0.1s.

## Usage (CLI)

First, run this command inside the cloned repo:

```bash
make config
```

It creates your own user config (`config.yaml`) where all your execution parameters live. While `config.yaml` comes with defaults you can test, you should populate `sources` (your input folders/files) and `db_directory` (DB location) sections with your preferred paths. For a detailed overview of all config parameters, see [Configuration](#configuration).

To start the CLI, use the `raggy <path-to-config-file>` command:

```bash
raggy config.yaml
```

> [!IMPORTANT]
> CLI automatically pulls all models listed in `config.yaml` and (re-)indexes your documents -- this might take a while on the first run, depending on models chosen, document count/size, and whether OCR is needed (scans, images, etc).

> [!IMPORTANT]
> Relative paths in `config.yaml` resolve against the current directory (from where `raggy` command is executed). Keep that in mind if you want to run `raggy` from other locations. To be safe, just always use absolute paths in your config file.

## Usage (programmatic)

Here is the basic snippet you can run via `uv run snippet.py`:

```python
from raggy import run_pipeline, source_label

query = "What is TS-RAG?"

response, retrieved_docs = run_pipeline(query, config_path="config.yaml")

print(f"\n*** RESPONSE:\n\n{response}\n\n***")

for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n\n=== Doc {i} [{source_label(doc)}] ===\n\n")
    print(doc.page_content)
```

## Configuration

All runtime settings are defined in the config file:

| Setting | Description |
| --- | --- |
| `sources` | list of source directories and/or files |
| `db_directory` | location where the DB itself is stored |
| `embedding_model` | Ollama embedding model (e.g. `nomic-embed-text`) |
| `chunk_size` | chunk size in characters |
| `chunk_overlap` | character overlap between adjacent chunks |
| `embed_batch_size` | max number of chunks embedded per batch into Chroma (`100` in the shipped config); the number of batches is derived automatically |
| `llm_provider` | where generation runs: `ollama` (local, the shipped value) or `openai`/`anthropic`/`google` (via API) |
| `llm_model` | chat model for generation (e.g. `phi4-mini` locally, or a remote model name like `gemini-3.7-flash`) |
| `llm_temperature` | LLM sampling temperature |
| `retrieve_k` | total chunks retrieved per query, split across the dense and lexical retrievals (`50` in the shipped config) |
| `hybrid_alpha` | fraction of `retrieve_k` spent on the vector retrieval; the remainder goes to lexical (`1.0` = vector only, `0.0` = lexical only, `0.5` in the shipped config) |
| `rerank_model` | Hugging Face ID of the cross-encoder model (e.g. `cross-encoder/ms-marco-MiniLM-L6-v2`) |
| `rerank_k` | number of chunks returned by the cross-encoder (must be `<= retrieve_k`) |
| `rerank_threshold` | drops reranked chunks whose relevance score is below this value (`0.0` = disabled, `0.3` in the shipped config) |
| `system_prompt` | system prompt dictating how the LLM should answer; must contain a `{context}` placeholder |

Notes:

- The current default config parameters were tested on a basic MacBook Air with 8GB RAM. Switching to much heavier local models likely needs appropriate GPU/memory.

- Chroma doesn't seem to be able to embed all chunks in one go; therefore, `embed_batch_size` was introduced so it's done in batches instead. 100 seems like a reasonable default, but if you get Chroma errors during embedding (such as `Error: Post "http://127.0.0.1:50175/tokenize": EOF (status code: 400)`), try lowering `embed_batch_size` further.

## Pipeline

Below are the main steps in the RAG pipeline (assuming the DB is already created):

**[1] Hybrid retrieval.** `retrieve_k` is a total *candidate budget*, split by
`hybrid_alpha` between two retrievals over the whole corpus:

- **dense** retrieval -- nearest chunks in Chroma by embedding similarity (good at
  paraphrase and synonyms);
- **lexical** retrieval -- the persisted `bm25s` index (good at exact terms:
  identifiers, names, acronyms, numbers).

So `retrieve_k: 50` with `hybrid_alpha: 0.5` takes 25 chunks from each arm. The two
ranked lists are merged by **reciprocal rank fusion**, which collapses duplicates and
needs no score calibration between the two very different scales. Fusion weights are
uniform on purpose: `hybrid_alpha` already sets each arm's influence by deciding how
many candidates it contributes.

**[2] Cross-encoder reranking.** Stage 1 favors recall and is noisy. The reranker
(`rerank_model`, run locally on onnxruntime) pushes the query and the chunk through
the model *together* and emits one relevance score per pair -- far sharper than
cosine distance between separately embedded texts, and affordable on ~50 chunks
though not on the whole corpus. The top `rerank_k` chunks survive.

**[3] Score threshold.** Each chunk carries its reranker score in
`doc.metadata["relevance_score"]`, and anything below `rerank_threshold` is dropped.
This keeps `rerank_k` from polluting the context when the corpus has no good answer;
`0.0` disables it. The cutoff is also **fail-open in the large**: when it would
discard *every* candidate, the best few are passed through instead of nothing. An
absolute cutoff only means something if the cross-encoder's scale is comparable
between queries, and it is not — "what documents are in this corpus" or "who wrote
this paper" score their genuinely relevant chunks near zero, while content questions
score near one. Emptying the context there does not protect the answer, it just
guarantees the "I cannot answer" reply and hides the sources that were retrieved.
The system prompt already tells the model to say when the context does not answer
the question, so a weak chunk is better evidence than silence.

**[4] Generation.** The survivors are concatenated into `{context}` in
`system_prompt` and sent to the configured LLM, along with the chat history (in chat
mode) and the question. The exact chunks the model saw are returned to the caller --
`retrieved_docs` above, and the source table the CLI prints.

## The DB mechanics

DB creates/updates itself automatically, so you don't need to think about it. Below is just a high-level overview of the mechanics.

### How the DB is created

On the first run, `initialize_db` (in `raggy/indexing.py`) loads every
supported file under each entry in `sources`, splits them into overlapping chunks,
and embeds them into Chroma. It also writes a `manifest.yaml` into the persist
directory recording these parameters:
- `sources`
- `chunk_size`
- `chunk_overlap`
- `embedding_model`
- `file_converters` — which DOCX/PPTX → PDF converters were in play (empty when
  none is installed), since a change there changes the text of every chunk
- `files` — a `{file path: SHA-256 content hash}` map of every indexed file
- `attempted` / `failed` — every walkable file, and the subset that could not be
  used (a corrupt file, or one with no extractable text), each with its hash

The same chunks are also indexed with `bm25s` (a lexical BM25 index), stored in
`<db_directory>/bm25_index/`, so the lexical half of hybrid retrieval runs
without re-indexing at retrieval time.

### How/when the DB is re-indexed

The `manifest.yaml` is cheap to recompute, so for every new run it is computed and
compared against the existing one. What happens next depends on what changed:

- **Incremental update (the common case)** — the source files changed but
  `chunk_size`, `chunk_overlap`, and `embedding_model` did not. In this case, only added
  and modified files are reloaded and embedded. Untouched files are never
  re-embedded, so editing one file in a large corpus costs one file's worth of work.
- **Full rebuild** — `chunk_size`, `chunk_overlap`, or `embedding_model` changed
  (every stored vector is then invalid), or there is no manifest yet. The persist
  directory is wiped, and everything is indexed from scratch.

The BM25 index has no incremental update path, so it is rebuilt after every update —
from the chunks already stored in Chroma, which needs no embedding calls and no
re-reading of source files. This should be very fast anyway.

Files that cannot be used (a truncated PDF with a `.pdf` name, a document whose
text does not survive extraction) are recorded in the manifest's `failed` map
rather than silently retried forever or counted as indexed: an unchanged failed
file is left alone, a *changed* one is retried, and an explicit GUI refresh
retries them all.

## Demo dataset

This article (https://arxiv.org/abs/2608.06223v1) is used here as a demo dataset. It's an 8-page document -- each page is saved in different file formats (including PDF, plaintext, images, MS Office) and saved inside the `sample_docs` directory. This directory is specified in `config.yaml` by default.

## Demo eval

`eval` folder currently contains a basic harness to test pipeline performance on the demo dataset. You can run it by:

```bash
uv run eval/run_eval.py
```

It computes basic retrieval/generation metrics and produces a summary (both printed and saved to `eval/results.json`). The Q&A pairs are about `sample_docs`, so the harness always runs against `default_config/default_config.yaml` rather than your own `config.yaml`.

## TODOs

- [x] Incremental indexing (only re-embed files that changed)
- [x] Hybrid retrieval, tuning config defaults
- [x] Support for popular LLM providers via API keys
- [x] Conversation memory in chat mode
- [x] Pydantic validation of config file
- [x] GUI: named corpora, refresh reporting, document viewer + chat
- [ ] UX/UI tuning of CLI (improved commands/statuses, etc)
- [ ] Performance optimizations (DB creation/update, pipeline execution)

If some features are not working or missing, feel free to open an issue or a PR.

## License

MIT
