# raggy GUI — Build Plan

## Context
`raggy` is a local-first RAG library (LangChain + Chroma + Ollama). The
repo is available locally in this workspace — read the actual source
directly rather than searching for it online. It currently exposes a CLI
and a Python library interface (`run_pipeline(query)`, `source_label(doc)`).
This plan covers building a GUI on top of it.

Note: some specifics below (file/module names, exact manifest fields) were
inferred from the project's README rather than the source itself. Treat
the actual repo code as the source of truth wherever it disagrees with
this document — especially for the verification tasks.

## Explicit non-goals (do not build)
- Token-by-token streaming output. `run_pipeline` returns a full response,
  not a generator — out of scope for this build.
- Conversation memory / multi-turn continuity. Each query is independent.
  Flagged as risky for phi4-mini specifically (small model + Q4 quant is
  more prone to conflating chat history with retrieved context) — not
  being pursued regardless of framework chosen.
- Concurrent multi-user session isolation. Single local user only.
- Automatic folder-watching for re-indexing (e.g. `watchdog`). Manual
  refresh is sufficient; folder-watching's debounce/partial-write/
  concurrency complexity isn't worth it for this use case.
- Drag-and-drop file upload into the GUI. Users add files to source
  folders outside the app; the app indexes what it finds via folder
  selection, not file upload.

## Feature list

### 1. Manual re-index ("Refresh") button
- Calls the existing indexing entry point (`initialize_db` /
  equivalent) against the active config's `sources` and `db_directory`.
- Relies on raggy's existing manifest-diff logic (per-file SHA-256 hash
  map) to do incremental updates — no new indexing logic needed here,
  just a UI trigger + progress indicator + success/error feedback.
- After a run, surface a summary: files indexed, files skipped
  (unsupported extension), files unchanged. Requires diffing the walked
  file list against the extensions raggy actually processes — this is
  new UI-facing reporting, not present today.

### 2. Multiple configs, added via folder selection
- Support multiple named "corpora," each backed by its own `sources`
  list and its own `db_directory` (avoid one shared DB across corpora —
  confirm this is enforced, since switching `sources` without changing
  `db_directory` will trigger churn against the manifest).
- UI: "Add corpus" flow where the user selects a folder from the local
  filesystem (not drag-and-drop) and gives it a name; store as a new
  config entry (could be one YAML file per corpus, or one config file
  with multiple named entries — agent's call, but keep `db_directory`
  isolation per corpus either way).
- A dropdown/selector to switch which corpus is active for querying.

### 3. Document viewer (main pane) + chat (side pane) layout
- Reorient the UI so the retrieved/source document is the primary
  content area, with the chat/query interface as a sidebar rather than
  the main focus.

### 4. Jump-to-citation (search-and-highlight, not bounding-box overlay)
- Approach: on citation click, switch the main pane to the cited file,
  jump to the correct page (PDF/converted DOCX/PPTX), then run a
  text-search-and-highlight within that page using the chunk's own
  text content as the search string (e.g. via a PDF viewer component's
  built-in find/highlight API — most `pdf.js`-based viewers expose this).
- This approach deliberately avoids needing bounding-box metadata for
  PDFs, DOCX, PPTX, and text-based formats — page number (or line range,
  for text formats) plus the chunk's text content is sufficient.
- **Images (standalone OCR'd images) are the exception** — no text layer
  exists to search against. Two options, pick one for v1:
  - (a) Simple fallback: display the raw OCR'd text in a panel next to
    the image, no interactive jump/highlight.
  - (b) Fuller fidelity: capture RapidOCR's bounding boxes at index
    time and burn/overlay an invisible positioned text layer onto the
    image for search purposes — this requires modifying the indexing
    code (RapidOCR already produces bounding boxes; raggy currently
    discards them, keeping only flat text). Larger lift — do not build
    unless (a) proves insufficient in practice.
- Recommend starting with (a) for images and revisiting only if it's a
  real pain point.

### 5. Format conversion during indexing (DOCX/PPTX → PDF)
- Convert DOCX and PPTX to PDF as a preprocessing step during indexing,
  using LibreOffice headless (`soffice --headless --convert-to pdf`).
  - Rationale: unifies rendering (one PDF viewer component handles
    native PDF + converted DOCX + converted PPTX) and unifies location
    metadata (page numbers via the same `PyPDFLoader` path used for
    native PDFs), instead of building separate DOCX/PPTX-specific
    viewers and metadata schemes.
  - Chunking should happen on the **converted PDF**, not on raggy's
    native DOCX/PPTX text extraction, so citation page numbers line up
    with what the viewer actually displays. This likely means DOCX/PPTX
    effectively get treated as "become a PDF, then go through the
    existing PDF path" rather than keeping two parallel extraction
    routes.
- Store converted files in a cache subfolder (e.g.
  `db_directory/render_cache/`), keyed by the same per-file SHA-256
  hash raggy's manifest already computes. This piggybacks on existing
  change-detection — a changed source file already triggers
  re-embedding, so trigger cache regeneration on the same condition,
  no separate invalidation logic required.
- Concurrency: LibreOffice headless does not handle parallel invocations
  well (can hang or produce corrupt output). Serialize conversions
  (queue or lock) rather than firing off multiple `soffice` calls in
  parallel — this matters as soon as a newly-added corpus folder has
  more than one DOCX/PPTX file to convert on first index.
- Scanned PDFs (no text layer, OCR'd page-by-page per raggy's existing
  behavior) get page-level jump like native PDFs, but no sub-page
  search/highlight unless treated the same as the images case above.

## Verification tasks (check before/while building, not assumed)

1. **Delete-file behavior.** The manifest is documented as tracking
   added/modified files explicitly; deleted files aren't mentioned.
   Verify: remove a file from a `sources` folder, re-run indexing,
   inspect `manifest.yaml` and the Chroma collection to confirm the
   file's vectors are actually pruned rather than left orphaned. If
   they're not pruned, build the deletion-handling logic (remove from
   manifest `files` map, remove corresponding vectors from Chroma, and
   note that BM25 already gets a full rebuild per the existing docs, so
   that side is self-correcting).
2. **Unsupported file handling.** Confirmed to be silently skipped per
   current docs — verify this holds for corrupted-but-correctly-named
   files too (e.g. a truncated/corrupt PDF with a `.pdf` extension):
   does it skip cleanly or throw? Decide whether the refresh summary
   (see Feature 1) needs to also surface "failed to process" separately
   from "skipped as unsupported."

## Suggested build order
1. Refresh button wired to existing indexing entry point + skip/success
   summary reporting.
2. Multiple named configs/corpora with folder-selection UI and per-corpus
   `db_directory` isolation.
3. Delete-file verification (and fix if needed) — do this before relying
   on corpus management in step 2 feeling "safe" to use.
4. Document viewer main pane + chat side pane layout (no citation
   linking yet — just get files rendering correctly per format).
5. DOCX/PPTX → PDF conversion pipeline, with render-cache and LibreOffice
   concurrency handling.
6. Jump-to-citation via page-jump + text-search-highlight, built against
   PDF/converted-DOCX/converted-PPTX/text formats.
7. Image fallback display (raw OCR text panel) as the simple v1 handling
   for the one format that can't support search-and-highlight.
