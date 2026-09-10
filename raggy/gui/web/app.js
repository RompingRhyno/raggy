// raggy GUI — state, wiring, and the jump-to-citation flow.
//
// Module map: api.js unwraps the server's error shape, dom.js keeps text out of
// HTML, markdown.js renders model answers, textview.js and pdfview.js own the
// main pane. This file is only allowed to hold state and glue them together.
//
// Layout intent (build plan feature 3): the document is the main pane and the
// chat is a sidebar, so an answer is something you check against its source
// rather than a wall of text with citations underneath.

import { api } from "./js/api.js";
import { renderMarkdown } from "./js/markdown.js";
import { basename, clear, el, formatScore } from "./js/dom.js";
import { HtmlViewer, ImageViewer, TextViewer } from "./js/textview.js";

const PDF_KINDS = new Set(["pdf", "document", "presentation"]);

// Every module-level constant is declared before the tables that read them.
// `state` below is initialized in place, so a constant it refers to has to
// already exist: a `const` further down the file is in its temporal dead zone,
// and reading it throws before anything is wired up — the page then loads and
// does nothing at all, which is exactly what happens when this order slips.
const MODE_KEY = "raggy.viewerMode";
const HIDDEN_KEY_PREFIX = "raggy.hiddenContext";
const ASK_SCOPE_ALL = "all";
const ASK_SCOPE_FILE = "file";

const dom = {
  corpusSelect: byId("corpus-select"),
  refreshBtn: byId("refresh-btn"),
  addCorpusBtn: byId("add-corpus-btn"),
  deleteCorpusBtn: byId("delete-corpus-btn"),
  deleteModal: byId("delete-corpus-modal"),
  deleteBackdrop: document.querySelector("#delete-corpus-modal .modal-backdrop"),
  deleteWarning: byId("delete-corpus-warning"),
  deleteConfirm: byId("delete-corpus-confirm"),
  deleteCancel: byId("delete-corpus-cancel"),
  deleteError: byId("delete-corpus-error"),
  filesToggle: byId("files-toggle"),
  filesPanel: byId("files-panel"),
  fileList: byId("file-list"),
  fileCounts: byId("file-counts"),
  emptyFilesToggle: byId("empty-files-toggle"),
  viewer: byId("viewer"),
  viewerTitle: byId("viewer-title"),
  viewerStatus: byId("viewer-status"),
  pdfControls: byId("pdf-controls"),
  pdfPrev: byId("pdf-prev"),
  pdfNext: byId("pdf-next"),
  pdfPageInfo: byId("pdf-pageinfo"),
  chatLog: byId("chat-log"),
  chatForm: byId("chat-form"),
  queryInput: byId("query-input"),
  sendBtn: byId("send-btn"),
  queryState: byId("query-state"),
  scopeAll: byId("scope-all"),
  scopeFile: byId("scope-file"),
  scopeNote: byId("scope-note"),
  clearChatBtn: byId("clear-chat-btn"),
  topbarNote: byId("topbar-note"),
  refreshOverlay: byId("refresh-overlay"),
  refreshMessage: byId("refresh-message"),
  refreshElapsed: byId("refresh-elapsed"),
  refreshCancel: byId("refresh-cancel"),
  modal: byId("add-corpus-modal"),
  modalClose: byId("browse-close"),
  browsePath: byId("browse-path"),
  browseUp: byId("browse-up"),
  browseHome: byId("browse-home"),
  browseList: byId("browse-list"),
  browseSelection: byId("browse-selection"),
  browseUseCurrent: byId("browse-use-current"),
  browseCreate: byId("browse-create"),
  corpusName: byId("corpus-name"),
  corpusSources: byId("corpus-sources"),
  modalError: byId("add-corpus-error"),
};

function byId(id) {
  const node = document.getElementById(id);
  if (!node) throw new Error(`raggy: missing #${id} in index.html`);
  return node;
}

const state = {
  corpora: [],
  activeId: null,
  documents: [],
  documentCounts: null,
  // The file list is fetched, not known at load time: listing a corpus walks its
  // index (and OCRs its images), which on a large one takes seconds. Until the
  // answer arrives the pane says "loading", never "no files" — claiming a corpus
  // is empty before checking is how a full corpus looks broken for ten seconds.
  documentsLoading: false,
  documentsLoaded: false,
  documentsFor: null,
  indexed: null,
  // Files that produced no text are hidden until the user asks for them: they
  // cannot contribute to an answer, so they are noise in the list.
  showEmptyFiles: false,
  // Files the user took out of context by clicking their status button. Not a
  // view filter: these are kept out of retrieval server-side, so an answer can
  // never cite them. Restored per corpus (see hiddenKey).
  hiddenFiles: new Set(),
  // What a question is asked of: the whole corpus, or just the open file.
  askScope: ASK_SCOPE_ALL,
  health: null,
  lastReport: null,
  currentFile: null,
  viewer: null,
  viewerKind: null,
  viewerRequest: 0,
  refreshing: false,
  refreshStart: 0,
  refreshElapsedTimer: null,
  statusTimer: null,
  selection: [],
  // True once the user has typed a name for the new corpus. Until then the name
  // follows the folder they are browsing, so the default is the folder's own
  // name and stops being suggested the moment they disagree with it.
  corpusNameEdited: false,
  browsePath: null,
  browseParent: null,
  browseHomePath: null,
  errorCard: null,
};

// The live viewer instance, exposed for the same reason `window.__raggy` exists:
// a browser console (or an automated probe) needs a handle on the object that
// owns the main pane without re-creating one, which would fetch the document
// twice.
window.raggyViewer = null;

// -- boot ----------------------------------------------------------------

init().catch((error) => {
  showFatal(`raggy could not start: ${error.message}`);
});

async function init() {
  wireEvents();

  await loadHealth();
  await loadCorpora();
  renderDashboard();

  if (state.activeId) {
    await Promise.all([loadDocuments(), checkStatusOnce()]);
    renderDashboard();
  }
}

async function loadHealth() {
  try {
    state.health = await api.health();
  } catch (error) {
    // Health is advisory: a corpus can still be listed and queried without it.
    state.health = null;
    note(`Could not read server capabilities: ${error.message}`, true);
    return;
  }
  if (state.health?.features?.document_conversion === false) {
    note("DOCX/PPTX are indexed from their extracted text; no PDF rendering is available.", false);
  }
}

async function loadCorpora() {
  try {
    const payload = await api.corpora();
    state.corpora = payload.corpora || [];
    state.activeId = payload.active || null;
  } catch (error) {
    // Reset to the empty state so the toolbar stays usable: the likely fix is
    // adding a corpus, which is exactly what the empty state offers.
    state.corpora = [];
    state.activeId = null;
    note(`Could not list corpora: ${errorText(error)}`, true);
  }
}

// -- toolbar -------------------------------------------------------------

function wireEvents() {
  dom.corpusSelect.addEventListener("change", () => switchCorpus(dom.corpusSelect.value));
  dom.refreshBtn.addEventListener("click", () => refreshCorpus({ showSpinner: true }));
  dom.addCorpusBtn.addEventListener("click", () => openModal());
  dom.deleteCorpusBtn.addEventListener("click", () => openDeleteModal());
  dom.filesToggle.addEventListener("click", toggleFiles);
  dom.emptyFilesToggle.addEventListener("click", () => toggleEmptyFiles());
  dom.refreshCancel.addEventListener("click", () => {
    dom.refreshOverlay.hidden = true;
  });
  dom.pdfPrev.addEventListener("click", () => state.viewer?.goToPage?.(state.viewer.currentPage - 1));
  dom.pdfNext.addEventListener("click", () => state.viewer?.goToPage?.(state.viewer.currentPage + 1));
  dom.scopeAll.addEventListener("click", () => setAskScope(ASK_SCOPE_ALL));
  dom.scopeFile.addEventListener("click", () => setAskScope(ASK_SCOPE_FILE));

  dom.chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    ask(dom.queryInput.value);
  });
  dom.queryInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      ask(dom.queryInput.value);
    }
  });
  dom.clearChatBtn.addEventListener("click", () => {
    state.currentFile = null;
    clear(dom.chatLog);
    renderViewer();
  });

  // -- add-corpus modal
  dom.modalClose.addEventListener("click", closeModal);
  dom.modal.addEventListener("click", (event) => {
    if (event.target.dataset.action === "close") closeModal();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !dom.modal.hidden) closeModal();
  });
  dom.browseUp.addEventListener("click", () => browse(state.browseParent));
  dom.browseHome.addEventListener("click", () => browse(state.browseHomePath));
  dom.browseUseCurrent.addEventListener("click", () => {
    if (state.browsePath) toggleSelection(state.browsePath, basename(state.browsePath));
  });
  dom.browseCreate.addEventListener("click", createCorpus);
  dom.corpusName.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      createCorpus();
    }
  });
  // A name the user typed is theirs; until then the suggestion follows the
  // folder being browsed (see suggestCorpusName).
  dom.corpusName.addEventListener("input", () => {
    state.corpusNameEdited = dom.corpusName.value.trim().length > 0;
  });

  // -- delete-corpus modal
  dom.deleteCancel.addEventListener("click", closeDeleteModal);
  dom.deleteBackdrop.addEventListener("click", closeDeleteModal);
  dom.deleteConfirm.addEventListener("click", deleteActiveCorpus);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !dom.deleteModal.hidden) closeDeleteModal();
  });
}

function renderToolbar() {
  const select = dom.corpusSelect;
  clear(select);
  if (!state.corpora.length) {
    select.append(el("option", { value: "", text: "No corpora yet" }));
    select.disabled = true;
  } else {
    select.disabled = false;
    for (const corpus of state.corpora) {
      select.append(
        el("option", {
          value: corpus.id,
          text: `${corpus.name}${corpus.indexed ? "" : " — not indexed"}`,
        })
      );
    }
    select.value = state.activeId || state.corpora[0].id;
  }
  const busy = state.refreshing || !state.activeId;
  dom.refreshBtn.disabled = busy;
  dom.refreshBtn.textContent = state.refreshing ? "Refreshing…" : "Refresh";
  // Nothing to delete without an active corpus, and not while one is indexing:
  // a refresh holds the corpus's lock, and deleting it mid-run would leave the
  // run writing into a directory that has been removed.
  dom.deleteCorpusBtn.disabled = busy;
}

function toggleFiles() {
  const hidden = !dom.filesPanel.hidden;
  dom.filesPanel.hidden = hidden;
  dom.filesToggle.setAttribute("aria-pressed", hidden ? "false" : "true");
}

/** Show or hide the files that produced no text. */
function toggleEmptyFiles() {
  state.showEmptyFiles = !state.showEmptyFiles;
  renderFiles();
}

function note(message, loud = false) {
  dom.topbarNote.textContent = message;
  dom.topbarNote.hidden = !message;
  dom.topbarNote.classList.toggle("is-error", loud);
}

// -- rendering -----------------------------------------------------------

function renderDashboard() {
  renderToolbar();
  renderFiles();
  renderAskScope();
  renderViewer();
}

function activeCorpus() {
  return state.corpora.find((corpus) => corpus.id === state.activeId) || null;
}

function renderFiles() {
  clear(dom.fileList);
  const corpus = activeCorpus();
  if (!corpus) {
    dom.fileCounts.textContent = "";
    dom.emptyFilesToggle.hidden = true;
    dom.fileList.append(el("div", { class: "browse-empty", text: "No corpus selected." }));
    return;
  }

  const counts = state.documentCounts;
  dom.fileCounts.textContent = state.documentsLoading
    ? "loading…"
    : counts
      ? `${counts.documents} files · ${counts.chunks} chunks`
      : "not indexed";

  // Before anything else: a list that has not arrived yet is not an empty list.
  // The messages below are claims about the corpus, and none of them may be made
  // on the strength of a request that is still in flight.
  if (state.documentsLoading) {
    dom.emptyFilesToggle.hidden = true;
    dom.fileList.append(fileListLoading());
    return;
  }

  if (!state.documents.length) {
    dom.emptyFilesToggle.hidden = true;
    dom.fileList.append(el("div", { class: "browse-empty", text: emptyListMessage(corpus) }));
    return;
  }

  // Files that produced no text (unreadable, or nothing extractable) add nothing
  // to an answer, so they stay out of the way until asked for. A file on disk
  // that has not been indexed yet is *not* one of them: it has not been read, so
  // its text is unknown rather than absent, and it stays visible.
  const isWithoutText = (document) => !document.chunks && document.status !== "new";
  const empty = state.documents.filter(isWithoutText);
  const withText = state.documents.filter((document) => !isWithoutText(document));
  // When they are shown, they go last: the files that can answer something are
  // what the list is for, and a run of dead entries in the middle of it would
  // bury them. Each group keeps the order the server sent.
  const shown = state.showEmptyFiles ? [...withText, ...empty] : withText;

  dom.emptyFilesToggle.hidden = empty.length === 0;
  dom.emptyFilesToggle.textContent = `No text (${empty.length})`;
  dom.emptyFilesToggle.setAttribute("aria-pressed", state.showEmptyFiles ? "true" : "false");
  dom.emptyFilesToggle.classList.toggle("is-on", state.showEmptyFiles);

  let lastGroup = null;
  for (const document of shown) {
    // A heading where the readable files stop and the dead ones start, so the
    // bottom run reads as a group rather than as more of the same.
    const group = isWithoutText(document) ? "empty" : "text";
    if (group !== lastGroup) {
      if (group === "empty") {
        dom.fileList.append(
          el("div", { class: "file-group", text: `Files with no text (${empty.length})` })
        );
      }
      lastGroup = group;
    }
    // A manifest entry can outlive its file (it was deleted from the source
    // folder and the corpus has not been refreshed since). There is nothing to
    // open, so the row is inert and says so rather than leading to an error.
    const missing = document.exists === false;
    const hidden = state.hiddenFiles.has(document.path);
    const row = el("div", { class: "file-item", role: "listitem" });
    if (state.currentFile && state.currentFile.path === document.path) row.classList.add("is-current");
    if (!document.chunks) row.classList.add("is-empty");
    if (missing) row.classList.add("is-missing");
    if (hidden) row.classList.add("is-hidden-context");

    // Opening the file and taking it out of context are two different actions,
    // so they cannot be one button. The row *is* the open button — a nested one
    // would be invalid markup and would swallow the click — and the status
    // marker beside the name is its own button.
    const opener = el("button", {
      type: "button",
      class: "file-open",
      disabled: missing,
      title: missing ? "No longer on disk — press Refresh to drop it from the corpus" : document.path,
    });
    if (!missing) opener.addEventListener("click", () => openFile(document));
    // The head and the detail are separate rows: `.file-open` is a column, so a
    // detail span left as a direct child would otherwise sit beside the name.
    opener.append(
      el("span", { class: "file-row" }, [
        el("span", { class: "file-name", text: document.name, title: document.path }),
        el("span", { class: "file-chunks", text: document.chunks ? `${document.chunks} ch` : "" }),
        el("span", { class: "file-meta", text: document.kind }),
      ])
    );
    if (missing) {
      opener.append(el("span", { class: "file-detail", text: "no longer on disk — Refresh to drop it" }));
    } else if (document.detail) {
      opener.append(el("span", { class: "file-detail", text: document.detail }));
    }

    row.append(contextToggle(document, hidden), opener);
    dom.fileList.append(row);
  }

  // With the empties hidden, say so in place: an unexplained gap in the list
  // reads as files having gone missing.
  if (!state.showEmptyFiles && empty.length) {
    dom.fileList.append(
      el("button", {
        type: "button",
        class: "file-hidden-note",
        text: `${empty.length} file${empty.length === 1 ? "" : "s"} with no text hidden — show`,
        onclick: () => toggleEmptyFiles(),
      })
    );
  }

  if (!shown.length) {
    dom.fileList.append(
      el("div", {
        class: "browse-empty",
        text: "Every file in this corpus produced no text. Use “No text” to see them.",
      })
    );
  }
}

/**
 * The round button that takes a file in or out of context.
 *
 * It replaces the plain status dot that used to sit here: green (indexed) means
 * the file can be cited, red means it is hidden and no question will reach it.
 * The border is a hover-only affordance — see `.context-toggle` in styles.css —
 * because a permanent box around every file's status marker would read as
 * decoration, and the point of the hover is to say "this is clickable".
 */
function contextToggle(document, hidden) {
  const button = el("button", {
    type: "button",
    class: `context-toggle status-${document.status}`,
    "aria-pressed": hidden ? "false" : "true",
    onclick: () => toggleFileHidden(document),
  });
  if (hidden) button.classList.add("is-hidden");
  setContextToggleLabel(button, document, hidden);
  return button;
}

function setContextToggleLabel(button, document, hidden) {
  const what = hidden
    ? "Hidden from context: no answer will use this file. Click to include it again."
    : "In context: answers may cite this file. Click to hide it from context.";
  button.title = `${document.name} — ${what}`;
  button.setAttribute("aria-label", `${document.name}: ${what}`);
}

/** Take a file in or out of the context the model is allowed to answer from. */
function toggleFileHidden(document) {
  const hidden = !state.hiddenFiles.has(document.path);
  if (hidden) state.hiddenFiles.add(document.path);
  else state.hiddenFiles.delete(document.path);
  saveHiddenFiles();
  renderFiles();
  renderAskScope();
}

function hiddenKey(corpusId) {
  return `${HIDDEN_KEY_PREFIX}.${corpusId}`;
}

/** Bring back the files this corpus had hidden the last time it was open. */
function loadHiddenFiles() {
  state.hiddenFiles = new Set();
  if (!state.activeId) return;
  try {
    const stored = JSON.parse(window.localStorage.getItem(hiddenKey(state.activeId)) || "[]");
    // A corpus can have been refreshed (or replaced) since: a path that is no
    // longer in the list would hide nothing but would still be sent as an
    // exclusion, so only names the server just gave us are kept.
    const known = new Set(state.documents.map((document) => document.path));
    for (const path of Array.isArray(stored) ? stored : []) {
      if (known.has(path)) state.hiddenFiles.add(path);
    }
  } catch {
    // Private-mode browsers can refuse storage; losing the hidden set is not
    // worth failing the page load over.
    state.hiddenFiles = new Set();
  }
}

function saveHiddenFiles() {
  if (!state.activeId) return;
  try {
    window.localStorage.setItem(hiddenKey(state.activeId), JSON.stringify([...state.hiddenFiles]));
  } catch {
    // Same as above: the in-memory set still works for this session.
  }
}

// -- ask scope -----------------------------------------------------------

/** Ask the next question of the whole corpus, or of the file on screen. */
function setAskScope(scope) {
  // The button is disabled in both cases; this is the same rule for the keyboard
  // path and for any future caller that is not the button.
  const document = currentDocument();
  if (scope === ASK_SCOPE_FILE && (!document || state.hiddenFiles.has(document.path))) return;
  state.askScope = scope;
  renderAskScope();
}

/** The file the main pane is showing, with the file list's own details. */
function currentDocument() {
  if (!state.currentFile) return null;
  return state.documents.find((document) => document.path === state.currentFile.path) || null;
}

function renderAskScope() {
  const document = currentDocument();
  const hidden = Boolean(document) && state.hiddenFiles.has(document.path);
  // Two ways the "this file" scope can stop pointing at anything answerable: the
  // open file left the corpus (deleted on disk and refreshed away), or it was
  // just hidden from context. Both fall back to the whole corpus rather than
  // letting the next question fail.
  if (state.askScope === ASK_SCOPE_FILE && (!document || hidden)) state.askScope = ASK_SCOPE_ALL;
  const isFile = state.askScope === ASK_SCOPE_FILE && Boolean(document) && !hidden;
  const hiddenCount = state.hiddenFiles.size;

  dom.scopeAll.setAttribute("aria-pressed", isFile ? "false" : "true");
  dom.scopeAll.classList.toggle("is-on", !isFile);
  dom.scopeAll.title = hiddenCount
    ? `Answer from every file except the ${hiddenCount} hidden one${hiddenCount === 1 ? "" : "s"}`
    : "Answer from every file in this corpus";

  dom.scopeFile.disabled = !document || hidden;
  dom.scopeFile.textContent = document ? `This file: ${document.name}` : "This file";
  dom.scopeFile.setAttribute("aria-pressed", isFile ? "true" : "false");
  dom.scopeFile.classList.toggle("is-on", isFile);
  dom.scopeFile.title = !document
    ? "Open a file to ask about it on its own"
    : hidden
      ? `${document.name} is hidden from context — show it again to ask about it alone`
      : `Answer from ${document.name} only`;

  dom.scopeNote.textContent = hiddenCount ? `${hiddenCount} hidden from context` : "";
  dom.queryInput.placeholder = isFile
    ? `Ask about ${document.name}…`
    : "Ask a question about this corpus…";
}

/** The retrieval scope a question carries: `{include_sources, exclude_sources}`. */
function askScopeBody() {
  const hidden = [...state.hiddenFiles];
  if (state.askScope !== ASK_SCOPE_FILE || !state.currentFile) {
    return { exclude_sources: hidden };
  }
  return { include_sources: [state.currentFile.path], exclude_sources: hidden };
}

function renderViewer() {
  destroyViewer();
  dom.pdfControls.hidden = true;
  clear(dom.viewer);
  const corpus = activeCorpus();

  if (!corpus) {
    dom.viewerTitle.textContent = "No document open";
    dom.viewer.append(
      emptyState(
        "Add your first corpus",
        "raggy indexes folders you already have. Pick a folder, name the corpus, and its files appear in the list on the left.",
        "Add corpus",
        () => openModal()
      )
    );
    return;
  }

  if (state.currentFile) {
    dom.viewerTitle.textContent = state.currentFile.title;
    return; // the viewer component already owns the pane's contents
  }

  if (state.errorCard) {
    dom.viewerTitle.textContent = state.errorCard.title;
    dom.viewer.append(state.errorCard.node);
    return;
  }

  dom.viewerTitle.textContent = "No document open";
  if (state.lastReport) {
    dom.viewer.append(refreshReportView(state.lastReport));
    return;
  }
  if (!state.indexed) {
    dom.viewer.append(
      emptyState(
        `${corpus.name} is not indexed yet`,
        "Press Refresh to read this corpus's folders, embed the files, and fill the file list.",
        "Refresh now",
        () => refreshCorpus({ showSpinner: true })
      )
    );
    return;
  }
  dom.viewer.append(
    emptyState(
      "Nothing open",
      "Pick a file from the list, or ask a question and click a citation to jump to the passage it came from.",
      null,
      null
    )
  );
}

function emptyState(title, body, actionLabel, onAction) {
  const node = el("div", { class: "empty-state" }, [
    el("h2", { text: title }),
    el("p", { text: body }),
  ]);
  if (actionLabel) node.append(el("button", { type: "button", class: "btn btn-primary", text: actionLabel, onclick: onAction }));
  return node;
}

function showFatal(message) {
  document.body.append(
    el("div", { class: "error-box", text: message })
  );
}

// -- corpora -------------------------------------------------------------

async function switchCorpus(corpusId) {
  if (!corpusId || corpusId === state.activeId) return;
  try {
    const payload = await api.activateCorpus(corpusId);
    state.activeId = payload.active;
    // The previous corpus's report and open document belong to it, not here.
    // Its file list is left in place until the new one arrives — `loadDocuments`
    // clears it — so the pane shows "loading" rather than an empty list it has
    // not checked yet.
    state.lastReport = null;
    state.currentFile = null;
    state.hiddenFiles = new Set();
    await Promise.all([loadCorpora(), loadDocuments(), checkStatusOnce()]);
    renderDashboard();
  } catch (error) {
    note(`Could not switch corpus: ${errorText(error)}`, true);
    renderToolbar();
  }
}

/**
 * What the file pane says while the file list is on its way back.
 *
 * Three dimmed bars where the rows will be, and a line naming the corpus: the
 * point is that the pane is visibly *busy*, so the wait reads as a wait rather
 * than as the answer. A refresh reports its own progress, so this is a first
 * load (or a corpus switch) and there is nothing else to say yet.
 */
function fileListLoading() {
  const corpus = activeCorpus();
  return el("div", { class: "file-loading", role: "status", "aria-live": "polite" }, [
    el("div", { class: "file-skeleton", "aria-hidden": "true" }, [
      el("div", { class: "file-skeleton-row" }),
      el("div", { class: "file-skeleton-row" }),
      el("div", { class: "file-skeleton-row" }),
    ]),
    el("div", { class: "file-loading-text" }, [
      el("span", { class: "spinner spinner-inline", "aria-hidden": "true" }),
      el("span", {
        text: corpus
          ? `Reading ${corpus.name}'s index…`
          : "Reading the corpus's index…",
      }),
    ]),
  ]);
}

/**
 * The honest empty states, which only apply once a listing has actually arrived.
 *
 * An unindexed corpus and an indexed-but-empty one are different problems with
 * different fixes, so they say different things; neither may be shown while the
 * list is still loading (`documentsLoading` above).
 */
function emptyListMessage(corpus) {
  if (!state.documentsLoaded) {
    // No listing came back at all. The reason is in the toolbar note; claiming
    // anything about the files here would be a guess.
    return "The file list could not be read. Press Refresh to try again.";
  }
  if (!corpus.indexed) return "This corpus has not been indexed yet. Press Refresh.";
  return "No files to show. Add files to the source folder, then press Refresh.";
}

/** Ask the server for the active corpus's files, and remember which corpus. */
async function loadDocuments() {
  const corpusId = state.activeId;
  if (!corpusId) {
    state.documents = [];
    state.documentCounts = null;
    state.indexed = null;
    state.hiddenFiles = new Set();
    state.documentsLoading = false;
    state.documentsLoaded = false;
    state.documentsFor = null;
    return;
  }

  state.documentsLoading = true;
  if (state.documentsFor !== corpusId) {
    // A corpus's own files must not be shown as the new one's, and the rows of
    // the corpus being left are not a preview of the one arriving.
    state.documents = [];
    state.documentCounts = null;
    state.hiddenFiles = new Set();
    state.documentsLoaded = false;
  }
  // Show the loading state now. Nothing else repaints between here and the
  // response, so without this the pane keeps whatever it said before the
  // request — the empty-corpus message this state exists to replace.
  renderFilePanel();

  try {
    const payload = await api.documents(corpusId);
    if (state.activeId !== corpusId) return; // a switch overtook this request
    state.documents = payload.documents || [];
    state.documentCounts = payload.counts || null;
    state.indexed = Boolean(payload.indexed);
    state.documentsFor = corpusId;
    // After the list, never before: restoring the hidden set means dropping the
    // paths this corpus no longer has, which needs the list to check against.
    loadHiddenFiles();
  } catch (error) {
    if (state.activeId !== corpusId) return;
    state.documents = [];
    state.documentCounts = null;
    state.hiddenFiles = new Set();
    note(`Could not list files: ${errorText(error)}`, true);
  } finally {
    if (state.activeId === corpusId) {
      state.documentsLoading = false;
      state.documentsLoaded = true;
      // The caller renders the dashboard when it is done with this; the panel is
      // redrawn here as well so a caller that does not (a corpus switch that
      // failed after activation) still leaves the pane telling the truth.
      renderFilePanel();
    }
  }
}

/** Everything that depends on the file list: the list itself and its counts. */
function renderFilePanel() {
  renderFiles();
  renderAskScope();
}

async function checkStatusOnce() {
  if (!state.activeId) return null;
  try {
    const status = await api.status(state.activeId);
    if (status.busy && !state.refreshing && status.message) {
      dom.queryState.textContent = status.message;
    }
    return status;
  } catch {
    // Status is a progress nicety; its failure must not block the UI.
    return null;
  }
}

// -- refresh + progress --------------------------------------------------

async function refreshCorpus({ showSpinner = false } = {}) {
  const corpusId = state.activeId;
  if (!corpusId || state.refreshing) return;

  state.refreshing = true;
  state.refreshStart = Date.now();
  state.errorCard = null;
  renderToolbar();
  if (showSpinner) {
    dom.refreshOverlay.hidden = false;
    dom.refreshMessage.textContent = "starting…";
    dom.refreshElapsed.textContent = "0s elapsed";
  }
  startRefreshTimers();

  try {
    const payload = await api.refresh(corpusId);
    state.lastReport = payload.report || null;
    note("");
    await Promise.all([loadCorpora(), loadDocuments()]);
  } catch (error) {
    // The previous report stays on screen: it still describes the last run that
    // actually finished, and blanking it would lose the only record of it.
    note(`Refresh failed: ${errorText(error)}`, true);
    state.errorCard = { title: "Refresh failed", node: errorBox(error) };
  } finally {
    stopRefreshTimers();
    state.refreshing = false;
    // Always, on every path: a spinner must never outlive the request.
    dom.refreshOverlay.hidden = true;
    renderDashboard();
  }
}

function startRefreshTimers() {
  const corpusId = state.activeId;
  state.refreshElapsedTimer = window.setInterval(() => {
    const seconds = (Date.now() - state.refreshStart) / 1000;
    dom.refreshElapsed.textContent = `${seconds.toFixed(0)}s elapsed`;
  }, 250);

  state.statusTimer = window.setInterval(async () => {
    try {
      const status = await api.status(corpusId);
      if (status.message) dom.refreshMessage.textContent = status.message;
      if (typeof status.elapsed_seconds === "number" && status.elapsed_seconds > 0) {
        // The server's own clock is the better number when it has one.
        dom.refreshElapsed.textContent = `corpus busy for ${status.elapsed_seconds.toFixed(0)}s`;
      }
    } catch (error) {
      // A failed poll is not a failed refresh: the blocking request is the
      // source of truth, so keep waiting and just say so.
      dom.refreshMessage.textContent = `waiting for the server… (${error.message})`;
    }
  }, 1000);
}

function stopRefreshTimers() {
  window.clearInterval(state.statusTimer);
  window.clearInterval(state.refreshElapsedTimer);
  state.statusTimer = null;
  state.refreshElapsedTimer = null;
}

/** Render the report: counts first, per-file detail behind expanders. */
function refreshReportView(report) {
  const counts = report.counts || {};
  const wrapper = el("div", { class: "report" });
  wrapper.append(
    el("div", { class: "note-line", text: report.summary || "Refresh finished." })
  );

  const grid = el("div", { class: "count-grid" });
  const tiles = [
    ["indexed", counts.indexed],
    ["unchanged", counts.unchanged],
    ["failed", counts.failed],
    ["skipped", counts.skipped],
    ["removed", counts.removed],
    ["orphan chunks", counts.orphans],
    ["chunks", counts.chunks],
  ];
  for (const [label, value] of tiles) {
    grid.append(
      el("div", { class: `count-tile${label === "failed" && value ? " is-bad" : ""}` }, [
        el("span", { class: "count-value", text: String(value ?? 0) }),
        el("span", { class: "count-label", text: label }),
      ])
    );
  }
  wrapper.append(grid);

  const indexed = report.indexed || [];
  wrapper.append(
    detailSection("Indexed now", indexed.length, true, [
      el("ul", { class: "detail-list" }, indexed.map((entry) =>
        el("li", {}, [
          el("span", { class: "detail-name", text: basename(entry.path), title: entry.path }),
          el("span", { class: "detail-note", text: `${entry.chunks ?? 0} chunks` }),
        ])
      )),
    ])
  );

  const unchanged = report.unchanged || [];
  if (unchanged.length) {
    wrapper.append(
      detailSection("Unchanged", unchanged.length, false, [
        el("ul", { class: "detail-list" }, unchanged.map((path) =>
          el("li", {}, [el("span", { class: "detail-name", text: basename(path), title: String(path) })])
        )),
      ])
    );
  }

  const failed = report.failed || [];
  if (failed.length) {
    // Open by default: a failed file is the one thing in a report a user has to
    // act on.
    wrapper.append(
      detailSection("Failed", failed.length, true, [
        el("ul", { class: "detail-list" }, failed.map((entry) =>
          el("li", {}, [
            el("span", { class: "detail-name", text: basename(entry.path), title: entry.path }),
            el("span", { class: "detail-error", text: entry.error || "unknown error" }),
          ])
        )),
      ])
    );
  }

  const skipped = report.skipped || [];
  if (skipped.length) {
    wrapper.append(
      detailSection("Skipped", skipped.length, failed.length === 0, [
        el("ul", { class: "detail-list" }, skipped.map((entry) =>
          el("li", {}, [
            el("span", { class: "detail-name", text: basename(entry.path), title: entry.path }),
            el("span", {
              class: "detail-note",
              text: [entry.reason, entry.detail].filter(Boolean).join(": "),
            }),
          ])
        )),
      ])
    );
  }

  for (const [title, list] of [
    ["Removed", report.removed],
    ["Orphans (in the index, no file on disk)", report.orphans],
  ]) {
    if (!list?.length) continue;
    wrapper.append(
      detailSection(title, list.length, false, [
        el("ul", { class: "detail-list" }, list.map((path) =>
          el("li", {}, [el("span", { class: "detail-name", text: basename(path), title: String(path) })])
        )),
      ])
    );
  }

  return wrapper;
}

function detailSection(title, count, open, children) {
  const details = el("details", { class: "detail-section", open });
  details.append(el("summary", { text: `${title} (${count})` }));
  for (const child of children) details.append(child);
  return details;
}

function errorBox(error) {
  const box = el("div", { class: "error-box" });
  box.append(el("div", { text: error?.message || String(error) }));
  if (error?.hint) box.append(el("div", { class: "error-hint", text: error.hint }));
  return box;
}

// -- viewers -------------------------------------------------------------

function destroyViewer() {
  state.viewerRequest += 1; // invalidates any in-flight open
  state.viewer?.destroy?.();
  state.viewer = null;
  state.viewerKind = null;
  window.raggyViewer = null;
}

async function loadPdfModule() {
  const module = await import("./js/pdfview.js");
  module.globalOptions.cMapUrl = "/vendor/pdfjs/cmaps/";
  module.globalOptions.cMapPacked = true;
  module.globalOptions.standardFontDataUrl = "/vendor/pdfjs/standard_fonts/";
  return module;
}

/**
 * Open (or re-open) the main pane on `spec`.
 *
 * `spec.viewerKind` is the API's viewer vocabulary (`pdf`, `text`, `html`,
 * `image`); `spec.fileKind` is the `/documents` vocabulary, which additionally
 * distinguishes `document`/`presentation` (converted to a PDF at index time)
 * from `other` (no viewer at all — the link is the honest answer there).
 */
async function openInViewer(spec) {
  const request = ++state.viewerRequest;
  destroyViewer();
  state.viewerRequest = request;
  clear(dom.viewer);
  dom.pdfControls.hidden = true;
  dom.viewerStatus.hidden = true;
  // `path` is what ties the open document back to the file list, which is what
  // the ask pane needs to name the file it would scope a question to. The file
  // list passes it in `path`, a citation passes it in `source`.
  state.currentFile = { ...spec, path: spec.path || spec.source || null };
  state.viewerKind = spec.viewerKind;
  dom.viewerTitle.textContent = spec.title;
  // The PDF viewer never clears its container, so its note (a converted
  // DOCX/PPTX caption) can live above the host element. The other viewers do
  // clear it on open, so they render the note themselves — see textview.js.
  if (spec.note && spec.viewerKind === "pdf") {
    dom.viewer.append(el("div", { class: "note-line", text: spec.note }));
  }
  renderFiles();
  try {
    window.localStorage.setItem(MODE_KEY, spec.viewerKind);
  } catch {
    // Private-mode browsers can refuse storage; the mode hint is not important.
  }

  if (spec.viewerKind === "pdf") {
    let module;
    try {
      module = await loadPdfModule();
    } catch (error) {
      if (request === state.viewerRequest) {
        dom.viewer.append(errorBox(new Error(`the PDF viewer is unavailable: ${error.message}`)));
      }
      return;
    }
    if (request !== state.viewerRequest) return;
    const host = el("div", { class: "pdf-host" });
    dom.viewer.append(host);
    const viewer = new module.PdfViewer(host);
    viewer.onPageChange = (page, total) => {
      dom.pdfControls.hidden = false;
      dom.pdfPageInfo.textContent = `${page} / ${total}`;
    };
    state.viewer = viewer;
    window.raggyViewer = viewer;
    await viewer.open({ url: spec.url, page: spec.page || 1, search: spec.search || "" });
    return;
  }

  let viewer;
  if (spec.viewerKind === "text") {
    viewer = new TextViewer(dom.viewer);
  } else if (spec.viewerKind === "html") {
    viewer = new HtmlViewer(dom.viewer);
  } else if (spec.viewerKind === "image") {
    viewer = new ImageViewer(dom.viewer);
  } else {
    dom.viewer.append(
      errorBox(new Error("There is no viewer for this file type. Open it from the file list after indexing it."))
    );
    return;
  }
  state.viewer = viewer;
  window.raggyViewer = viewer;
  await viewer.open(spec);
  if (request !== state.viewerRequest) return;
  if (spec.viewerKind === "text") {
    if (spec.search) viewer.jump(spec.search);
  } else if (spec.viewerKind === "html") {
    if (spec.search) viewer.jump(spec.search);
  }
}

function openFile(document) {
  const viewerKind = viewerKindFor(document.kind);
  // The ask pane is rendered after the viewer, not before: what it offers
  // depends on which file is open, and a hidden file can be read but must not be
  // offered as the one thing a question is asked of.
  return openInViewer({
    title: document.name,
    path: document.path,
    url: document.content_url,
    viewerKind,
    fileKind: document.kind,
    search: "",
    page: 1,
    // An indexed image's text has no chunk to travel in, so the file list sends
    // it along; without it the OCR panel would have nothing to show.
    ocrText: document.ocr_text || "",
    note: imageNote(viewerKind, document),
  }).finally(renderAskScope);
}

/**
 * The one-line explanation a viewer needs when jump-to-citation cannot apply.
 *
 * Images are the plan's feature 4 option (a): the OCR text is shown beside the
 * image instead of a highlight, and saying so is the point of the note. An image
 * the corpus could read nothing out of says *that* instead — and distinguishes
 * "not indexed yet" from "indexed, nothing readable in it", because only the
 * first is something the user can act on.
 */
function imageNote(viewerKind, document) {
  if (viewerKind !== "image") return "";
  if (!document) return "";
  if (document.status === "new") {
    return "Not indexed yet: press Refresh to read this image. An image with no text in it is then hidden from the list.";
  }
  if (!document.chunks && !document.ocr_text) {
    return "No text could be read from this image, so it is not part of any answer and the list hides it behind “No text”.";
  }
  return "Images have no text layer to search, so this image's OCR text is shown beside it instead of a highlight.";
}

function viewerKindFor(kind) {
  if (kind === "pdf" || kind === "document" || kind === "presentation") return "pdf";
  if (kind === "text") return "text";
  if (kind === "web") return "html";
  if (kind === "image") return "image";
  return "none";
}

function openCitation(citation) {
  const target = citation.target;
  if (!target) return;
  const viewerKind = target.kind === "office" ? "pdf" : target.kind;
  return openInViewer({
    title: target.name || basename(target.source || ""),
    path: target.source || null,
    url: target.content_url,
    viewerKind,
    fileKind: viewerKind,
    search: target.search || "",
    page: target.page || 1,
    startLine: target.start_line,
    ocrText: target.ocr_text || "",
    note: target.note || "",
  });
}

// -- chat ----------------------------------------------------------------

async function ask(rawQuery) {
  const query = String(rawQuery || "").trim();
  if (!query) return;
  if (state.refreshing) {
    dom.queryState.textContent = "waiting for the refresh to finish…";
    return;
  }
  if (!state.activeId) {
    appendError(new Error("Add a corpus before asking a question."));
    return;
  }

  dom.queryInput.value = "";
  dom.chatLog.append(el("div", { class: "msg" }, [
    el("div", { class: "msg-role", text: "You" }),
    el("div", { class: "msg-question", text: query }),
  ]));

  const answerSlot = el("div", { class: "msg" }, [
    el("div", { class: "msg-role", text: "raggy" }),
    el("div", { class: "retrieving" }, [
      el("span", { class: "spinner spinner-inline", "aria-hidden": "true" }),
      el("span", { class: "blink", text: "Retrieving…" }),
    ]),
  ]);
  dom.chatLog.append(answerSlot);
  scrollChat();
  setQueryBusy(true);

  try {
    const payload = await api.query(query, state.activeId, askScopeBody());
    answerSlot.replaceChildren(
      el("div", { class: "msg-role", text: "raggy" }),
      answerBlock(payload.answer || ""),
      citationList(payload.citations || [])
    );
  } catch (error) {
    // The spinner is inside answerSlot, so replacing it on both paths is what
    // guarantees no spinner is left running.
    answerSlot.replaceChildren(el("div", { class: "msg-role", text: "raggy" }), errorBox(error));
  } finally {
    setQueryBusy(false);
    dom.queryInput.focus();
    scrollChat();
  }
}

function answerBlock(markdown) {
  const node = el("div", { class: "msg-answer" });
  // renderMarkdown escapes every character of the answer before adding any tag
  // of its own, so this innerHTML assignment cannot introduce model-controlled
  // markup.
  node.innerHTML = renderMarkdown(markdown);
  return node;
}

function citationList(citations) {
  if (!citations.length) {
    return el("div", { class: "citations" }, [
      el("div", { class: "citations-title", text: "Citations" }),
      el("div", { class: "browse-empty", text: "No sources were retrieved for this answer." }),
    ]);
  }
  const wrapper = el("div", { class: "citations" }, [
    el("div", { class: "citations-title", text: `Citations (${citations.length})` }),
  ]);
  for (const citation of citations) wrapper.append(citationCard(citation));
  return wrapper;
}

function citationCard(citation) {
  const target = citation.target || null;
  const clickable = Boolean(target) && target.highlightable !== false;
  const card = clickable
    ? el("button", { type: "button", class: "citation", onclick: () => openCitation(citation) })
    : el("div", { class: `citation citation-static${target ? " is-nojump" : ""}` });

  card.append(
    el("div", { class: "citation-head" }, [
      el("span", { class: "citation-label", text: citation.label || basename(target?.source || "") || "source" }),
      citation.score === null || citation.score === undefined
        ? null
        : el("span", { class: "citation-score", text: formatScore(citation.score) }),
    ]),
    el("span", { class: "citation-snippet", text: citation.snippet || "" })
  );

  if (!target) {
    card.append(el("span", { class: "citation-note", text: "No source file could be resolved for this passage." }));
  } else if (!clickable) {
    card.append(el("span", { class: "citation-note", text: "This source has no text layer to jump into." }));
  }
  return card;
}

function setQueryBusy(busy) {
  dom.sendBtn.disabled = busy;
  dom.sendBtn.textContent = busy ? "Retrieving…" : "Send";
  dom.queryState.textContent = busy ? "Retrieving…" : "";
  dom.queryState.classList.toggle("blink", busy);
}

function appendError(error) {
  dom.chatLog.append(el("div", { class: "msg" }, [errorBox(error)]));
  scrollChat();
}

function scrollChat() {
  dom.chatLog.scrollTop = dom.chatLog.scrollHeight;
}

// -- delete corpus modal -------------------------------------------------

/**
 * Ask before removing the active corpus.
 *
 * Deleting is the one action here that cannot be undone from the UI, so it is
 * confirmed rather than announced, and the dialog names the corpus it is about:
 * "delete this corpus?" is not a question a user can answer if they have several.
 */
function openDeleteModal() {
  const corpus = activeCorpus();
  if (!corpus) return;
  dom.deleteError.hidden = true;
  dom.deleteWarning.replaceChildren(
    el("span", { text: "Remove " }),
    el("strong", { class: "modal-strong", text: corpus.name }),
    el("span", {
      text: corpus.indexed
        ? " and its index? This cannot be undone."
        : "? It has not been indexed yet, so there is nothing to re-read. This cannot be undone.",
    })
  );
  dom.deleteModal.hidden = false;
  // The safe choice holds the focus, so Enter does not delete anything.
  dom.deleteCancel.focus?.();
}

function closeDeleteModal() {
  dom.deleteModal.hidden = true;
}

async function deleteActiveCorpus() {
  const corpus = activeCorpus();
  if (!corpus) {
    closeDeleteModal();
    return;
  }

  dom.deleteConfirm.disabled = true;
  dom.deleteConfirm.textContent = "Deleting…";
  dom.deleteError.hidden = true;
  try {
    // deleteDb is the server's default and what this offers: the corpus's own
    // DB directory is raggy's copy of the files, and leaving it behind would
    // orphan an index nothing can reach. The source files are never touched.
    const payload = await api.deleteCorpus(corpus.id, true);
    closeDeleteModal();
    // Everything on screen belonged to the corpus that just went away.
    state.activeId = payload.active || null;
    state.lastReport = null;
    state.currentFile = null;
    state.documents = [];
    state.documentCounts = null;
    state.hiddenFiles = new Set();
    state.documentsFor = null;
    state.documentsLoaded = false;
    await loadCorpora();
    if (state.activeId) await Promise.all([loadDocuments(), checkStatusOnce()]);
    renderDashboard();
    note(`Deleted "${corpus.name}".`);
  } catch (error) {
    dom.deleteError.hidden = false;
    clear(dom.deleteError);
    dom.deleteError.append(el("div", { text: error?.message || String(error) }));
    if (error?.hint) dom.deleteError.append(el("div", { class: "error-hint", text: error.hint }));
  } finally {
    dom.deleteConfirm.disabled = false;
    dom.deleteConfirm.textContent = "Delete";
  }
}

// -- add corpus modal ----------------------------------------------------

function openModal() {
  dom.modal.hidden = false;
  dom.modalError.hidden = true;
  state.selection = [];
  state.corpusNameEdited = false;
  dom.corpusName.value = "";
  renderSelection();
  renderCorpusSources();
  browse(null);
}

function closeModal() {
  dom.modal.hidden = true;
}

/**
 * Offer the folder being browsed as the corpus's name, unless the user has
 * named it themselves.
 *
 * Following the browse rather than being set once means the default keeps up
 * with the choice: someone who navigates into `D:\\papers\\2024` gets "2024",
 * not the home directory they landed in first.
 */
function suggestCorpusName(folderPath) {
  if (state.corpusNameEdited || !folderPath) return;
  dom.corpusName.value = basename(folderPath);
}

async function browse(path, isFallback = false) {
  try {
    const payload = await api.browse(path);
    state.browsePath = payload.path;
    // The server computes the parent (and nulls it at a drive root), so path
    // arithmetic stays in one place instead of being re-derived per platform.
    state.browseParent = payload.parent;
    state.browseHomePath = payload.home;
    renderBrowse(payload);
    dom.browseUp.disabled = !payload.parent;
    suggestCorpusName(payload.path);
    renderCorpusSources();
  } catch (error) {
    if (!path && !isFallback) {
      // The dialog opens on the home directory, which the server resolves
      // itself — and a home directory that does not exist is an error the user
      // cannot act on. Opening somewhere valid instead leaves them with a dialog
      // they can use: a name field with a suggestion in it, and folders to click.
      await browse(await filesystemRoot(), true);
      return;
    }
    showModalError(error);
  }
}

/** Somewhere that exists, for a browse that cannot open the home directory. */
async function filesystemRoot() {
  // Walk up to a drive or filesystem root by following the parents the server
  // reports, so no path arithmetic happens here ("" is the server's word for
  // "this is a root"). Starting points in order of preference, because the browse
  // that failed was for the home directory and the server never told us where
  // that was: the parent of what it did tell us, then the server's working
  // directory, then each parent of that in turn.
  const starts = [state.browsePath, "..", ".", "/"].filter(Boolean);
  for (const start of starts) {
    let current = start;
    for (let step = 0; step < 32 && current; step += 1) {
      let payload;
      try {
        payload = await api.browse(current);
      } catch {
        break; // this starting point is not usable; try the next
      }
      if (!payload.parent) return payload.path;
      current = payload.parent;
    }
  }
  return "/";
}

function renderBrowse(payload) {
  dom.browsePath.textContent = payload.path;
  clear(dom.browseList);
  const directories = payload.entries.filter((entry) => entry.is_dir);
  const files = payload.entries.filter((entry) => !entry.is_dir);

  if (!payload.entries.length) {
    dom.browseList.append(el("li", { class: "browse-empty", text: "This folder is empty." }));
  }

  for (const entry of directories) {
    const checkbox = el("input", {
      type: "checkbox",
      dataset: { path: entry.path },
      checked: state.selection.some((item) => item.path === entry.path),
      onchange: () => toggleSelection(entry.path, entry.name),
    });
    const nameNode = el("span", { class: "browse-name is-dir", text: `${entry.name}/` });
    // The name navigates, the checkbox selects. Merging the two would make it
    // impossible to select a folder and still look inside it.
    nameNode.addEventListener("click", (event) => {
      event.preventDefault();
      browse(entry.path);
    });
    dom.browseList.append(
      el("li", { class: "browse-row" }, [el("label", {}, [checkbox, nameNode])])
    );
  }
  for (const entry of files) {
    dom.browseList.append(
      el("li", { class: "browse-row" }, [
        el("span", {
          class: "browse-name browse-file",
          text: entry.name,
          title: "Files are indexed through their folder",
        }),
      ])
    );
  }
}

function toggleSelection(path, name) {
  const index = state.selection.findIndex((item) => item.path === path);
  if (index === -1) state.selection.push({ path, name });
  else state.selection.splice(index, 1);
  renderSelection();
  renderBrowseChecks();
}

/**
 * The folders "Add corpus" will index.
 *
 * The ticked ones; and if nothing is ticked, the folder being browsed — a user
 * who navigated to the folder they want has already chosen it, and refusing
 * because they did not also tick it is a worse answer than using it.
 */
function effectiveSources() {
  const chosen = state.selection.map((item) => item.path);
  if (chosen.length) return chosen;
  return state.browsePath ? [state.browsePath] : [];
}

function renderCorpusSources() {
  const sources = effectiveSources();
  if (!sources.length) {
    dom.corpusSources.textContent = "Choose a folder to index.";
    return;
  }
  const names = sources.map((path) => basename(path)).join(", ");
  dom.corpusSources.textContent =
    sources.length === state.selection.length
      ? `Will index: ${names}`
      : `Will index this folder: ${names}`;
}

function renderSelection() {
  clear(dom.browseSelection);
  for (const item of state.selection) {
    dom.browseSelection.append(
      el("span", { class: "chip" }, [
        el("span", { text: item.name }),
        el("button", {
          type: "button",
          class: "chip-remove",
          text: "×",
          title: `Remove ${item.name}`,
          onclick: () => toggleSelection(item.path, item.name),
        }),
      ])
    );
  }
  // Which folders will be indexed changes with the selection, and with the
  // folder being browsed when nothing is selected.
  renderCorpusSources();
}

/** Keep the visible checkboxes in step with the selection set, not the DOM text. */
function renderBrowseChecks() {
  for (const checkbox of dom.browseList.querySelectorAll('input[type="checkbox"]')) {
    checkbox.checked = state.selection.some((item) => item.path === checkbox.dataset.path);
  }
}

async function createCorpus() {
  const name = dom.corpusName.value.trim();
  const sources = effectiveSources();
  dom.modalError.hidden = true;

  if (!name) {
    showModalError(new Error("Give the corpus a name."));
    return;
  }
  if (!sources.length) {
    showModalError(new Error("Pick at least one folder to index."));
    return;
  }

  dom.browseCreate.disabled = true;
  try {
    const payload = await api.createCorpus(name, sources);
    state.activeId = payload.corpus?.id || state.activeId;
    closeModal();
    await loadCorpora();
    await Promise.all([loadDocuments(), checkStatusOnce()]);
    renderDashboard();
    note(`Added "${payload.corpus?.name || name}". Press Refresh to index it.`);
  } catch (error) {
    showModalError(error);
  } finally {
    dom.browseCreate.disabled = false;
  }
}

function showModalError(error) {
  dom.modalError.hidden = false;
  clear(dom.modalError);
  // Verbatim from the server, including the hint when it sent one.
  dom.modalError.append(el("div", { text: error?.message || String(error) }));
  if (error?.hint) dom.modalError.append(el("div", { class: "error-hint", text: error.hint }));
}

function errorText(error) {
  return error?.message || String(error);
}
