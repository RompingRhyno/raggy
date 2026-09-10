// pdf.js-based document viewer: one canvas + one text layer per page.
//
// The hard part is jump-to-citation (build plan feature 4): the API hands us a
// 1-based page number and the chunk's first substantial line, and we have to put
// that line back on the page. Because pdf.js gives us a real text layer, the
// highlight is a positioned <mark> inside it rather than a bounding-box overlay
// computed at index time — so it works for any PDF, including converted
// DOCX/PPTX renders, with no extra metadata.
//
// Imported dynamically by app.js so that a missing/corrupt vendored pdf.js
// degrades to an error message inside the viewer instead of breaking the whole
// app at module-evaluation time.

import {
  getDocument,
  GlobalWorkerOptions,
  setLayerDimensions,
  TextLayer,
} from "/vendor/pdfjs/pdf.min.mjs";

GlobalWorkerOptions.workerSrc = "/vendor/pdfjs/pdf.worker.min.mjs";

/**
 * Options merged into every `getDocument` call. app.js sets `cMapUrl`,
 * `cMapPacked` and `standardFontDataUrl` here at startup; keeping them in one
 * place means the viewer itself has no opinion about where the vendored assets
 * live.
 */
export const globalOptions = {};

const MAX_CONCURRENT_RENDERS = 2;
const PREFETCH_PAGES = 2;
// Pages outside this window release their canvas; a 500-page PDF must not grow
// the tab's memory without bound.
const KEEP_RENDERED_PAGES = 6;

export class PdfViewer {
  constructor(container) {
    this.container = container;
    this.doc = null;
    this.url = null;
    this.numPages = 0;
    this.currentPage = 1;
    this.states = new Map();
    this.search = "";
    this.currentMarkIndex = 0;
    this.onProgress = null;
    this.onPageChange = null;
    this._token = 0;
    this._destroyed = false;
    this._loading = null;
    this._rendering = new Set();
    this._surface = null;
    this._observer = null;
    // The page a jump asked for and has not yet seen on screen; see
    // `_settlePendingPage`.
    this._pendingPage = null;
  }

  /**
   * Show `url` at `page` (1-based) and highlight `search` there.
   * Re-opening the same URL keeps the loaded document and its rendered pages.
   */
  async open({ url, page = 1, search = "" } = {}) {
    if (this._destroyed) return;
    if (!url) {
      this._error("The citation did not include a file URL.");
      return;
    }

    // Claim a token before anything async happens: a second open() (or a
    // destroy()) invalidates this one's remaining work.
    const token = ++this._token;

    if (url !== this.url || !this.doc) {
      this._clearAll();
      this.url = url;
      const loading = getDocument({ url, ...globalOptions });
      this._loading = loading;
      try {
        const doc = await loading.promise;
        if (this._destroyed || token !== this._token) {
          doc.destroy();
          return;
        }
        this.doc = doc;
        this.numPages = doc.numPages;
        // Page 1's geometry is fetched up front: every placeholder takes the
        // real page box, which keeps the scrollbar honest and makes a jump to
        // page 300 land in the right place before page 300 has been rendered.
        try {
          const first = await doc.getPage(1);
          this._baseViewport = first.getViewport({ scale: 1 });
        } catch {
          this._baseViewport = null;
        }
        if (this._destroyed || token !== this._token) return;
        this._scale = chooseScale(this._baseViewport?.width || 612, this.container.clientWidth);
        this._buildSurface();
      } catch (error) {
        if (this._destroyed || token !== this._token) return;
        this._error(describePdfError(error));
        return;
      } finally {
        if (this._loading === loading) this._loading = null;
      }
    }

    if (token !== this._token) return;

    const target = clamp(Number(page) || 1, 1, Math.max(this.numPages, 1));
    this.currentPage = target;
    this.search = String(search || "");
    this._clearHighlights();
    this._scrollToPage(target, "auto");
    this._renderAround(target);
    this.onPageChange?.(target, this.numPages);

    // _renderPage re-applies the highlight when it finishes rendering the target
    // page. When that page is already rendered — the same citation clicked
    // twice, or a new search on a document already on screen — nothing would
    // otherwise apply it, so do it here too. The viewport is left to
    // `_settlePendingPage`, which is the one place that decides where a jump
    // ends up: a second opinion here would only fight it.
    if (this.search && this.states.get(target)?.status === "rendered") {
      await this._applyHighlights(target);
      this.currentMarkIndex = 0;
      this._settlePendingPage();
    }
  }

  /** Jump to an already-loaded page. */
  goToPage(page) {
    if (!this.doc) return;
    const target = clamp(Number(page) || 1, 1, this.numPages);
    this.currentPage = target;
    this._scrollToPage(target, "smooth");
    this._renderAround(target);
    this.onPageChange?.(target, this.numPages);
  }

  /**
   * Highlight every occurrence of `search` on the current page.
   * Returns the number of matches.
   */
  async highlight(search) {
    if (search !== undefined) this.search = String(search || "");
    const count = await this._applyHighlights(this.currentPage);
    this.currentMarkIndex = 0;
    this._settlePendingPage();
    return count;
  }

  /** Move to the next match on the current page. Returns false when there is none. */
  nextMatch() {
    const state = this.states.get(this.currentPage);
    const marks = state?.marks || [];
    if (!marks.length) return false;
    this.currentMarkIndex = (this.currentMarkIndex + 1) % marks.length;
    this._focusMark(this.currentMarkIndex);
    return true;
  }

  /** Reset the viewer, aborting any in-flight load or render. */
  destroy() {
    this._destroyed = true;
    this._token += 1;
    this._loading?.destroy?.().catch(() => {});
    this._loading = null;
    this._clearAll();
    this.container.replaceChildren();
    this.doc = null;
    this.url = null;
  }

  // -- structure ---------------------------------------------------------

  _buildSurface() {
    this._disconnectObserver();
    this.container.replaceChildren();
    const surface = document.createElement("div");
    surface.className = "pdf-surface";
    this._surface = surface;
    this.container.append(surface);

    for (let pageNumber = 1; pageNumber <= this.numPages; pageNumber += 1) {
      const element = document.createElement("div");
      element.className = "pdf-page";
      element.dataset.page = String(pageNumber);
      // The placeholder is given the real page's aspect ratio so that scroll
      // position and the jump target are correct before anything renders.
      const size = this._pageSize();
      element.style.width = `${Math.round(size.width)}px`;
      element.style.height = `${Math.round(size.height)}px`;
      element.append(placeholder(`page ${pageNumber}`));
      surface.append(element);
      this.states.set(pageNumber, { element, canvas: null, textLayerDiv: null, textLayer: null, marks: [], status: "idle" });
    }

    this._observe(surface);
  }

  _pageSize() {
    const base = this._baseViewport;
    if (base) return { width: base.width, height: base.height };
    // Letter-ish fallback: a page with no known geometry still gets a stable
    // box, so the surface does not reflow once rendering starts.
    return { width: 612, height: 792 };
  }

  _observe(surface) {
    if (typeof IntersectionObserver === "undefined") {
      // No observer (very old browser): render on demand instead of never.
      this._renderAllSoon();
      return;
    }
    this._observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const pageNumber = Number(entry.target.dataset.page);
          if (entry.isIntersecting) this._queue(pageNumber);
          else this._releaseFarPages();
        }
      },
      { root: this.container, rootMargin: "600px 0px" }
    );
    for (const state of this.states.values()) this._observer.observe(state.element);
  }

  _disconnectObserver() {
    this._observer?.disconnect();
    this._observer = null;
  }

  // -- loading & rendering ----------------------------------------------

  _renderAllSoon() {
    for (const pageNumber of this.states.keys()) this._queue(pageNumber);
  }

  /** Render the target page (and its neighbours) without waiting for scroll. */
  _renderAround(pageNumber) {
    this._queue(pageNumber);
    for (let offset = 1; offset <= PREFETCH_PAGES; offset += 1) {
      this._queue(pageNumber + offset);
      this._queue(pageNumber - offset);
    }
  }

  _queue(pageNumber) {
    const state = this.states.get(pageNumber);
    if (!state || state.status !== "idle") return;
    state.status = "queued";
    this._pump();
  }

  _pump() {
    while (this._rendering.size < MAX_CONCURRENT_RENDERS) {
      const next = [...this.states.entries()].find(([, state]) => state.status === "queued");
      if (!next) return;
      const [pageNumber, state] = next;
      state.status = "rendering";
      this._rendering.add(pageNumber);
      this._renderPage(pageNumber, state)
        .catch((error) => this._pageError(pageNumber, error))
        .finally(() => {
          this._rendering.delete(pageNumber);
          this._pump();
        });
    }
  }

  async _renderPage(pageNumber, state) {
    const token = this._token;
    const page = await this.doc.getPage(pageNumber);
    if (this._destroyed || token !== this._token) return;

    const scale = this._scale || 1.2;
    const viewport = page.getViewport({ scale });
    const devicePixelRatio = Math.min(window.devicePixelRatio || 1, 2);

    state.element.replaceChildren();
    state.element.style.width = `${Math.round(viewport.width)}px`;
    state.element.style.height = `${Math.round(viewport.height)}px`;

    const canvas = document.createElement("canvas");
    canvas.width = Math.floor(viewport.width * devicePixelRatio);
    canvas.height = Math.floor(viewport.height * devicePixelRatio);
    canvas.style.width = `${Math.floor(viewport.width)}px`;
    canvas.style.height = `${Math.floor(viewport.height)}px`;
    const context = canvas.getContext("2d", { alpha: false });
    state.element.append(canvas);
    state.canvas = canvas;

    const textLayerDiv = document.createElement("div");
    textLayerDiv.className = "textLayer";
    // The page box is measured in CSS pixels (the canvas is scaled down to them
    // by its own width/height style), so the scale the layer is factored by is
    // the viewport's, with no device-pixel correction. pdf.js reads this in the
    // font-size calc that sizes each text run; without it the runs are laid out
    // at the wrong size, and their selection and hit-testing drift from the
    // glyphs they are supposed to sit on.
    textLayerDiv.style.setProperty("--total-scale-factor", String(scale));
    state.element.append(textLayerDiv);
    state.textLayerDiv = textLayerDiv;
    // pdf.js's own helper for publishing that scale: it writes the width/height
    // from the viewport using the same variable.
    setLayerDimensions(textLayerDiv, viewport);

    await page.render({
      canvasContext: context,
      viewport,
      transform: devicePixelRatio === 1 ? null : [devicePixelRatio, 0, 0, devicePixelRatio, 0, 0],
    }).promise;
    if (this._destroyed || token !== this._token) return;

    try {
      const textLayer = new TextLayer({
        textContentSource: await page.getTextContent(),
        container: textLayerDiv,
        viewport,
      });
      state.textLayer = textLayer;
      const rendered = textLayer.render();
      await rendered?.promise;
    } catch (error) {
      // A page whose text layer fails still shows its image; only the
      // highlight is lost.
      console.warn("raggy: text layer failed for page", pageNumber, error);
    }
    if (this._destroyed || token !== this._token) return;

    state.status = "rendered";
    state.viewport = viewport;
    // A page's real box can differ from the placeholder it was built with (a
    // document is not obliged to keep one page size), which moves every page
    // below it — so a jump made before this render can end up pointing at the
    // wrong place. Position is re-confirmed here, once this page's geometry is
    // final, and again after the highlight below, which may create the very mark
    // the jump is aiming at.
    this._settlePendingPage();
    if (pageNumber === this.currentPage && this.search) await this._applyHighlights(pageNumber);
    this._settlePendingPage();
  }

  /**
   * Put the viewport where a jump asked for it, now that geometry is known.
   *
   * A jump is aimed at an *address*: a page, and usually a phrase on it. The page
   * is what can be scrolled to reliably, so it is the fallback; but when the
   * phrase has been found, the phrase is what the user asked to see, and it is
   * scrolled to instead. Without that, a citation near the bottom of a page lands
   * on the page and leaves its own highlight off screen — which is what happens
   * if the page is centred by itself after the highlight was placed.
   *
   * Cheap and idempotent: it measures, and returns immediately when the goal is
   * already met, so the rendered neighbour pages do not disturb a settled view.
   */
  _settlePendingPage() {
    const pageNumber = this._pendingPage;
    if (pageNumber === null || !this.doc) return;
    const state = this.states.get(pageNumber);
    if (!state?.element) {
      this._pendingPage = null;
      return;
    }
    const scroller = this.container.getBoundingClientRect();
    if (!scroller.height) return; // not laid out yet; try again on the next render

    const box = state.element.getBoundingClientRect();
    const overlap =
      Math.min(box.bottom, scroller.bottom) - Math.max(box.top, scroller.top);

    if (overlap <= 8) {
      // Not there at all: nothing else to aim at yet.
      this.container.scrollTo({
        top: Math.max(0, state.element.offsetTop - 8),
        behavior: "auto",
      });
      return;
    }

    // The phrase is found by an async pass over the page's text layer, so the
    // first settle after a render can arrive before the marks exist. Claiming
    // the page is settled then is what left a citation at the bottom of a page
    // off screen: the page had arrived, the highlight had not. So while the page
    // is still rendering, or a search is still unmatched on it, the jump stays
    // pending and this runs again once there is something to show.
    const waiting =
      state.status !== "rendered" || (this.search && state.marks.length === 0);
    if (waiting) return;

    const target = this.states.get(this.currentPage)?.marks[this.currentMarkIndex];
    if (target) {
      // The phrase is what the user asked to see, so it wins over the page.
      //
      // Done in one instant step, and only when the mark is not already in
      // view: this runs after every page render, and a smooth scroll restarted
      // by each of them never converges — it is cancelled and re-aimed from
      // wherever it had got to, which leaves the viewport short of the target.
      const rect = target.getBoundingClientRect();
      const inView = rect.top >= scroller.top + 8 && rect.bottom <= scroller.bottom - 8;
      if (!inView) {
        const absolute = rect.top - scroller.top + this.container.scrollTop;
        const centred = absolute - this.container.clientHeight / 2 + rect.height / 2;
        this.container.scrollTo({
          top: Math.min(
            Math.max(0, centred),
            Math.max(0, this.container.scrollHeight - this.container.clientHeight)
          ),
          behavior: "auto",
        });
      }
      this._pendingPage = null;
      return;
    }
    if (overlap >= Math.min(box.height, scroller.height) - 8) {
      this._pendingPage = null; // the whole page is visible and nothing is marked
    }
  }

  _pageError(pageNumber, error) {
    const state = this.states.get(pageNumber);
    if (!state || this._destroyed) return;
    state.status = "failed";
    state.element.replaceChildren(errorBox(describePdfError(error)));
    console.warn("raggy: page", pageNumber, "failed to render", error);
  }

  _releaseFarPages() {
    for (const [pageNumber, state] of this.states) {
      if (state.status !== "rendered") continue;
      if (Math.abs(pageNumber - this.currentPage) <= KEEP_RENDERED_PAGES) continue;
      state.canvas?.remove();
      state.canvas = null;
      state.textLayerDiv?.remove();
      state.textLayerDiv = null;
      state.textLayer = null;
      state.marks = [];
      state.status = "idle";
      state.element?.replaceChildren(placeholder(`page ${pageNumber}`));
    }
  }

  // -- highlighting ------------------------------------------------------

  /**
   * Highlight `this.search` on `pageNumber`.
   *
   * The text layer's spans hold the page's text. Their textContent is
   * concatenated (pdf.js places one span per text chunk and a newline-ish item
   * between lines), the search string is matched with whitespace treated as
   * flexible — PDF extraction inserts line breaks and odd spacing that the
   * chunk's own text does not have — and each match is then written back as
   * <mark> elements inside the spans it covers. Because the marks are children
   * of the text-layer spans, they inherit pdf.js's own positioning and are
   * aligned with the rendered text by construction.
   */
  async _applyHighlights(pageNumber) {
    const state = this.states.get(pageNumber);
    if (!state || !state.textLayerDiv) return 0;
    clearMarks(state.marks);
    state.marks = [];
    const needle = this.search.trim();
    if (!needle) return 0;

    // Matching only makes sense once the text layer exists. If the page was
    // scrolled away and released, queue it again: _renderPage re-applies the
    // highlight for the current page when it finishes.
    if (state.status !== "rendered") {
      this._queue(pageNumber);
      return 0;
    }

    const spans = [...state.textLayerDiv.querySelectorAll("span")];
    if (!spans.length) return 0;

    const { text, map } = concatSpans(spans);
    state.marks = markRanges(spans, map, matchFlexible(text, needle));
    return state.marks.length;
  }

  /**
   * Remove every mark from every page.
   *
   * The marks live *inside* the text layer's spans, so dropping a mark with
   * `remove()` would delete the highlighted text with it — and the citation
   * would then stop matching on the next search. Hoisting the mark's own text
   * nodes back into its parent is what makes clearing lossless.
   */
  _clearHighlights() {
    for (const state of this.states.values()) {
      clearMarks(state.marks);
      state.marks = [];
    }
  }

  /**
   * Centre the current mark on `index`, and report whether there was one.
   *
   * The return value is what lets a jump tell "this page has nothing marked on it
   * yet" from "the phrase was found and is now on screen".
   */
  _focusMark(index) {
    const marks = this.states.get(this.currentPage)?.marks || [];
    marks.forEach((mark, position) => mark.classList.toggle("is-current", position === index));
    const target = marks[index];
    if (!target) return false;

    // Measured as an absolute offset in the scroller, not as `offsetTop`.
    //
    // A mark lives inside a text-layer span, and that span is absolutely
    // positioned by pdf.js — so `offsetTop` is a couple of pixels within its own
    // span, not a position in the document. Scrolling to it lands at the top of
    // the document and, worse, overrides the page jump that just happened: a
    // citation on page 2 scrolled back to page 1 with the highlight off screen.
    // A bounding rect is in viewport space, so the arithmetic is the same
    // whatever else is positioned around it.
    const scroller = this.container.getBoundingClientRect();
    const rect = target.getBoundingClientRect();
    const markTop = rect.top - scroller.top + this.container.scrollTop;
    const desired = markTop - this.container.clientHeight / 2 + rect.height / 2;
    this.container.scrollTo({ top: Math.max(0, desired), behavior: "smooth" });
    return true;
  }

  _scrollToPage(pageNumber, behavior) {
    const element = this.states.get(pageNumber)?.element;
    if (!element) return;
    this._pendingPage = pageNumber;
    this.container.scrollTo({ top: Math.max(0, element.offsetTop - 8), behavior });
  }

  // -- teardown ----------------------------------------------------------

  _clearAll() {
    this._disconnectObserver();
    this._clearHighlights();
    for (const state of this.states.values()) {
      state.canvas?.remove();
      state.textLayerDiv?.remove();
      state.canvas = null;
      state.textLayerDiv = null;
      state.textLayer = null;
    }
    this.states.clear();
    this._surface = null;
  }

  _error(message) {
    this._disconnectObserver();
    this.states.clear();
    this.container.replaceChildren(errorBox(message));
  }
}

// -- module helpers ------------------------------------------------------

function clamp(value, low, high) {
  return Math.min(Math.max(value, low), high);
}

function chooseScale(pageWidth, containerWidth) {
  const available = Math.max(containerWidth - 24, 320);
  const raw = available / pageWidth;
  return clamp(Math.round(raw * 20) / 20, 0.5, 2.5);
}

function placeholder(text) {
  const node = document.createElement("div");
  node.className = "pdf-placeholder";
  node.textContent = text;
  return node;
}

function errorBox(message) {
  const box = document.createElement("div");
  box.className = "error-box";
  box.textContent = message;
  return box;
}

/** Unwrap marks in place, keeping the text they were wrapping. */
function clearMarks(marks) {
  for (const mark of marks) {
    if (mark.parentNode) mark.replaceWith(...mark.childNodes);
  }
}

function describePdfError(error) {
  const name = error?.name || "Error";
  if (name === "MissingPDFException") return "That file could not be read as a PDF.";
  if (name === "InvalidPDFException") return "That file is not a valid PDF, or it is truncated.";
  if (name === "PasswordException") return "That PDF is password protected, so it cannot be displayed.";
  if (name === "UnexpectedResponseException") {
    return `The server refused to serve that PDF (${error.message}).`;
  }
  return `Could not display that PDF: ${error?.message || name}`;
}

/**
 * Concatenate the text-layer spans into one string plus a map from string index
 * to (span index, index within that span's text). A space is inserted between
 * spans because pdf.js splits text chunks without preserving the whitespace that
 * separated them on the page.
 */
function concatSpans(spans) {
  let text = "";
  const map = [];
  spans.forEach((span, spanIndex) => {
    const content = span.textContent || "";
    if (spanIndex > 0 && text) {
      text += " ";
      map.push(null);
    }
    for (let offset = 0; offset < content.length; offset += 1) {
      map.push({ spanIndex, offset });
    }
    text += content;
  });
  return { text, map };
}

/**
 * Find `needle` in `haystack`, tolerating the two ways extraction mangles the
 * text a chunk was built from.
 *
 * 1. **Whitespace.** Layout adds line breaks and spacing the source text does not
 *    have, so any run of whitespace matches any other.
 * 2. **Hyphenation.** Justified text — and every converted DOCX/PPTX, whose PDF
 *    comes from the document's own layout — breaks long words at line ends: the
 *    page reads "we com-" / "pute" where the chunk reads "we compute". A literal
 *    match fails at the first such word, and one failed word loses the whole
 *    highlight, which is the bug this avoids.
 *
 * So each needle word may be matched in fragments, and between words only
 * whitespace (plus the one hyphen a line break leaves) may be skipped. Word
 * boundaries are respected on both sides, which keeps "compute" out of
 * "computer"; the needle's own spacing is ignored, because extraction decides
 * where the spaces are.
 *
 * Returns [start, end) index pairs into `haystack`: one per needle word matched,
 * so a phrase yields several ranges that the highlight merges per text span.
 * Verified against the real cases in tests/js/matcher_check.mjs.
 */
function matchFlexible(haystack, needle) {
  const hay = haystack.toLowerCase();
  const words = needle.toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return [];

  const wordChar = (index) =>
    index >= 0 && index < hay.length && /[a-z0-9]/.test(hay[index]);
  /** Only whitespace, and at most one hyphen, may separate two matched runs. */
  const isGap = (from, to) => /^\s*-?\s*$/.test(hay.slice(from, to));
  const skipGap = (from) => from + /^\s*-?\s*/.exec(hay.slice(from))[0].length;

  /**
   * The span covering `word` from `at`, or null when it does not fit.
   *
   * A word broken by the layout ("com-" + "pute") is matched fragment by
   * fragment: take as much of the word as sits right here, then continue after
   * the gap. The returned span includes the hyphen and any spaces, so it can be
   * marked as one highlight.
   */
  const matchWordAt = (word, at) => {
    let position = at;
    let consumed = 0;
    while (consumed < word.length) {
      const fragmentStart = skipGap(position);
      const rest = word.slice(consumed);
      if (hay.startsWith(rest, fragmentStart)) {
        return [at, fragmentStart + rest.length];
      }
      if (hay[fragmentStart] !== word[consumed]) return null;
      position = fragmentStart + 1;
      consumed += 1;
    }
    return [at, position];
  };

  /** The spans for every needle word from `start`, or null when they do not fit. */
  const matchNeedleFrom = (start) => {
    const spans = [];
    let searchFrom = start;
    for (const [index, word] of words.entries()) {
      let found = null;
      // Every character is stepped over by hand: `continue` inside a `for` with
      // an increment would return to the same position and never advance.
      for (let at = searchFrom; at < hay.length; ) {
        if (hay[at] !== word[0] || wordChar(at - 1)) {
          at += 1;
          continue;
        }
        if (index > 0 && !isGap(searchFrom, at)) {
          at += 1;
          continue;
        }
        const span = matchWordAt(word, at);
        if (!span || wordChar(span[1])) {
          at += 1;
          continue;
        }
        found = span;
        break;
      }
      if (!found) return null;
      spans.push(found);
      searchFrom = found[1];
    }
    return spans;
  };

  const ranges = [];
  for (let at = 0; at < hay.length; at += 1) {
    if (hay[at] !== words[0][0] || wordChar(at - 1)) continue;
    const spans = matchNeedleFrom(at);
    if (!spans) continue;
    ranges.push(...spans);
    // Carry on after this occurrence: overlapping matches are never wanted, and
    // the next occurrence still has to be found.
    at = spans[spans.length - 1][1] - 1;
  }
  return ranges;
}

// Exported for the node-based matcher test (tests/js/matcher_check.mjs): the
// hyphen/whitespace tolerance is the part of the viewer that a screenshot cannot
// show, so it is checked directly against the real cases.
export const __test = { matchFlexible };

/**
 * Turn every [start, end) match into `<mark class="pdf-highlight">` elements.
 *
 * All ranges are written in one pass, from the end of each span backwards,
 * because inserting a mark before an earlier offset would invalidate the
 * character indices the remaining ranges were computed against.
 *
 * A span's text is measured by the browser, so splitting it into before/hit/
 * after runs of text nodes makes each mark exactly as wide as the text it
 * covers — no glyph-width arithmetic, and correct for proportional fonts. A
 * match spanning two spans yields one mark per span, split at pdf.js's own line
 * boundary, which is the best a mark element can express.
 */
function markRanges(spans, map, ranges) {
  /** spanIndex -> array of [from, to) pairs within that span's text */
  const perSpan = new Map();
  for (const [start, end] of ranges) {
    for (let index = start; index < end; index += 1) {
      const location = map[index];
      if (!location) continue;
      const list = perSpan.get(location.spanIndex) || [];
      const last = list[list.length - 1];
      if (last && location.offset <= last[1]) last[1] = Math.max(last[1], location.offset + 1);
      else list.push([location.offset, location.offset + 1]);
      perSpan.set(location.spanIndex, list);
    }
  }

  const marks = [];
  for (const [spanIndex, list] of perSpan) {
    const span = spans[spanIndex];
    const content = span.textContent || "";
    const pieces = [];
    let cursor = 0;
    for (const [from, to] of list) {
      if (from > cursor) pieces.push(document.createTextNode(content.slice(cursor, from)));
      const mark = document.createElement("mark");
      mark.className = "pdf-highlight";
      mark.textContent = content.slice(from, to);
      pieces.push(mark);
      marks.push(mark);
      cursor = to;
    }
    if (cursor < content.length) pieces.push(document.createTextNode(content.slice(cursor)));
    span.replaceChildren(...pieces);
  }
  return marks;
}
