// Viewers for everything that is not a PDF: plain text with line numbers,
// sanitized HTML, and images with their OCR text beside them.
//
// All three expose the same shape app.js relies on:
//   open({url, ...}) / destroy() / highlight(search) / jump(search)
// and never throw for bad content — they render an inline error instead, so a
// failed viewer cannot leave a spinner or a blank pane behind.

import { fetchText } from "./api.js";
import { clear, el, findMatchRanges } from "./dom.js";

// -- text ----------------------------------------------------------------

export class TextViewer {
  constructor(container) {
    this.container = container;
    this.lines = [];
    this.marks = [];
  }

  async open({ url, startLine = null, note = "" } = {}) {
    this.destroy();
    clear(this.container);
    if (note) this.container.append(noteLine(note));
    let text;
    try {
      text = await fetchText(url);
    } catch (error) {
      this.container.append(errorBox(error));
      return;
    }
    if (!this.container.isConnected) return;

    // Trailing newline yields one empty last line; dropping it keeps the gutter
    // numbers honest about the file's content.
    const content = text.replace(/\r\n?/g, "\n").replace(/\n$/, "");
    const lines = content.split("\n");
    const list = el("div", { class: "text-lines" });
    lines.forEach((line, index) => {
      // The line number itself comes from a CSS counter on .text-line, so the
      // gutter is invisible to text selection and to the DOM.
      list.append(
        el("div", { class: "text-line", dataset: { line: String(index + 1) } }, [
          el("span", { class: "text-line-body", text: line || " " }),
        ])
      );
      this.lines.push(list.lastElementChild.firstElementChild);
    });
    this.container.append(el("div", { class: "text-scroll" }, [list]));
    this._scroller = this.container.querySelector(".text-scroll");
    this._list = list;

    if (Number.isFinite(startLine)) this.jumpLine(startLine);
  }

  /** Centre `line` (1-based) and mark it as the citation's target line. */
  jumpLine(line) {
    const target = this.lines[line - 1];
    if (!target) return;
    this._list.querySelectorAll(".text-line.is-target").forEach((node) => node.classList.remove("is-target"));
    target.closest(".text-line")?.classList.add("is-target");
    // offsetTop of the row inside .text-lines, minus a margin for the header.
    const row = target.closest(".text-line");
    if (this._scroller && row) {
      this._scroller.scrollTo({ top: Math.max(0, row.offsetTop - 80), behavior: "smooth" });
    }
  }

  highlight(search) {
    this.clearMarks();
    const needle = String(search || "").trim();
    if (!needle) return 0;
    for (const body of this.lines) {
      this.marks.push(...wrapTextRange(body, 0, (body.textContent || "").length, needle));
    }
    return this.marks.length;
  }

  /** Scroll to and highlight the citation's search string. */
  jump(search = null) {
    const needle = String(search || "").trim();
    if (!needle) return false;
    const count = this.highlight(needle);
    const first = this.marks[0];
    if (!first) return false;
    const row = first.closest(".text-line");
    if (this._scroller && row) {
      this._scroller.scrollTo({ top: Math.max(0, row.offsetTop - 80), behavior: "smooth" });
    }
    first.classList.add("is-current");
    return true;
  }

  clearMarks() {
    for (const mark of this.marks) {
      const parent = mark.parentNode;
      if (!parent) continue;
      parent.replaceWith(...parent.childNodes);
    }
    this.marks = [];
  }

  destroy() {
    this.lines = [];
    this.marks = [];
    this._scroller = null;
    this._list = null;
    clear(this.container);
  }
}

/**
 * Wrap every occurrence of `needle` inside `node`'s [from, to) character range
 * in a <mark>, leaving the rest of the DOM untouched. Works from the end of the
 * range backwards so earlier offsets stay valid while nodes are split.
 */
function wrapTextRange(node, from, to, needle) {
  const text = (node.textContent || "").slice(from, to);
  const ranges = findMatchRanges(text, needle).map(([start, end]) => [from + start, from + end]);
  if (!ranges.length) return [];
  const marks = [];
  for (const [start, end] of ranges.reverse()) markOffsets(node, start, end, marks);
  return marks.reverse();
}

/** Split text nodes so that [start, end) is inside a single new <mark>. */
function markOffsets(root, start, end, marks) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let offset = 0;
  let node = walker.nextNode();
  let startNode = null;
  let endNode = null;
  let startOffset = 0;
  let endOffset = 0;
  while (node) {
    const length = node.nodeValue.length;
    if (startNode === null && offset + length > start) {
      startNode = node;
      startOffset = start - offset;
    }
    if (offset + length >= end) {
      endNode = node;
      endOffset = end - offset;
      break;
    }
    offset += length;
    node = walker.nextNode();
  }
  if (!startNode || !endNode) return;
  const range = document.createRange();
  range.setStart(startNode, startOffset);
  range.setEnd(endNode, endOffset);
  const mark = document.createElement("mark");
  mark.className = "hit";
  try {
    range.surroundContents(mark);
  } catch {
    // A range crossing element boundaries cannot be surrounded; fall back to
    // the text of the range, which is still better than no highlight.
    mark.textContent = range.toString();
    range.deleteContents();
    range.insertNode(mark);
  }
  marks.push(mark);
}

// -- html ----------------------------------------------------------------

export class HtmlViewer {
  constructor(container) {
    this.container = container;
    this.marks = [];
  }

  async open({ url, note = "" } = {}) {
    this.destroy();
    clear(this.container);
    if (note) this.container.append(noteLine(note));
    let text;
    try {
      text = await fetchText(url);
    } catch (error) {
      this.container.append(errorBox(error));
      return;
    }
    if (!this.container.isConnected) return;
    const body = el("div", { class: "html-body" });
    body.append(sanitizeHtml(text));
    this.container.append(el("div", { class: "html-scroll" }, [body]));
    this._scroller = this.container.querySelector(".html-scroll");
    this._body = body;
  }

  highlight(search) {
    this.clearMarks();
    const needle = String(search || "").trim();
    if (!needle || !this._body) return 0;
    for (const node of textNodesOf(this._body)) {
      this.marks.push(...wrapTextRange(node, 0, (node.nodeValue || "").length, needle));
    }
    return this.marks.length;
  }

  jump(search = null) {
    const needle = String(search || "").trim();
    if (!needle) return false;
    const count = this.highlight(needle);
    const first = this.marks[0];
    if (!first) return false;
    first.classList.add("is-current");
    const top = offsetWithin(first, this._scroller);
    this._scroller?.scrollTo({ top: Math.max(0, top - 80), behavior: "smooth" });
    return true;
  }

  clearMarks() {
    for (const mark of this.marks) {
      const parent = mark.parentNode;
      if (!parent) continue;
      parent.replaceWith(...parent.childNodes);
    }
    this.marks = [];
  }

  destroy() {
    this.marks = [];
    this._scroller = null;
    this._body = null;
    clear(this.container);
  }
}

function textNodesOf(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes = [];
  let node = walker.nextNode();
  while (node) {
    nodes.push(node);
    node = walker.nextNode();
  }
  return nodes;
}

function offsetWithin(node, ancestor) {
  let top = 0;
  let current = node;
  while (current && current !== ancestor) {
    top += current.offsetTop || 0;
    current = current.offsetParent;
  }
  return top;
}

// Tags that either execute, load remote content, or restate the page — all of
// which are wrong when a corpus file is rendered inside the app's own document.
const DROPPED_TAGS = new Set([
  "script",
  "style",
  "link",
  "meta",
  "base",
  "iframe",
  "frame",
  "frameset",
  "object",
  "embed",
  "applet",
  "form",
  "input",
  "button",
  "textarea",
  "select",
  "option",
  "template",
  "noscript",
  "svg",
  "math",
]);

const URL_ATTRIBUTES = new Set(["href", "src", "srcset", "action", "formaction", "poster", "xlink:href", "data"]);

/**
 * Parse untrusted HTML and return a node safe to insert.
 *
 * The corpus file is data, not code: scripts, styles, frames, form controls and
 * every `on*` handler are removed, and only absolute or fragment URLs survive.
 * `<style>` goes too, which is why the app supplies its own typography for
 * `.html-body` — a file cannot restyle the app's chrome through the viewer.
 */
export function sanitizeHtml(source) {
  const parsed = new DOMParser().parseFromString(String(source ?? ""), "text/html");
  clean(parsed.body);
  const fragment = document.createDocumentFragment();
  fragment.append(...parsed.body.childNodes);
  return fragment;
}

function clean(root) {
  for (const element of [...root.querySelectorAll("*")]) {
    if (DROPPED_TAGS.has(element.tagName.toLowerCase())) {
      element.remove();
      continue;
    }
    for (const attribute of [...element.attributes]) {
      const name = attribute.name.toLowerCase();
      const value = attribute.value;
      if (name.startsWith("on") || name === "srcdoc" || name === "style" || name === "formaction") {
        element.removeAttribute(attribute.name);
        continue;
      }
      if (URL_ATTRIBUTES.has(name) && !safeUrl(value)) {
        // A relative URL would resolve against the GUI server and 404; a
        // scripting URL would execute. Neither is worth keeping.
        element.removeAttribute(attribute.name);
      }
    }
    if (element.tagName.toLowerCase() === "a") {
      element.setAttribute("target", "_blank");
      element.setAttribute("rel", "noopener noreferrer");
    }
  }
}

function safeUrl(raw) {
  const value = String(raw ?? "").trim();
  if (!value) return false;
  return /^(https?:|mailto:|tel:|data:image\/|#|\/)/i.test(value);
}

// -- image ---------------------------------------------------------------

export class ImageViewer {
  constructor(container) {
    this.container = container;
  }

  /**
   * `note` is rendered inside this viewer rather than by app.js, because open()
   * clears the container: anything appended from outside would be wiped before
   * it could be seen. It carries the "no text layer to search" line that the
   * build plan's feature 4 option (a) requires for images.
   *
   * The OCR panel is only rendered when there is OCR text. An image with none —
   * a blank scan, a photo with no lettering, a file indexed before it was
   * refreshed — has nothing to put in a panel, and an empty one with a
   * placeholder is worse than no panel at all.
   */
  async open({ url, ocrText = "", note = "" } = {}) {
    this.destroy();
    const text = String(ocrText || "").trim();
    if (note) this.container.append(noteLine(note));
    const image = el("img", { src: url, alt: "Source image", loading: "eager" });
    image.addEventListener("error", () => {
      clear(this.container);
      this.container.append(errorBox(new Error("that image could not be loaded")));
    });
    const stage = el("div", { class: "image-stage" }, [image]);
    if (!text) {
      this.container.append(el("div", { class: "image-layout" }, [el("div", { class: "image-row image-row-solo" }, [stage])]));
      return;
    }
    const panel = el("div", { class: "ocr-panel" }, [
      el("div", { class: "ocr-head", text: "Text read from this image (OCR)" }),
      el("pre", { class: "ocr-text", text }),
    ]);
    this.container.append(
      el("div", { class: "image-layout" }, [
        el("div", { class: "image-row" }, [stage, panel]),
      ])
    );
  }

  /** Images cannot be searched; the OCR panel is the whole feature. */
  highlight() {
    return 0;
  }

  jump() {
    return false;
  }

  destroy() {
    clear(this.container);
  }
}

function errorBox(error) {
  const box = el("div", { class: "error-box" });
  box.append(el("div", { text: error?.message || String(error) }));
  if (error?.hint) box.append(el("div", { class: "error-hint", text: error.hint }));
  return box;
}

function noteLine(text) {
  return el("div", { class: "note-line", text });
}
