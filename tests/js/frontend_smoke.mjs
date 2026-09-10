// The GUI's front end, driven over a fake DOM against a real server.
//
// Nothing else in the suite can see this layer. `test_gui_api.py` proves the
// endpoints answer correctly and the static tests prove the files agree with
// each other, but neither runs the page: a startup-time bug in app.js — a
// temporal-dead-zone read, an element that is looked up before it exists, a
// handler never attached because the module threw on its first line — leaves a
// page that renders and then does nothing, with a clean server log and every
// endpoint test passing. That is exactly the failure this catches, so it drives
// the real module (fetched from the server) and clicks the real controls.
//
// Only Node's standard library is used: no jsdom, no browser. The fake DOM
// below implements what app.js and dom.js actually touch, which is deliberately
// a small surface — `el()` assigns reflected properties, `classList` carries the
// state, and every string is written with textContent.
//
// Usage: node frontend_smoke.mjs <base-url>

const BASE = process.argv[2];
if (!BASE) {
  console.error("usage: node frontend_smoke.mjs <base-url>");
  process.exit(2);
}

// `el()` assigns reflected properties (`title`, `placeholder`) directly, so a
// value can be on the property or on the attribute depending on how it was set.
function attribute(node, name) {
  if (!node) return null;
  const value = node.getAttribute(name);
  return value === null || value === undefined ? (node[name] ?? null) : value;
}

const failures = [];
function check(ok, description, detail) {
  if (ok) {
    console.log(`  ok   ${description}`);
  } else {
    failures.push(detail === undefined ? description : `${description} — ${detail}`);
    console.log(`  FAIL ${description}${detail === undefined ? "" : ` — ${detail}`}`);
  }
}

// -- a fake DOM ---------------------------------------------------------------

class ClassList {
  constructor() {
    this.set = new Set();
  }
  add(...names) {
    for (const name of names) if (name) this.set.add(name);
  }
  remove(...names) {
    for (const name of names) this.set.delete(name);
  }
  contains(name) {
    return this.set.has(name);
  }
  toggle(name, force) {
    const on = force === undefined ? !this.set.has(name) : Boolean(force);
    if (on) this.set.add(name);
    else this.set.delete(name);
    return on;
  }
  get value() {
    return [...this.set].join(" ");
  }
}

class Node {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.children = [];
    this.parent = null;
    this.attributes = new Map();
    this.dataset = {};
    this.style = {};
    this.listeners = new Map();
    this.classList = new ClassList();
    this._text = "";
    this.hidden = false;
    this.disabled = false;
    this.checked = false;
    this.value = "";
    this.id = "";
    this.role = "";
    // Reflected properties `el()` assigns directly, because `key in node` holds
    // for them; reading only the attribute would report them as unset.
    this.placeholder = "";
    this.src = "";
    this.href = "";
    this.alt = "";
    this.name = "";
    this.title = null;
  }
  get className() {
    return this.classList.value;
  }
  set className(value) {
    this.classList.set = new Set(String(value).split(/\s+/).filter(Boolean));
  }
  get textContent() {
    return this.children.length
      ? this.children.map((child) => child.textContent).join("")
      : this._text;
  }
  set textContent(value) {
    this.children = [];
    this._text = String(value ?? "");
  }
  get innerHTML() {
    return this._text;
  }
  set innerHTML(value) {
    this._text = String(value ?? "");
  }
  append(...nodes) {
    for (const node of nodes) {
      const child = typeof node === "string" ? new TextNode(node) : node;
      child.parent = this;
      this.children.push(child);
    }
    return this;
  }
  replaceChildren(...nodes) {
    this.children = [];
    return this.append(...nodes);
  }
  setAttribute(key, value) {
    this.attributes.set(key, String(value));
    if (key === "id") this.id = String(value);
    else if (key === "class") this.className = value;
    else if (key === "role") this.role = String(value);
    // Boolean attributes: present means true, and the markup writes them bare.
    else if (key === "hidden") this.hidden = true;
    else if (key === "disabled") this.disabled = true;
    else if (key.startsWith("data-")) this.dataset[key.slice(5)] = String(value);
  }
  getAttribute(key) {
    return this.attributes.has(key) ? this.attributes.get(key) : null;
  }
  removeAttribute(key) {
    this.attributes.delete(key);
  }
  addEventListener(type, handler) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(handler);
  }
  removeEventListener(type, handler) {
    const list = this.listeners.get(type) || [];
    const index = list.indexOf(handler);
    if (index >= 0) list.splice(index, 1);
  }
  dispatch(type, event = {}) {
    const list = [...(this.listeners.get(type) || [])];
    for (const handler of list) {
      handler({ type, target: this, preventDefault() {}, ...event });
    }
    return list.length;
  }
  focus() {}
  scrollTo() {}
  get offsetTop() {
    return 0;
  }
  get scrollTop() {
    return 0;
  }
  set scrollTop(_) {}
  get scrollHeight() {
    return 0;
  }
  querySelectorAll(selector) {
    return descendants(this).filter((node) => matches(node, selector));
  }
  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }
}

class TextNode extends Node {
  constructor(text) {
    super("#text");
    this._text = text;
  }
  get textContent() {
    return this._text;
  }
}

function descendants(node, out = []) {
  for (const child of node.children) {
    if (child instanceof TextNode) continue;
    out.push(child);
    descendants(child, out);
  }
  return out;
}

function matches(node, selector) {
  // Compound selectors matter: `.context-toggle.is-hidden` is how the hidden
  // state is read back, and matching only the first class would report every
  // toggle as visible.
  for (const part of selector.split(",").map((s) => s.trim())) {
    const classes = [...part.matchAll(/\.([A-Za-z0-9_-]+)/g)].map((m) => m[1]);
    const id = part.match(/#([A-Za-z0-9_-]+)/);
    const tag = part.match(/^[A-Za-z][A-Za-z0-9]*/);
    if (classes.length && classes.every((name) => node.classList.contains(name))) {
      if (!id || node.id === id[1]) return true;
      continue;
    }
    if (id && node.id === id[1]) return true;
    if (tag && !classes.length && !id && node.tagName === tag[0].toUpperCase()) return true;
  }
  return false;
}

/** Build the page's DOM from the served HTML: elements, attributes, nesting. */
function parseHtml(source) {
  const root = new Node("body");
  const stack = [root];
  const tag = /<(\/?)([a-zA-Z0-9]+)((?:\s+[^>]*?)?)(\/?)>/g;
  let cursor = 0;
  let match;
  while ((match = tag.exec(source))) {
    const text = source.slice(cursor, match.index);
    if (text.trim()) stack[stack.length - 1].append(text.replace(/\s+/g, " ").trim());
    cursor = tag.lastIndex;
    const [, closing, name, attrs, selfClosing] = match;
    if (closing) {
      if (stack.length > 1) stack.pop();
      continue;
    }
    const node = new Node(name);
    for (const attr of attrs.matchAll(/([a-zA-Z0-9_:.-]+)(?:\s*=\s*"([^"]*)")?/g)) {
      const [, key, value] = attr;
      node.setAttribute(key, value === undefined ? "" : value);
    }
    stack[stack.length - 1].append(node);
    if (
      !selfClosing &&
      !["input", "br", "img", "meta", "link"].includes(name.toLowerCase())
    ) {
      stack.push(node);
    }
  }
  return root;
}

// -- fetch, recorded ----------------------------------------------------------

const calls = [];
let pdfviewRequested = false;
const realFetch = globalThis.fetch;
async function fetchAndRecord(url, options = {}) {
  const target = String(url).startsWith("http") ? String(url) : `${BASE}${url}`;
  calls.push(`${options.method || "GET"} ${target.slice(BASE.length)}`);
  if (target.includes("/js/pdfview.js")) pdfviewRequested = true;
  const response = await realFetch(target, options);
  if (target.includes("/documents")) {
    console.log(`  [info] documents request served: ${response.status}`);
  }
  return response;
}

// -- run ---------------------------------------------------------------------

const html = await (await realFetch(`${BASE}/`)).text();
const dom = parseHtml(html);
const byId = new Map();
for (const node of descendants(dom)) if (node.id) byId.set(node.id, node);

const document = {
  body: dom,
  getElementById: (id) => byId.get(id) || null,
  createElement: (tag) => new Node(tag),
  createTextNode: (text) => new TextNode(text),
  addEventListener: () => {},
  querySelector: (selector) => dom.querySelector(selector),
  querySelectorAll: (selector) => dom.querySelectorAll(selector),
};

const window = {
  location: { href: `${BASE}/` },
  localStorage: {
    store: new Map(),
    getItem(key) {
      return this.store.has(key) ? this.store.get(key) : null;
    },
    setItem(key, value) {
      this.store.set(key, String(value));
    },
    removeItem(key) {
      this.store.delete(key);
    },
  },
  setInterval: () => 0,
  clearInterval: () => {},
  addEventListener: () => {},
};
window.fetch = fetchAndRecord;

globalThis.window = window;
globalThis.document = document;
globalThis.localStorage = window.localStorage;
globalThis.HTMLElement = Node;
globalThis.Node = Node;
globalThis.fetch = fetchAndRecord;

const rejections = [];
process.on("unhandledRejection", (error) => rejections.push(error));

// app.js imports pdf.js lazily, and only when a PDF is opened; this script never
// opens one. The stub is here so an accidental import fails loudly instead of
// reaching the network.
globalThis.FormData = globalThis.FormData || class FormData {};

// Node's ESM loader imports file: URLs only, so the served modules are mirrored
// into a directory that keeps their layout (`app.js` beside `js/`), leaving the
// relative specifiers inside them untouched. The content is byte-for-byte what
// the server sent, which is what is under test.
const { mkdir, writeFile } = await import("node:fs/promises");
const MIRROR = new URL("./.frontend-smoke/", import.meta.url);
await mkdir(MIRROR, { recursive: true });
const MODULES = [
  "app.js",
  "js/api.js",
  "js/dom.js",
  "js/markdown.js",
  "js/pdfview.js",
  "js/textview.js",
];
for (const name of MODULES) {
  const source = await (await realFetch(`${BASE}/${name}`)).text();
  const directory = name.split("/").slice(0, -1).join("/");
  if (directory) await mkdir(new URL(`${directory}/`, MIRROR), { recursive: true });
  await writeFile(new URL(name, MIRROR), source);
}
// Only the mirroring is direct; from here on every fetch goes through the
// recording wrapper, so the page's own requests are visible in `calls`.
globalThis.fetch = fetchAndRecord;

console.log("front end smoke test");

const list = byId.get("file-list");

// The file list is fetched, and listing a real corpus takes seconds (it walks
// the index and OCRs its images). Freeze that one response so the state in
// between is observable: while it is in flight the pane must say it is loading,
// and must not claim the corpus has no files — which is what a user watches for
// the whole wait on a corpus that is in fact full.
let held = null;
const holdDocuments = (url) => String(url).includes("/documents");
globalThis.fetch = (url, options) => {
  const call = fetchAndRecord(url, options);
  if (!holdDocuments(url)) return call;
  return new Promise((resolve, reject) => {
    held = { resolve, reject, call };
  });
};

let startupError = null;
try {
  await import(new URL("app.js", MIRROR).href);
} catch (error) {
  startupError = error;
}

let readDuringLoad = "";
// The listing is requested after the corpus list comes back, so wait for the
// request rather than assuming it has already been made.
for (let i = 0; i < 100 && !held; i += 1) {
  await new Promise((resolve) => setTimeout(resolve, 50));
}
if (held) {
  await new Promise((resolve) => setTimeout(resolve, 250));
  readDuringLoad = byId.get("file-list").textContent;
  console.log(`  [info] pane while listing: ${JSON.stringify(readDuringLoad)}`);
  console.log(
    `  [info] counts pill while listing: ${JSON.stringify(byId.get("file-counts").textContent)}`
  );
  const pending = held;
  globalThis.fetch = fetchAndRecord;
  pending.call.then(
    (response) => pending.resolve(response),
    (error) => pending.reject(error)
  );
} else {
  console.log("  [info] the listing was never requested, so the loading state could not be read");
}

check(!startupError, "app.js evaluates", startupError && startupError.message);
check(rejections.length === 0, "no unhandled rejections", rejections[0]?.message);
check(
  calls.some((call) => call.includes("/api/corpora")),
  "startup asks the server for its corpora",
  `calls: ${calls.join(", ")}`
);
check(
  readDuringLoad.length > 0 && !/no files|not been indexed|no files indexed/i.test(readDuringLoad),
  "the pane does not claim the corpus is empty while the list is loading",
  JSON.stringify(readDuringLoad)
);
check(
  /reading|loading/i.test(readDuringLoad),
  "the pane says it is busy while the list is in flight",
  JSON.stringify(readDuringLoad)
);
const deadline = Date.now() + 60_000;
while (Date.now() < deadline && !list.querySelectorAll(".file-item").length) {
  await new Promise((resolve) => setTimeout(resolve, 200));
}

check(
  byId.get("file-list").querySelectorAll(".file-skeleton-row").length === 0,
  "the loading placeholder is gone once the list arrives"
);
check(!pdfviewRequested, "pdf.js is still loaded lazily, only when a PDF is opened");

const select = byId.get("corpus-select");
const options = descendants(select).filter((node) => node.tagName === "OPTION");
check(options.length > 0, "the corpus dropdown is filled", `${options.length} options`);
check(
  options.every((option) => option.textContent && !option.textContent.includes("No corpora")),
  "the dropdown names a real corpus",
  options.map((option) => option.textContent).join(" | ")
);

// The list is re-rendered from scratch by every state change, so each step
// re-queries it: a node captured before a re-render is detached, and reading it
// reports the previous state.
const rowsNow = () => list.querySelectorAll(".file-item");
const togglesNow = () => list.querySelectorAll(".context-toggle");

const rows = rowsNow();
check(rows.length > 0, "the file list has rows", `${rows.length} rows`);
check(
  rows.length > 0 && rows.every((row) => row.querySelectorAll(".file-open").length === 1),
  "each row has its open button"
);

const toggles = togglesNow();
check(toggles.length === rows.length, "each row has a context toggle", `${toggles.length} toggles`);
check(
  toggles.length > 0 && toggles.every((toggle) => toggle.getAttribute("aria-pressed") === "true"),
  "files start in context"
);
check(
  toggles.length > 0 && toggles.every((toggle) => /status-/.test(toggle.className)),
  "toggles carry their file's status colour"
);

// Hiding a file must change the answer's scope, not just the row's looks. Each
// check guards the list being empty, so a dead page reports what is wrong rather
// than crashing the script on the first read.
const firstRow = rows[0];
const firstPath = firstRow ? attribute(firstRow.querySelector(".file-name"), "title") : null;
const firstName = firstRow ? firstRow.querySelector(".file-name").textContent : null;
if (toggles.length) togglesNow()[0].dispatch("click");
check(
  togglesNow().filter((toggle) => toggle.classList.contains("is-hidden")).length === 1,
  "clicking the toggle hides that file"
);
check(
  String(attribute(togglesNow()[0], "title")).includes("Hidden from context"),
  "the hidden toggle says so",
  attribute(togglesNow()[0], "title")
);
check(
  Boolean(rowsNow()[0]?.classList.contains("is-hidden-context")),
  "the hidden row is marked",
  `classes=${JSON.stringify([...(rowsNow()[0]?.classList.set ?? [])])}`
);
check(
  (byId.get("scope-note").textContent || "").includes("1 hidden"),
  "the ask pane counts the hidden file",
  byId.get("scope-note").textContent
);

// Opening a file must offer it as the scope of the next question. The hidden
// file is deliberately skipped: it must not be offerable.
const scopeFile = byId.get("scope-file");
const scopeAll = byId.get("scope-all");
check(scopeFile.disabled === true, "'This file' is disabled with nothing open");

const visible = rowsNow().find((row) => !row.classList.contains("is-hidden-context"));
const openedName = visible ? visible.querySelector(".file-name").textContent : null;
if (visible) visible.querySelector(".file-open").dispatch("click");
for (let i = 0; i < 20 && scopeFile.disabled; i += 1) {
  await new Promise((resolve) => setTimeout(resolve, 100));
}
check(scopeFile.disabled === false, "'This file' enables once a file is open");
check(scopeFile.textContent.includes(openedName || "\u0000"), "'This file' names the open file", scopeFile.textContent);
check(openedName !== firstName, "the opened file is a different one from the hidden file");

scopeFile.dispatch("click");
check(scopeAll.getAttribute("aria-pressed") === "false", "'All files' deselects");
check(scopeFile.getAttribute("aria-pressed") === "true", "'This file' is selected");
check(
  (byId.get("query-input").placeholder || "").includes(openedName),
  "the question box says what it is asked of",
  byId.get("query-input").placeholder
);

// And the scope has to reach the wire.
const bodies = [];
globalThis.fetch = async (url, options = {}) => {
  if (String(url).includes("/api/query")) bodies.push(JSON.parse(options.body));
  return fetchAndRecord(url, options);
};
byId.get("query-input").value = "what does this file say?";
byId.get("chat-form").dispatch("submit", { preventDefault() {} });
for (let i = 0; i < 20 && !bodies.length; i += 1) {
  await new Promise((resolve) => setTimeout(resolve, 100));
}
const body = bodies[0] || {};
check(Array.isArray(body.include_sources), "the question carries the file scope", JSON.stringify(body));
check(
  (body.include_sources || []).length === 1,
  "the scope names exactly the open file",
  JSON.stringify(body.include_sources)
);
check(
  (body.exclude_sources || []).includes(firstPath),
  "the question carries the hidden file",
  JSON.stringify(body.exclude_sources)
);

// -- deleting a corpus --------------------------------------------------------
//
// The one action here that cannot be undone, so it is checked end to end: the
// button opens a dialog that names what it is about, cancelling deletes
// nothing, and confirming sends a DELETE.

console.log("\n--- delete corpus ---");
const deleteBtn = byId.get("delete-corpus-btn");
const deleteModal = byId.get("delete-corpus-modal");
const confirmBtn = byId.get("delete-corpus-confirm");
check(!deleteBtn.disabled, "the delete button is available with a corpus open");
check(deleteModal.hidden, "the confirmation starts hidden");

deleteBtn.dispatch("click");
check(!deleteModal.hidden, "the delete button opens a confirmation");
const warning = byId.get("delete-corpus-warning").textContent;
check(
  warning.includes("Administrator") || /Remove /.test(warning),
  "the confirmation names the corpus it will remove",
  JSON.stringify(warning)
);

const deletesBefore = calls.filter((call) => call.startsWith("DELETE")).length;
byId.get("delete-corpus-cancel").dispatch("click");
check(deleteModal.hidden, "Cancel closes the confirmation");
check(
  calls.filter((call) => call.startsWith("DELETE")).length === deletesBefore,
  "Cancel deletes nothing",
  JSON.stringify(calls.filter((call) => call.startsWith("DELETE")))
);

deleteBtn.dispatch("click");
confirmBtn.dispatch("click");
for (let i = 0; i < 30 && !calls.some((call) => call.startsWith("DELETE")); i += 1) {
  await new Promise((resolve) => setTimeout(resolve, 100));
}
const deleted = calls.filter((call) => call.startsWith("DELETE"));
check(deleted.length === deletesBefore + 1, "confirming sends the delete", JSON.stringify(deleted));
check(
  deleted.some((call) => call.includes("delete_db=true")),
  "it asks for the index to go too, not just the config",
  JSON.stringify(deleted)
);

// A page that threw on startup never gets here, and the failure above already
// said why; this keeps the exit code honest either way.
if (failures.length) {
  console.log(`\n${failures.length} check(s) failed:`);
  for (const failure of failures) console.log(`  - ${failure}`);
  process.exit(1);
}
console.log("\nall checks passed");
