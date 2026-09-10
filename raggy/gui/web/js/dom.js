// DOM helpers shared by the viewers and the chat pane.
//
// Everything that renders model output or file content goes through this
// module, so there is exactly one place to look when asking "can a string from
// disk or from the LLM end up as markup?". The answer is no: text is only ever
// written with `textContent`, `document.createTextNode`, or the literal
// element-building in `el`. No helper here takes an HTML string.

/** Build an element; props are assigned, except `class`/`text` and `on*` handlers. */
export function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key in node && key !== "list" && key !== "form" && key !== "type") node[key] = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

export function clear(node) {
  node.replaceChildren();
  return node;
}

/** Case-insensitive substring search used by every highlight path. */
export function findMatchRanges(haystack, needle) {
  const ranges = [];
  if (!needle) return ranges;
  const hay = haystack.toLowerCase();
  const target = needle.toLowerCase();
  let index = hay.indexOf(target);
  while (index !== -1) {
    ranges.push([index, index + target.length]);
    index = hay.indexOf(target, index + target.length);
  }
  return ranges;
}

/**
 * Replace `node`'s children with its text, wrapping every match of `needle` in
 * `<mark>`. Returns the created marks so callers can style or scroll to one.
 */
export function highlightTextIn(node, needle, markClass = "hit") {
  const text = node.textContent || "";
  const ranges = findMatchRanges(text, needle);
  if (!ranges.length) return [];
  node.replaceChildren();
  const marks = [];
  let cursor = 0;
  for (const [start, end] of ranges) {
    if (start > cursor) node.append(document.createTextNode(text.slice(cursor, start)));
    const mark = el("mark", { class: markClass, text: text.slice(start, end) });
    node.append(mark);
    marks.push(mark);
    cursor = end;
  }
  if (cursor < text.length) node.append(document.createTextNode(text.slice(cursor)));
  return marks;
}

/**
 * Scroll `target` into view inside `scroller`, centring it when it fits.
 * `block: "center"` alone would hide the first line of a text file under the
 * panel header, so the offset is applied by hand.
 */
export function revealInScroller(scroller, target, offset = 60) {
  if (!scroller || !target) return;
  const top = target.offsetTop - offset;
  scroller.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
}

export function formatBytes(size) {
  if (typeof size !== "number" || !Number.isFinite(size)) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = size;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value < 10 && unit > 0 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}

export function formatScore(score) {
  if (typeof score !== "number" || !Number.isFinite(score)) return "";
  return score.toFixed(3);
}

export function basename(path) {
  const parts = String(path || "").split(/[\\/]/);
  return parts[parts.length - 1] || String(path || "");
}
