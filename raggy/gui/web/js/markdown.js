// A deliberately small markdown renderer for model answers.
//
// Why hand-rolled instead of a library: the GUI has no build step and no CDN, so
// there is no dependency to install — and the security property matters more
// than the feature list. `renderMarkdown` NEVER passes model text through as
// markup. Every character of the answer is escaped first; the only tags in the
// result are the ones this file writes itself, and the only values interpolated
// into those tags are (a) indices into the escape-fragment table used for code
// spans and links, or (b) URLs that survived `safeUrl`.
//
// Supported: paragraphs, ATX headings, fenced code blocks, blockquotes, ordered
// and nested unordered lists, horizontal rules, pipe tables, and the inline set
// `**bold**`, `*italic*`/`_italic_`, `` `code` ``, and `[text](url)`.

const MAX_INLINE_DEPTH = 3;
const LIST_ITEM = /^(\s*)([-*+]|\d{1,3}[.)])\s+(.*)$/;
const HEADING = /^\s{0,3}(#{1,6})\s+(.*)$/;
const FENCE = /^\s*(```|~~~)\s*([\w+#.-]*)\s*$/;
const RULE = /^\s{0,3}([-*_])\s*(\1\s*){2,}$/;
const QUOTE = /^\s{0,3}>/;

export function renderMarkdown(source) {
  return renderBlocks(String(source ?? "").replace(/\r\n?/g, "\n").split("\n"));
}

// -- block level ---------------------------------------------------------

function renderBlocks(lines) {
  const out = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];

    if (!line.trim()) {
      index += 1;
      continue;
    }

    const fence = line.match(FENCE);
    if (fence) {
      const closer = new RegExp(`^\\s*${fence[1]}\\s*$`);
      const body = [];
      index += 1;
      while (index < lines.length && !closer.test(lines[index])) {
        body.push(lines[index]);
        index += 1;
      }
      index += 1; // consume the closing fence, or run off the end of the answer
      const language = fence[2] ? ` class="language-${escapeHtml(fence[2])}"` : "";
      out.push(`<pre><code${language}>${escapeHtml(body.join("\n"))}</code></pre>`);
      continue;
    }

    const heading = line.match(HEADING);
    if (heading) {
      // Deeper than h3 reads as noise in a 400px sidebar.
      const level = Math.min(heading[1].length, 3);
      const text = heading[2].replace(/\s+#+\s*$/, "");
      out.push(`<h${level}>${inline(text, out, 0)}</h${level}>`);
      index += 1;
      continue;
    }

    if (RULE.test(line)) {
      out.push("<hr>");
      index += 1;
      continue;
    }

    if (QUOTE.test(line)) {
      const body = [];
      while (index < lines.length && QUOTE.test(lines[index])) {
        body.push(lines[index].replace(/^\s{0,3}>\s?/, ""));
        index += 1;
      }
      out.push(`<blockquote>${renderBlocks(body)}</blockquote>`);
      continue;
    }

    if (LIST_ITEM.test(line)) {
      index = renderList(lines, index, out);
      continue;
    }

    if (line.includes("|") && index + 1 < lines.length && isTableSeparator(lines[index + 1])) {
      index = renderTable(lines, index, out);
      continue;
    }

    const paragraph = [];
    while (index < lines.length && lines[index].trim() && !startsBlock(lines, index)) {
      paragraph.push(lines[index].trim());
      index += 1;
    }
    if (paragraph.length) {
      out.push(`<p>${inline(paragraph.join(" "), out, 0)}</p>`);
    } else {
      // A line that looks like a block start but matched no branch above would
      // otherwise be consumed by nothing and spin forever.
      index += 1;
    }
  }

  return resolveFragments(out);
}

function startsBlock(lines, index) {
  const line = lines[index];
  return (
    FENCE.test(line) ||
    HEADING.test(line) ||
    RULE.test(line) ||
    QUOTE.test(line) ||
    LIST_ITEM.test(line) ||
    (line.includes("|") && index + 1 < lines.length && isTableSeparator(lines[index + 1]))
  );
}

/**
 * Render one list, recursively.
 *
 * The shape that falls out — `<ul><li>a<ul><li>b</li></ul></li><li>c</li></ul>`
 * — is why this is recursive rather than a stack of open tags: a sub-list is a
 * child of the item that introduced it, so the parent `<li>` cannot be closed
 * until the sub-list has been fully rendered.
 */
function renderList(lines, start, out) {
  let index = start;
  const first = lines[index].match(LIST_ITEM);
  const baseIndent = indentOf(first);
  const ordered = /\d/.test(first[2][0]);
  const tag = ordered ? "ol" : "ul";
  out.push(`<${tag}>`);

  while (index < lines.length) {
    const match = lines[index].match(LIST_ITEM);
    if (!match) break;
    const indent = indentOf(match);
    if (indent < baseIndent) break;
    if (indent === baseIndent && /\d/.test(match[2][0]) !== ordered) break;
    if (indent > baseIndent) {
      // A deeper bullet: recurse, and the returned index is where the parent
      // list continues.
      index = renderList(lines, index, out);
      continue;
    }

    const body = [match[3]];
    index += 1;
    while (index < lines.length && lines[index].trim() && !LIST_ITEM.test(lines[index]) && /^\s{2,}/.test(lines[index])) {
      body.push(lines[index].trim());
      index += 1;
    }
    out.push(`<li>${inline(body.join(" "), out, 0)}`);

    // A deeper bullet here belongs inside this item; anything else closes it.
    const next = index < lines.length ? lines[index].match(LIST_ITEM) : null;
    if (next && indentOf(next) > indent) {
      index = renderList(lines, index, out);
    }
    out.push("</li>");
  }

  out.push(`</${tag}>`);
  return index;
}

function indentOf(match) {
  return match[1].replace(/\t/g, "    ").length;
}

function isTableSeparator(line) {
  const cells = splitRow(line);
  return cells.length > 0 && cells.every((cell) => /^:?-{2,}:?$/.test(cell));
}

function splitRow(line) {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function renderTable(lines, start, out) {
  const head = splitRow(lines[start]);
  const rows = [];
  let index = start + 2;
  while (index < lines.length && lines[index].trim() && lines[index].includes("|")) {
    rows.push(splitRow(lines[index]));
    index += 1;
  }
  const cells = (tag, values) =>
    values.map((value) => `<${tag}>${inline(value, out, 0)}</${tag}>`).join("");
  out.push(
    `<table><thead><tr>${cells("th", head)}</tr></thead><tbody>${rows
      .map((row) => `<tr>${cells("td", row)}</tr>`)
      .join("")}</tbody></table>`
  );
  return index;
}

// -- inline level --------------------------------------------------------

/**
 * Render one line of markdown to an HTML fragment.
 *
 * `fragments` (the enclosing block's output array) doubles as the escape table:
 * code spans and links push their finished HTML there and leave an opaque token
 * behind, so the surrounding text can be escaped in one pass without the
 * fragments' own markup being escaped with it. The tokens are substituted once,
 * at the end of the block, by :func:`resolveFragments` — which is also what
 * makes a link whose label contains a code span come out right, since by then
 * every fragment exists.
 */
function inline(text, fragments, depth) {
  const withoutCode = extract(text, /`([^`\n]+)`/g, fragments, (match, into) => {
    into.push(`<code>${escapeHtml(match[1])}</code>`);
    return into.length - 1;
  });

  // Links are extracted before emphasis so that a URL's underscores or
  // asterisks cannot be mistaken for emphasis markers.
  const withoutLinks = extract(
    withoutCode,
    /\[([^\]\n]*)\]\(([^)\s]*)\)/g,
    fragments,
    (match, into) => {
      const url = safeUrl(match[2]);
      const label =
        depth < MAX_INLINE_DEPTH ? inline(match[1], into, depth + 1) : escapeHtml(match[1]);
      if (!url) {
        // Refusing the scheme must not delete the words: keep the label as text.
        into.push(label);
      } else {
        into.push(
          `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${label}</a>`
        );
      }
      return into.length - 1;
    }
  );

  let html = escapeHtml(withoutLinks);
  if (depth < MAX_INLINE_DEPTH) {
    html = html
      .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
      .replace(/__([^_\n]+)__/g, "<strong>$1</strong>")
      .replace(/(^|[^*\w])\*([^*\n]+)\*/g, "$1<em>$2</em>")
      // The leading class excludes snake_case: `a_b_c` must stay literal.
      .replace(/(^|[^_\w])_([^_\n]+)_(?![\w])/g, "$1<em>$2</em>");
  }

  return html;
}

/**
 * Substitute the tokens left by :func:`extract` with their fragments.
 *
 * Runs once per block, after every fragment exists: a fragment may itself hold a
 * token (a link whose label contains a code span), and expanding recursively is
 * what resolves that nesting. The array passed to :func:`inline` as the fragment
 * context is the one joined here — a nested ``renderBlocks`` call resolves its
 * own block first, so by this point every token in these strings has a slot.
 */
function resolveFragments(out) {
  const resolved = new Set();
  const expand = (value) =>
    String(value).replace(/\u0000(\d+)\u0000/g, (whole, index) => {
      const position = Number(index);
      // A fragment that is itself a token slot (a link label made of a code
      // span) resolves through to its content; the guard stops a fragment from
      // expanding itself.
      if (resolved.has(position)) return whole;
      resolved.add(position);
      return out[position] === undefined ? "" : expand(out[position]);
    });

  for (let index = 0; index < out.length; index += 1) out[index] = expand(out[index]);
  return out.join("");
}

/**
 * Replace every `pattern` match in `text` with an opaque token, pushing the
 * match's finished fragment into `fragments` through `make`. Tokens are
 * `\u0000<i>\u0000`, which cannot occur in real text and survives `escapeHtml`
 * untouched. `make` receives the sink and returns the index of the fragment it
 * wants substituted, so a callback that pushes several fragments (a recursively
 * rendered link label) points the token at the right one.
 */
function extract(text, pattern, fragments, make) {
  const value = text.replace(pattern, (...args) => {
    const match = args.slice(0, -2); // drop offset and the input string
    return `\u0000${make(match, fragments)}\u0000`;
  });
  return value;
}

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

/**
 * Only absolute, non-scripting URLs survive. A markdown link from a local model
 * is untrusted input: `javascript:` in an href is script execution in the app's
 * own origin, and a relative path silently 404s against the GUI server.
 */
function safeUrl(raw) {
  const url = String(raw ?? "").trim().replace(/^<|>$/g, "");
  if (!url) return null;
  if (/^(https?:|mailto:|tel:)/i.test(url)) return url;
  if (url.startsWith("/") || url.startsWith("#")) return url;
  return null;
}
