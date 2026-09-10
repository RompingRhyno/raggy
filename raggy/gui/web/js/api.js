// Thin fetch wrapper for the raggy GUI API (see raggy/gui/API.md).
//
// Two rules shape this module:
//   1. Every failure on the wire is an `ApiError` carrying the server's own
//      `message` and `hint`. The UI shows those verbatim — the server writes
//      them for the person who clicked the button, so rewording them here would
//      throw away the useful part.
//   2. Errors are thrown, never returned. Callers get either data or an
//      exception, which keeps "did this work?" out of every call site.

export class ApiError extends Error {
  constructor(message, { status = 0, hint = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.hint = hint;
  }
}

async function request(path, { method = "GET", body, signal } = {}) {
  let response;
  try {
    response = await fetch(path, {
      method,
      signal,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (cause) {
    // fetch only rejects for transport-level problems; the server being gone is
    // the common one while it restarts.
    if (cause && cause.name === "AbortError") throw cause;
    throw new ApiError(`could not reach the raggy server: ${cause.message}`);
  }

  const payload = await readJson(response);
  if (!response.ok) {
    const error = payload && typeof payload === "object" ? payload.error : null;
    throw new ApiError(messageFrom(response, payload, error), {
      status: response.status,
      hint: error?.hint || null,
    });
  }
  if (payload === null) {
    throw new ApiError(`${path} returned a non-JSON response (HTTP ${response.status})`, {
      status: response.status,
    });
  }
  return payload;
}

async function readJson(response) {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

/**
 * The documented shape is `{error: {message, hint}}`, but FastAPI's own request
 * validation (HTTP 422) answers with `{detail: [...]}` instead — a bare "422
 * Unprocessable Entity" would tell the user nothing, so those messages are
 * lifted out verbatim too.
 */
function messageFrom(response, payload, error) {
  if (error?.message) return error.message;
  const detail = payload && typeof payload === "object" ? payload.detail : null;
  if (Array.isArray(detail) && detail.length) {
    const parts = detail.map((item) => item?.msg || JSON.stringify(item)).filter(Boolean);
    if (parts.length) return parts.join("; ");
  }
  if (typeof detail === "string" && detail) return detail;
  return `${response.status} ${response.statusText || "request failed"}`;
}

export const api = {
  health: () => request("/api/health"),

  corpora: () => request("/api/corpora"),
  createCorpus: (name, sources) => request("/api/corpora", { method: "POST", body: { name, sources } }),
  activateCorpus: (id) => request(`/api/corpora/${encodeURIComponent(id)}/activate`, { method: "POST" }),
  patchCorpus: (id, patch) => request(`/api/corpora/${encodeURIComponent(id)}`, { method: "PATCH", body: patch }),
  addSource: (id, name, sources) =>
    request(`/api/corpora/${encodeURIComponent(id)}/sources`, { method: "POST", body: { name, sources } }),
  deleteCorpus: (id, deleteDb = true) =>
    request(`/api/corpora/${encodeURIComponent(id)}?delete_db=${deleteDb ? "true" : "false"}`, { method: "DELETE" }),

  browse: (path, showFiles = true) => {
    const params = new URLSearchParams();
    if (path) params.set("path", path);
    if (showFiles) params.set("show_files", "true");
    return request(`/api/browse?${params.toString()}`);
  },

  documents: (id) => request(`/api/corpora/${encodeURIComponent(id)}/documents`),

  // Blocking: resolves when the whole indexing run is finished, which may be
  // minutes later. Callers own the progress UI and are expected to poll
  // `status()` while this promise is pending.
  refresh: (id, signal) => request(`/api/corpora/${encodeURIComponent(id)}/refresh`, { method: "POST", signal }),

  status: (id) => request(`/api/corpora/${encodeURIComponent(id)}/status`),

  // `scope` scopes retrieval to a set of files: `include_sources` narrows to
  // those (the ask pane's "this file"), `exclude_sources` takes files out of
  // whatever that selected (the files hidden from context). Omitted, the whole
  // corpus is searched.
  query: (query, corpusId, scope = {}) => {
    const suffix = corpusId ? `?corpus_id=${encodeURIComponent(corpusId)}` : "";
    return request(`/api/query${suffix}`, { method: "POST", body: { query, ...scope } });
  },
};

/** Fetch a document's bytes as text (the text/html viewers). */
export async function fetchText(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const payload = await readJson(response);
    throw new ApiError(payload?.error?.message || `could not read the file (HTTP ${response.status})`, {
      status: response.status,
      hint: payload?.error?.hint || null,
    });
  }
  return response.text();
}
