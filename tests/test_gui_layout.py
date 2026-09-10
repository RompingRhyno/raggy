"""The file list's layout, measured in a real browser.

Why this exists: a status dot that sits below its file name, or a button whose
hover border is clipped by the row around it, passes every other test in this
suite. The server answers correctly and the stylesheet says what it means — the
bug is in the geometry, which only a layout engine can report. Each of these
checks stands for a bug that was actually shipped and seen.
"""

import json
import socket
import threading
import time

import pytest
import uvicorn

from raggy import indexing, pipeline
from raggy.gui.server import create_app
from tests.browser import HeadlessBrowser, find_chromium
from tests.test_gui_api import EvenScores, FakeEmbeddings, StubLLM

# What the row's own hover border and the button's border are, from styles.css.
ROW_BORDER = "rgba(88, 166, 255, 0.35)"
BUTTON_BORDER = "rgb(51, 59, 72)"

MEASURE = """
(() => {
  // A single-line row: a failed file's row carries an extra detail line and is
  // taller by design, so it is not what "how tall is a row" means.
  const rows = [...document.querySelectorAll('.file-item')];
  const row = rows.find((r) => !r.querySelector('.file-detail')) || rows[0];
  if (!row) return JSON.stringify({ error: 'no row rendered' });
  const toggle = row.querySelector('.context-toggle');
  const name = row.querySelector('.file-name');
  const box = (el) => { const r = el.getBoundingClientRect(); return {
    top: r.top, bottom: r.bottom, left: r.left, right: r.right,
    width: r.width, height: r.height, cx: r.left + r.width / 2, cy: r.top + r.height / 2 }; };
  const style = getComputedStyle(toggle);
  const rowStyle = getComputedStyle(row);
  // A pseudo-element has no box of its own, so the dot's size is read from its
  // declared width; it is centred in the button by the button's own flex rules.
  const dotSize = parseFloat(getComputedStyle(toggle, '::before').width);
  const toggleBox = box(toggle);
  return JSON.stringify({
    row: box(row),
    rowBorderColor: rowStyle.borderTopColor,
    rowBorderWidth: parseFloat(rowStyle.borderTopWidth),
    toggle: toggleBox,
    borderColor: style.borderTopColor,
    borderWidth: parseFloat(style.borderTopWidth),
    background: style.backgroundColor,
    dotSize: dotSize,
    name: box(name),
    label: (() => { const r = name.getBoundingClientRect();
      return { top: r.top, bottom: r.bottom, cy: r.top + r.height / 2 }; })(),
  });
})()
"""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _wait_until_up(port: int, deadline: float) -> bool:
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


@pytest.fixture
def served_gui(tmp_path, monkeypatch):
    """A real server with one indexed corpus whose names are long enough to
    truncate, which is the case where the row's geometry is tightest."""
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "user"))
    monkeypatch.setattr(indexing, "get_embeddings", lambda model: FakeEmbeddings())
    monkeypatch.setattr(pipeline, "get_llm", lambda *a, **k: StubLLM())
    monkeypatch.setattr(pipeline, "get_cross_encoder", lambda model: EvenScores([]))
    monkeypatch.setattr(
        "raggy.gui.state.ensure_models", lambda cfg, progress=None: None
    )

    docs = tmp_path / "docs"
    docs.mkdir()
    for name in (
        "a-report-with-a-rather-long-name-that-truncates.pdf",
        "broken.pdf",
        "added-yesterday.txt",
    ):
        (docs / name).write_text("Some indexed content " * 30, encoding="utf-8")

    app = create_app(tmp_path / "gui")
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    assert _wait_until_up(port, time.time() + 20), "the test server did not start"
    try:
        import httpx

        client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=120)
        corpus = client.post(
            "/api/corpora", json={"name": "Layout", "sources": [str(docs)]}
        ).json()["corpus"]
        client.post(f"/api/corpora/{corpus['id']}/refresh")
        yield f"http://127.0.0.1:{port}/"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
def browser(tmp_path):
    executable = find_chromium()
    if executable is None:
        pytest.skip("no Chromium-based browser is installed")
    instance = HeadlessBrowser(executable, tmp_path / "browser-profile", (1200, 800))
    try:
        yield instance
    finally:
        instance.close()


def measure(browser, base_url):
    browser.open(base_url)
    return json.loads(browser.evaluate(MEASURE))


def test_the_status_dot_is_centred_on_its_file_name(served_gui, browser):
    """The dot is the row's control; the name is what it belongs to.

    Centring each of them against the other is wrong — one is a box and the other
    is a line of text with a baseline — and the result is a dot sitting visibly
    below the name it marks.
    """
    data = measure(browser, served_gui)

    assert "error" not in data, data.get("error")
    # The dot is centred in the button by the button's own flex rules, so the
    # button's centre is the dot's centre.
    dot_centre = (data["toggle"]["top"] + data["toggle"]["bottom"]) / 2
    offset = abs(dot_centre - data["label"]["cy"])
    assert offset <= 1.5, f"the dot is {offset:.1f}px off the centre of the file name"


def test_the_toggle_border_only_appears_once_the_row_is_hovered(served_gui, browser):
    """A permanent box around every marker reads as decoration; the point of
    showing it on hover is to say "this is a button"."""
    data = measure(browser, served_gui)

    assert data["borderColor"] == "rgba(0, 0, 0, 0)", data["borderColor"]
    assert data["background"] == "rgba(0, 0, 0, 0)", data["background"]

    # Hover the row away from the button: this is the gesture the user makes.
    browser.hover(data["row"]["left"] + data["row"]["width"] * 0.6, data["row"]["cy"])
    hovered = json.loads(browser.evaluate(MEASURE))

    assert hovered["borderColor"] == BUTTON_BORDER, hovered["borderColor"]
    assert hovered["borderWidth"] == 1
    assert hovered["background"] != "rgba(0, 0, 0, 0)", "the button needs a face"


def test_the_toggle_border_is_inside_the_row_and_not_clipped(served_gui, browser):
    """The bug this guards: a negative margin pulled the button's border into the
    row's own border, so the two overlapped instead of nesting."""
    data = measure(browser, served_gui)
    row, toggle = data["row"], data["toggle"]

    assert toggle["left"] >= row["left"] + data["rowBorderWidth"] - 0.01, (
        f"the button starts outside the row (row {row['left']:.1f}, "
        f"button {toggle['left']:.1f})"
    )
    assert toggle["right"] <= row["right"] - data["rowBorderWidth"] + 0.01
    assert toggle["top"] >= row["top"] + data["rowBorderWidth"] - 0.01
    assert toggle["bottom"] <= row["bottom"] - data["rowBorderWidth"] + 0.01

    # And it has to keep clear of the file name it sits beside.
    assert toggle["right"] <= data["name"]["left"] + 0.01


def test_the_toggle_is_centred_in_its_row(served_gui, browser):
    """The button is the row's tallest element, so its own box is what centres
    it. A fixed height with a hand-tuned top margin is what put it half a line
    below the name; centring has to come from the layout.

    The tolerance is a pixel because the row's own border is on the outside of
    that arithmetic: pad(4) + border(1) above, and one pixel less below.
    """
    data = measure(browser, served_gui)
    row, toggle = data["row"], data["toggle"]

    above = toggle["top"] - row["top"]
    below = row["bottom"] - toggle["bottom"]

    assert abs(above - below) <= 2, (
        f"the button is not centred in its row: {above:.1f}px above, {below:.1f}px below"
    )
    assert 2 <= above <= 7, f"the button has {above:.1f}px of clearance above it"


def test_the_toggle_is_almost_as_tall_as_the_row_it_sits_in(served_gui, browser):
    """The button was a 14px box in a row this size: too small to read as a
    control beside the row's own border. It is sized to nearly the row's height
    now — the padding and the border are what is left between the two."""
    data = measure(browser, served_gui)
    row_height = data["row"]["bottom"] - data["row"]["top"]
    toggle_height = data["toggle"]["height"]

    assert toggle_height >= 20, f"the button is only {toggle_height}px tall"
    assert toggle_height < row_height, (
        "the button cannot fill the row: the gap is the point"
    )
    # Row = button + 2*(4px padding + 1px border). Any more than that means the
    # button has shrunk relative to the row.
    assert row_height - toggle_height <= 12, (
        f"the button is {row_height - toggle_height:.0f}px shorter than the row"
    )
    # The dot inside it is unchanged: the button grew, the marker did not.
    assert data["dotSize"] == 7
