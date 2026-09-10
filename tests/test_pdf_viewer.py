"""The PDF viewer's citation highlight, measured in a real browser.

Clicking a citation opens the page and marks the cited line. The marking is done
by putting `<mark>` elements inside pdf.js's text layer, which only works while
they stay *inline*: the span is what pdf.js positions, so a mark that is itself
positioned collapses onto its containing block's origin and every fragment piles
up at one point on the page — the citation text overlapping itself into an
unreadable blob. That is a geometry bug, so it is checked as geometry.
"""

import itertools
import json
import socket
import threading
import time

import pymupdf
import pytest
import uvicorn

from raggy import indexing, pipeline
from raggy.gui.server import create_app
from tests.browser import HeadlessBrowser, find_chromium
from tests.test_gui_api import EvenScores, FakeEmbeddings, StubLLM

CITED_LINE = "Total amount due is 137.50 dollars"

# Three well-separated lines, so a highlight that lands on the wrong one is
# visible as a wrong y rather than as a near miss.
LINES = (
    "INVOICE FOR SERVICES RENDERED",
    CITED_LINE,
    "Payment terms are thirty days from issue",
)


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
def served_pdf(tmp_path, monkeypatch):
    """A real server holding one indexed PDF whose text is known."""
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "user"))
    (tmp_path / "user").mkdir()
    monkeypatch.setattr(indexing, "get_embeddings", lambda model: FakeEmbeddings())
    monkeypatch.setattr(pipeline, "get_llm", lambda *a, **k: StubLLM())
    monkeypatch.setattr(pipeline, "get_cross_encoder", lambda model: EvenScores([]))
    monkeypatch.setattr(
        "raggy.gui.state.ensure_models", lambda cfg, progress=None: None
    )

    docs = tmp_path / "docs"
    docs.mkdir()
    # Three pages, with the cited line on the *second*: a jump that lands on the
    # wrong page is only visible when the target is not page 1.
    pdf = pymupdf.open()
    for number in (1, 2, 3):
        page = pdf.new_page(width=612, height=792)
        for index, line in enumerate(
            [f"Page {number} heading"]
            + [
                f"Page {number} body line {index} with some words on it"
                for index in range(6)
            ]
            + ([CITED_LINE] if number == 2 else [])
        ):
            page.insert_text((72, 120 + index * 60), line, fontsize=14)
    pdf.save(docs / "invoice.pdf")
    pdf.close()

    # A second document whose pages are not all the same shape: page 1 is short
    # and the cited page 2 is tall, which is the arrangement where a page box
    # taken from page 1 is wrong for the page being jumped to.
    uneven = pymupdf.open()
    for number, (width, height) in enumerate(
        ((842, 480), (595, 842), (595, 842)), start=1
    ):
        page = uneven.new_page(width=width, height=height)
        for index, line in enumerate(
            [f"Page {number} heading"]
            + [
                f"Page {number} body line {index} with some words on it"
                for index in range(5)
            ]
            + ([CITED_LINE] if number == 2 else [])
        ):
            page.insert_text((60, 100 + index * 50), line, fontsize=13)
    uneven.save(docs / "uneven.pdf")
    uneven.close()

    # A document whose cited line sits at the *bottom* of page 2: centring the
    # page by itself leaves that line below the fold.
    bottom = pymupdf.open()
    for number in (1, 2):
        page = bottom.new_page(width=612, height=792)
        for index in range(8):
            page.insert_text(
                (72, 100 + index * 60),
                f"Page {number} body line {index} with some words on it",
                fontsize=14,
            )
        if number == 2:
            page.insert_text((72, 740), CITED_LINE, fontsize=14)
    bottom.save(docs / "bottom.pdf")
    bottom.close()

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
            "/api/corpora", json={"name": "Pdf", "sources": [str(docs)]}
        ).json()["corpus"]
        client.post(f"/api/corpora/{corpus['id']}/refresh")
        entries = {
            entry["name"]: entry
            for entry in client.get(f"/api/corpora/{corpus['id']}/documents").json()[
                "documents"
            ]
        }
        yield {
            "base": f"http://127.0.0.1:{port}/",
            "invoice": entries["invoice.pdf"]["content_url"],
            "uneven": entries["uneven.pdf"]["content_url"],
            "bottom": entries["bottom.pdf"]["content_url"],
        }
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
def browser(tmp_path):
    executable = find_chromium()
    if executable is None:
        pytest.skip("no Chromium-based browser is installed")
    instance = HeadlessBrowser(executable, tmp_path / "browser-profile", (1200, 900))
    try:
        yield instance
    finally:
        instance.close()


def open_cited_page(browser, base_url, content_url, page=2, wait=3500):
    """Open the viewer the way a citation click does, and measure the marks.

    The cited line is on page 2, so `page` is what a citation's jump target
    carries; opening it from page 1 is where a broken jump is visible. ``wait``
    has to outlast the smooth centring as well as the rendering, or the
    measurement catches the viewport mid-scroll.
    """
    browser.open(base_url, settle=1.5)
    value = browser.evaluate(
        f"""(async () => {{
             const {{ PdfViewer, globalOptions }} = await import('/js/pdfview.js');
             globalOptions.cMapUrl = '/vendor/pdfjs/cmaps/';
             globalOptions.cMapPacked = true;
             globalOptions.standardFontDataUrl = '/vendor/pdfjs/standard_fonts/';
             const host = document.createElement('div');
             host.className = 'pdf-host';
             host.style.cssText = 'width: 900px; height: 700px; overflow: auto;';
             document.body.append(host);
             const viewer = new PdfViewer(host);
             window.__probeViewer = viewer; // kept so a test can re-open the same one
             await viewer.open({{ url: {json.dumps(content_url)}, page: {page},
                                  search: {json.dumps(CITED_LINE)} }});
             // The jump settles as the page renders, and centres itself with a
             // smooth scroll, so its final position arrives a variable time after
             // open(). Wait for the viewport to stop moving rather than for a
             // fixed delay, which is what made this measurement flaky.
             let settled = 0;
             let previous = null;
             for (let i = 0; i < 80; i += 1) {{
               await new Promise((r) => setTimeout(r, 100));
               const mark = host.querySelector('.pdf-highlight');
               const position = `${{Math.round(host.scrollTop)}}:${{mark ? Math.round(mark.getBoundingClientRect().top) : 'x'}}`;
               if (position === previous) {{
                 settled += 1;
                 if (settled >= 4) break;
               }} else {{
                 settled = 0;
                 previous = position;
               }}
             }}
             await new Promise((r) => setTimeout(r, {wait}));
             const pages = [...host.querySelectorAll('.pdf-page')];
             const page = pages[{page} - 1];
             if (!page) return {{ error: 'the cited page was not built' }};
             const pageBox = page.getBoundingClientRect();
             const scroller = host.getBoundingClientRect();
             const marks = [...host.querySelectorAll('.pdf-highlight')];
             const visible = pages
               .map((element, index) => {{
                 const box = element.getBoundingClientRect();
                 const overlap =
                   Math.min(box.bottom, scroller.bottom) - Math.max(box.top, scroller.top);
                 return {{ page: index + 1, visible: Math.max(0, overlap) }};
               }})
               .filter((entry) => entry.visible > 40);
             return {{
               count: marks.length,
               text: marks.map((m) => m.textContent).join(' '),
               currentPage: viewer.currentPage,
               visiblePages: visible.map((entry) => entry.page),
               markVisible: marks.length
                 ? marks.every((m) => {{
                     const box = m.getBoundingClientRect();
                     return box.top >= scroller.top - 2 && box.bottom <= scroller.bottom + 2;
                   }})
                 : null,
               markInViewport: marks.length
                 ? (() => {{
                     const box = marks[0].getBoundingClientRect();
                     return {{
                       top: box.top,
                       bottom: box.bottom,
                       viewportTop: scroller.top,
                       viewportBottom: scroller.bottom,
                     }};
                   }})()
                 : null,
               markColours: [...new Set(marks.map((m) => getComputedStyle(m).color))],
               markBackgrounds: [...new Set(marks.map((m) => getComputedStyle(m).backgroundColor))],
               fontSizes: [
                 ...new Set(
                   [...page.querySelectorAll('.textLayer span')].map(
                     (s) => getComputedStyle(s).fontSize
                   )
                 ),
               ],
               marks: marks.map((m) => {{
                 const r = m.getBoundingClientRect();
                 return {{
                   x: r.left - pageBox.left,
                   y: r.top - pageBox.top,
                   width: r.width,
                   height: r.height,
                 }};
               }}),
             }};
           }})()"""
    )
    return json.loads(value) if isinstance(value, str) else value


def test_the_cited_line_is_highlighted(served_pdf, browser):
    base_url, content_url = served_pdf["base"], served_pdf["invoice"]

    result = open_cited_page(browser, base_url, content_url)

    assert "error" not in result, result
    assert result["count"] >= 2, f"expected the cited words to be marked: {result}"
    assert "Total" in result["text"] and "dollars" in result["text"], result["text"]


def test_a_citation_jumps_to_the_page_it_is_on(served_pdf, browser):
    """The reported bug: a citation on page 2 stayed on page 1.

    The jump happened, and was then undone by the highlight centring itself on a
    position measured from the mark's `offsetTop` — which, for a mark inside
    pdf.js's absolutely positioned spans, is a couple of pixels within its own
    span rather than a place in the document. That scrolled back to the top.
    """
    base_url, content_url = served_pdf["base"], served_pdf["invoice"]

    result = open_cited_page(browser, base_url, content_url, page=2)

    assert result["currentPage"] == 2, result
    assert result["visiblePages"] == [2], (
        f"the viewport is showing {result['visiblePages']}"
    )
    assert result["markVisible"] is True, "the highlighted line is off screen"


def test_opening_the_cited_page_again_does_not_jump_back(served_pdf, browser):
    """Clicking a second citation on the page already on screen.

    Nothing needs to scroll on the way in, so this is the case where a bad
    measurement has nothing to hide behind: it scrolls away from the page it is
    already on.
    """
    base_url, content_url = served_pdf["base"], served_pdf["invoice"]
    open_cited_page(browser, base_url, content_url, page=2)

    result = browser.evaluate(
        f"""(async () => {{
             const host = document.querySelector('.pdf-host');
             const viewer = window.__probeViewer;
             await viewer.open({{ url: {json.dumps(content_url)}, page: 2,
                                 search: {json.dumps(CITED_LINE)} }});
             await new Promise((r) => setTimeout(r, 2500));
             const scroller = host.getBoundingClientRect();
             const pages = [...host.querySelectorAll('.pdf-page')];
             const visible = pages
               .map((element, index) => {{
                 const box = element.getBoundingClientRect();
                 const overlap =
                   Math.min(box.bottom, scroller.bottom) - Math.max(box.top, scroller.top);
                 return {{ page: index + 1, visible: Math.max(0, overlap) }};
               }})
               .filter((entry) => entry.visible > 40)
               .map((entry) => entry.page);
             const marks = [...host.querySelectorAll('.pdf-highlight')];
             return {{
               visiblePages: visible,
               markVisible: marks.every((m) => {{
                 const box = m.getBoundingClientRect();
                 return box.top >= scroller.top - 2 && box.bottom <= scroller.bottom + 2;
               }}),
             }};
           }})()"""
    )

    assert result["visiblePages"] == [2], f"re-opening moved the viewport: {result}"
    assert result["markVisible"] is True, result


def test_a_jump_to_a_page_of_another_shape_still_lands_on_it(served_pdf, browser):
    """Pages are not obliged to be the same size.

    Every page box is built from page 1's dimensions, so a page that is a
    different shape carries the wrong box until it renders -- and a jump made
    before then points at the wrong place. The jump has to be re-confirmed once
    that page's real geometry is known.
    """
    base_url, uneven_url = served_pdf["base"], served_pdf["uneven"]

    result = open_cited_page(browser, base_url, uneven_url, page=2)

    assert result["currentPage"] == 2, result
    assert 2 in result["visiblePages"], (
        f"the viewport is showing {result['visiblePages']}"
    )
    assert result["markVisible"] is True, "the highlighted line is off screen"


def test_a_citation_at_the_bottom_of_the_page_is_brought_into_view(served_pdf, browser):
    """Landing on the right page is not enough.

    A jump is aimed at a phrase. If the page is centred by itself after the
    phrase has been found, a cited line near the bottom of the page ends up below
    the fold: the viewer reports the right page and the user sees no highlight.
    """
    base_url, bottom_url = served_pdf["base"], served_pdf["bottom"]

    result = open_cited_page(browser, base_url, bottom_url, page=2)

    assert result["currentPage"] == 2, result
    assert result["markVisible"] is True, "the cited line is off screen"
    assert result["markInViewport"] is not None, result
    # Brought into view, but not left as the last sliver of the page.
    below = (
        result["markInViewport"]["viewportBottom"] - result["markInViewport"]["bottom"]
    )
    assert below > 40, f"the cited line is only {below:.0f}px above the bottom edge"


def test_the_highlight_never_paints_its_own_text(served_pdf, browser):
    """The doubled-text report.

    The highlight is a `<mark>`, and the browser's default `mark { color: black }`
    beats the transparent colour pdf.js sets on the span. Every cited word was
    therefore drawn a second time in black over the canvas - visibly doubled
    wherever the text layer and the rendered glyphs are not pixel-identical. Only
    the background may be ours.
    """
    base_url, content_url = served_pdf["base"], served_pdf["invoice"]

    result = open_cited_page(browser, base_url, content_url)

    assert result["markColours"] == ["rgba(0, 0, 0, 0)"], (
        f"the highlight is painting its own text: {result['markColours']}"
    )
    assert result["markBackgrounds"] != ["rgba(0, 0, 0, 0)"], (
        "the highlight has no background"
    )


def test_every_overlay_run_is_sized_from_the_page(served_pdf, browser):
    """The overlay's runs must all be sized by pdf.js's own rule.

    A run that misses that rule keeps an inherited font size and no transform, and
    is laid out at the wrong scale: the text layer then drifts off the glyphs it is
    supposed to sit on, which is what makes the overlay visible as doubled text.
    The `.markedContent` wrappers pdf.js can emit are deliberately not covered
    here -- it only emits them for `includeMarkedContent: true`, which this viewer
    does not pass, so no fixture can produce one. The CSS keeps pdf.js's selector
    for them regardless.
    """
    base_url, content_url = served_pdf["base"], served_pdf["invoice"]
    browser.open(base_url, settle=1.5)

    result = browser.evaluate(
        f"""(async () => {{
             const {{ PdfViewer }} = await import('/js/pdfview.js');
             const host = document.createElement('div');
             host.className = 'pdf-host';
             host.style.cssText = 'width: 900px; height: 700px; overflow: auto;';
             document.body.append(host);
             const viewer = new PdfViewer(host);
             await viewer.open({{ url: {json.dumps(content_url)}, page: 1 }});
             await new Promise((r) => setTimeout(r, 2000));
             const layer = host.querySelector('.textLayer');
             if (!layer) return {{ error: 'no text layer' }};
             const spans = [...layer.querySelectorAll('span')];
             return {{
               spans: spans.length,
               unsized: spans.filter(
                 (s) => getComputedStyle(s).getPropertyValue('--font-height') === ''
               ).length,
               untransformed: spans.filter(
                 (s) => getComputedStyle(s).transform === 'none'
               ).length,
               transparent: spans.filter(
                 (s) => getComputedStyle(s).color === 'rgba(0, 0, 0, 0)'
               ).length,
               positioned: spans.filter(
                 (s) => getComputedStyle(s).position === 'absolute'
               ).length,
             }};
           }})()"""
    )
    result = json.loads(result) if isinstance(result, str) else result

    assert "error" not in result, result
    assert result["spans"] > 0, result
    # Every run carries what the rule gives it, and stays invisible.
    assert result["unsized"] == 0, result
    assert result["untransformed"] == 0, result
    assert result["positioned"] == result["spans"], result
    assert result["transparent"] == result["spans"], result


def test_the_highlight_fragments_do_not_pile_up_on_each_other(served_pdf, browser):
    """The bug this guards, stated as geometry.

    Each mark covers the words of one span. Marks that are positioned — rather
    than inline inside the span pdf.js placed — all collapse onto the same point,
    so their widths still look right but their x positions stop advancing and the
    text is drawn on top of itself.
    """
    base_url, content_url = served_pdf["base"], served_pdf["invoice"]
    result = open_cited_page(browser, base_url, content_url)
    marks = sorted(result["marks"], key=lambda mark: mark["x"])

    assert len(marks) >= 2, result
    # The cited line is one line, so every fragment shares its vertical position.
    tops = {round(mark["y"]) for mark in marks}
    assert len(tops) == 1, f"the fragments are not on one line: {sorted(tops)}"
    # And they advance along it, left to right, one after another.
    for earlier, later in itertools.pairwise(marks):
        assert later["x"] >= earlier["x"] + earlier["width"] - 1, (
            f"fragment at x={later['x']:.0f} overlaps the one before it "
            f"(x={earlier['x']:.0f}, width={earlier['width']:.0f})"
        )
    # The words of the line add up to a run along it, not a stack in one spot.
    assert marks[-1]["x"] - marks[0]["x"] >= 200, (
        f"every fragment sits within {marks[-1]['x'] - marks[0]['x']:.0f}px of the first"
    )


def test_the_highlight_covers_only_the_cited_line(served_pdf, browser):
    """The other half of the bug: the text was unreadable because it was all in
    one place. Marks must also be the width of the words they cover, not the
    width of the page."""
    base_url, content_url = served_pdf["base"], served_pdf["invoice"]
    result = open_cited_page(browser, base_url, content_url)
    marks = result["marks"]

    span = max(mark["x"] + mark["width"] for mark in marks) - min(
        mark["x"] for mark in marks
    )
    assert span < 500, f"the highlight spans {span:.0f}px, which is most of the page"
    for mark in marks:
        assert mark["height"] < 60, f"a fragment is {mark['height']:.0f}px tall: {mark}"
