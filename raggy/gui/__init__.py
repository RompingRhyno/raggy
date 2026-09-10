"""The raggy GUI: an HTTP API and a static single-page front end.

The layout is the build plan's: the retrieved document is the main pane and the
chat is a sidebar, so the answer is something you check against the source
rather than a wall of text with citations underneath it.

``raggy.gui.server`` owns the HTTP surface; ``raggy.gui.state`` owns the
per-corpus lifecycle (open store, lock, status line). The web assets under
``web/`` are plain HTML/CSS/ES modules — no build step, no npm — so the GUI
runs from a source checkout with nothing installed but raggy itself.
"""

from .server import create_app

__all__ = ["create_app"]
