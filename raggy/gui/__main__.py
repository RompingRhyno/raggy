"""``raggy-gui``: launch the local GUI server.

Binds to the loopback interface and prints the URL to open. One process, one
user, no authentication — the plan's assumption is a local single-user app, and
binding to 127.0.0.1 is what keeps that assumption true.
"""

import argparse
import logging
import sys


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="raggy-gui",
        description="Run the raggy GUI (document viewer + chat) in a local web server.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to bind (default: 127.0.0.1, loopback only)",
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="port to listen on (default: 8765)"
    )
    parser.add_argument(
        "--home",
        default=None,
        help="directory holding GUI state (default: $RAGGY_GUI_HOME or ~/.raggy/gui)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="server log level (default: info)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        import uvicorn
    except ImportError:  # pragma: no cover - uvicorn is a declared dependency
        print(
            "The GUI needs uvicorn, which is not installed. "
            "Install it with: uv sync  (or: pip install uvicorn)",
            file=sys.stderr,
        )
        return 1

    from pathlib import Path

    from ..corpora import CorpusStore
    from .server import create_app

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s: %(message)s",
    )

    home = Path(args.home) if args.home else None
    store = CorpusStore(home)
    store.home.mkdir(parents=True, exist_ok=True)

    app = create_app(home)
    corpus = store.active()
    print(f"raggy GUI on http://{args.host}:{args.port}")
    print(f"GUI state: {store.home}")
    if corpus is None:
        print("No corpus yet — add one from the GUI's corpus menu.")
    else:
        print(f"Active corpus: {corpus.name} ({len(corpus.sources)} source(s))")

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
