"""Drive a headless browser over the DevTools protocol, with no dependencies.

Layout is the one thing this project cannot test any other way. The API tests
prove the server answers, and the static checks prove the stylesheet says what it
means — but neither can see that a button's hover border is clipped by its row,
or that a status dot sits four pixels below the file name it belongs to. Those
are geometry, and geometry needs a layout engine.

Chromium is driven directly: ``--headless=new --remote-debugging-port`` plus a
websocket and JSON. No Playwright, no selenium, no download — and where no
browser is installed, :func:`headless_chromium` returns None and the test skips.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import subprocess
import time
from pathlib import Path

CHROMIUM_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/usr/bin/microsoft-edge",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


def find_chromium() -> str | None:
    """Path to a Chromium-based browser, or None if there is not one."""
    for candidate in CHROMIUM_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class HeadlessBrowser:
    """One page in a headless browser, evaluated on demand."""

    def __init__(self, executable: str, profile: Path, window: tuple[int, int]):
        self.executable = executable
        self.profile = profile
        self.port = _free_port()
        self._next_id = 0
        self._console: list[str] = []
        # The browser writes to this for as long as it runs, so it outlives this
        # call by design; `close` is what closes it.
        profile.parent.mkdir(parents=True, exist_ok=True)
        self._log = (profile.parent / "browser.log").open("wb")
        self.process = subprocess.Popen(
            [
                executable,
                "--headless=new",
                f"--remote-debugging-port={self.port}",
                "--remote-allow-origins=*",
                f"--user-data-dir={profile}",
                "--no-first-run",
                "--no-default-browser-check",
                f"--window-size={window[0]},{window[1]}",
                "about:blank",
            ],
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )
        page = self._wait_for_page()
        self.sock = self._connect(page["webSocketDebuggerUrl"])
        self.call("Page.enable")

    # -- lifecycle ---------------------------------------------------------
    def _wait_for_page(self, timeout: float = 30) -> dict:
        import urllib.request

        deadline = time.time() + timeout
        last_error = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/json/list", timeout=1
                ) as response:
                    targets = json.load(response)
                pages = [t for t in targets if t["type"] == "page"]
                if pages:
                    return pages[0]
            except Exception as e:  # noqa: BLE001 - the browser is still starting
                last_error = e
            time.sleep(0.2)
        raise RuntimeError(f"the browser never exposed a page target: {last_error}")

    def _connect(self, url: str) -> socket.socket:
        from urllib.parse import urlparse

        parts = urlparse(url)
        sock = socket.create_connection((parts.hostname, parts.port))
        sock.settimeout(30)
        key = base64.b64encode(os.urandom(16)).decode()
        sock.sendall(
            (
                f"GET {parts.path} HTTP/1.1\r\nHost: {parts.hostname}:{parts.port}\r\n"
                f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
            ).encode()
        )
        buffer = b""
        while b"\r\n\r\n" not in buffer:
            chunk = sock.recv(4096)
            if not chunk:
                raise RuntimeError("the browser closed the websocket handshake")
            buffer += chunk
        status = buffer.split(b"\r\n", 1)[0].decode(errors="replace")
        if "101" not in status:
            raise RuntimeError(f"websocket refused: {status}")
        return sock

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - stubborn browser
            self.process.kill()
        self._log.close()

    # -- protocol ----------------------------------------------------------
    def _send(self, payload: str) -> None:
        data = payload.encode()
        # Client frames must be masked: an unmasked one is a protocol error, and
        # the browser closes the connection without saying so.
        mask = os.urandom(4)
        masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(data))
        header = bytearray([0x81])
        length = len(data)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header += struct.pack(">H", length)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", length)
        self.sock.sendall(bytes(header) + mask + masked)

    def _receive(self) -> str:
        def read(count: int) -> bytes:
            buffer = b""
            while len(buffer) < count:
                chunk = self.sock.recv(count - len(buffer))
                if not chunk:
                    raise RuntimeError("the browser closed the connection")
                buffer += chunk
            return buffer

        first = read(2)
        length = first[1] & 0x7F
        if length == 126:
            length = struct.unpack(">H", read(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", read(8))[0]
        return read(length).decode()

    def call(self, method: str, **params):
        self._next_id += 1
        message_id = self._next_id
        self._send(json.dumps({"id": message_id, "method": method, "params": params}))
        while True:
            message = json.loads(self._receive())
            if message.get("method") == "Runtime.consoleAPICalled":
                self._remember_console(message)
                continue
            if message.get("id") == message_id:
                if "error" in message:
                    raise RuntimeError(f"{method} failed: {message['error']}")
                return message.get("result", {})

    def _remember_console(self, message: dict) -> None:
        """Record a console message, rendering object arguments readably."""
        parts = []
        for argument in message.get("params", {}).get("args", []):
            if "value" in argument:
                parts.append(str(argument["value"]))
                continue
            preview = argument.get("preview") or {}
            fields = [
                f"{prop.get('name')}: {prop.get('value')}"
                for prop in preview.get("properties", [])
            ]
            if fields:
                parts.append("{" + ", ".join(fields) + "}")
            else:
                parts.append(str(argument.get("description") or argument.get("type")))
        text = " ".join(parts)
        self._console.append(f"[{message['params'].get('type')}] {text}")

    # -- the two things a test needs ---------------------------------------
    def open(self, url: str, settle: float = 1.5) -> None:
        self.call("Page.navigate", url=url)
        time.sleep(settle)

    def enable_console(self) -> None:
        """Collect console output, which is where a page's own tracing lands.

        Console messages arrive as protocol *events*, interleaved with the
        responses to our own calls, so they are collected by the message loop
        rather than by a call of their own.
        """
        self._console = []
        self.call("Runtime.enable")

    def console(self) -> list[str]:
        """The console output collected so far."""
        return list(self._console)

    def evaluate(self, expression: str):
        """Run ``expression`` in the page and return its value.

        ``awaitPromise`` so an async expression (one that clicks something and
        waits for the page to react) resolves to its value rather than to a
        serialized ``Promise``.
        """
        result = self.call(
            "Runtime.evaluate",
            expression=expression,
            returnByValue=True,
            awaitPromise=True,
        )
        outcome = result.get("result", {})
        if result.get("exceptionDetails"):
            raise RuntimeError(f"page error: {result['exceptionDetails']}")
        return outcome.get("value")

    def hover(self, x: float, y: float) -> None:
        """Move the pointer, which is what CSS `:hover` responds to."""
        self.call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y, buttons=0)
        time.sleep(0.3)

    def screenshot(self, path: Path) -> Path:
        data = self.call("Page.captureScreenshot", format="png")["data"]
        path.write_bytes(base64.b64decode(data))
        return path
