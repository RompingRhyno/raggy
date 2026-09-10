"""Vendor pdf.js into ``raggy/gui/web/vendor/pdfjs``.

python -m raggy.gui.vendor_pdfjs

The GUI renders PDFs (native ones and the DOCX/PPTX the indexer converts) with
pdf.js, and it must work offline on a machine with no npm: the library is
committed as static assets under the package, and this script is only the
one-time (or upgrade-time) step that puts them there.

Needs ``npm`` on PATH to fetch the package tarball; everything else is stdlib.
Nothing at runtime depends on npm — the app just serves the extracted files.
"""

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

PACKAGE = "pdfjs-dist"
DEFAULT_VERSION = "6.3.289"

VENDOR_DIR = Path(__file__).resolve().parent / "web" / "vendor" / "pdfjs"

# Everything the viewer needs: the module build, its worker, the CJK character
# maps, the standard font data (without which some PDFs render with the wrong
# glyphs or none at all), and pdf.js's own stylesheet. That last one matters: the
# rules that position the text layer's spans over the rendered page live there,
# not in the JS, so without it every span sits at the page's top-left corner.
INCLUDE_PREFIXES = (
    "package/build/pdf.min.mjs",
    "package/build/pdf.worker.min.mjs",
    "package/web/pdf_viewer.css",
)
INCLUDE_DIRS = {
    "package/cmaps/": "cmaps",
    "package/standard_fonts/": "standard_fonts",
}


def fetch_tarball(version: str, workdir: Path) -> Path:
    """Download the pdfjs-dist tarball with ``npm pack`` and return its path."""
    npm = shutil.which("npm")
    if npm is None:
        raise SystemExit(
            "npm not found on PATH. Either install Node.js, or copy the contents "
            f"of a pdfjs-dist checkout into '{VENDOR_DIR}' by hand "
            "(build/pdf.min.mjs, build/pdf.worker.min.mjs, cmaps/, standard_fonts/)."
        )
    print(f"Fetching {PACKAGE}@{version} ...")
    result = subprocess.run(
        [npm, "pack", f"{PACKAGE}@{version}"],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"npm pack failed:\n{result.stderr or result.stdout}")
    tarballs = sorted(workdir.glob("*.tgz"))
    if not tarballs:
        raise SystemExit("npm pack produced no tarball")
    return tarballs[-1]


def extract(tarball: Path, dest: Path) -> list[Path]:
    """Copy the pieces the viewer uses out of ``tarball`` into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with tarfile.open(tarball) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if member.name in INCLUDE_PREFIXES:
                target = dest / Path(member.name).name
                _copy(tar, member, target)
                written.append(target)
                continue
            for prefix, subdir in INCLUDE_DIRS.items():
                if member.name.startswith(prefix):
                    relative = Path(member.name).relative_to(prefix).as_posix()
                    target = dest / subdir / relative
                    _copy(tar, member, target)
                    written.append(target)
                    break
    return written


def _copy(tar: tarfile.TarFile, member: tarfile.TarInfo, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source = tar.extractfile(member)
    if source is None:  # pragma: no cover - getmembers only yields files
        return
    with source, target.open("wb") as out:
        shutil.copyfileobj(source, out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument(
        "--dest",
        default=str(VENDOR_DIR),
        help=f"target directory (default: {VENDOR_DIR})",
    )
    parser.add_argument(
        "--clean", action="store_true", help="remove the target directory first"
    )
    args = parser.parse_args(argv)

    dest = Path(args.dest)
    if args.clean and dest.exists():
        shutil.rmtree(dest)

    with tempfile.TemporaryDirectory(prefix="raggy-pdfjs-") as tmp:
        tarball = fetch_tarball(args.version, Path(tmp))
        written = extract(tarball, dest)

    if not written:
        raise SystemExit("nothing was extracted; the package layout may have changed")
    print(f"Wrote {len(written)} file(s) to {dest}")
    for path in sorted(written)[:6]:
        print(f"  {path.relative_to(dest)}")
    if len(written) > 6:
        print(f"  ... and {len(written) - 6} more")
    print("pdf.js is vendored; the GUI needs no npm from here on.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
