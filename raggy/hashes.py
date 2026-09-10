"""Content hashing, shared by the indexer and the corpus walk.

Kept in its own module so the loaders can fingerprint files without importing
the indexing layer (which imports the loaders).
"""

import hashlib
from pathlib import Path

# Read in 1 MiB blocks rather than slurping whole files: the corpus can include
# large PDFs/images, and every file is hashed on each pipeline start.
# (hashlib.file_digest would replace this loop, but it needs Python 3.11.)
HASH_BLOCK_SIZE = 1048576


def hash_file(path: Path | str) -> str:
    """Return the SHA-256 digest of a file's contents."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(HASH_BLOCK_SIZE)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
