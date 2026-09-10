"""Named corpora: one config file per corpus, one DB directory per corpus.

raggy's runtime config is a single YAML file, which is all the CLI needs. A GUI
holds several of them at once — one per "corpus" the user added — so this module
owns the small amount of bookkeeping that implies:

- where corpora live (``RAGGY_GUI_HOME``, defaulting to ``~/.raggy/gui``);
- one validated ``<id>.yaml`` per corpus, built from the shipped default config
  so a new corpus inherits sane pipeline settings rather than raggy's defaults
  drifting into the GUI;
- one ``registry.yaml`` naming the active corpus;
- **one ``db_directory`` per corpus, always.** Switching a corpus's ``sources``
  while it shares a DB with another corpus would diff two different file sets
  against one manifest and re-embed (or delete) the wrong chunks. The directory
  is derived from the corpus id here and is not user-editable through the GUI.

Annotations in this module are strings (PEP 563): ``CorpusStore.list`` is a
method that shadows the builtin inside the class body, so an eagerly evaluated
``list[str]`` in a method signature would be read against the method object.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import RaggySettings

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "registry.yaml"
CORPORA_DIRNAME = "corpora"

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "default_config" / "default_config.yaml"
)

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def gui_home() -> Path:
    """Root directory for GUI state (``RAGGY_GUI_HOME`` or ``~/.raggy/gui``)."""
    override = os.environ.get("RAGGY_GUI_HOME")
    return Path(override).expanduser() if override else Path.home() / ".raggy" / "gui"


def default_settings() -> dict[str, Any]:
    """The shipped default config as a plain dict.

    Every new corpus starts here and then overrides ``sources``/``db_directory``.
    Using the shipped file (rather than hard-coded values) keeps the GUI in step
    with whatever the project's tested defaults are.
    """
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def slugify(name: str) -> str:
    """Turn a display name into a filesystem-safe corpus id."""
    slug = _SLUG_STRIP.sub("-", name.strip().lower()).strip("-")
    return slug or "corpus"


@dataclass(frozen=True)
class Corpus:
    """A named corpus and where its pieces live."""

    id: str
    name: str
    config_path: Path
    db_directory: Path
    sources: list[str]
    created: str | None = None

    def as_dict(self, active: bool = False) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "config_path": str(self.config_path),
            "db_directory": str(self.db_directory),
            "sources": list(self.sources),
            "created": self.created,
            "active": active,
            "indexed": (self.db_directory / "manifest.yaml").exists(),
        }


class CorpusError(RuntimeError):
    """A corpus operation that cannot be carried out (bad id, bad folder, ...)."""


class CorpusStore:
    """Reads and writes corpora under :func:`gui_home`."""

    def __init__(self, home: Path | None = None):
        self.home = Path(home) if home is not None else gui_home()

    # -- locations ---------------------------------------------------------
    @property
    def corpora_dir(self) -> Path:
        return self.home / CORPORA_DIRNAME

    @property
    def registry_path(self) -> Path:
        return self.home / REGISTRY_FILENAME

    def config_path(self, corpus_id: str) -> Path:
        return self.corpora_dir / f"{corpus_id}.yaml"

    def db_path(self, corpus_id: str) -> Path:
        # One DB per corpus, kept inside the GUI home so the app owns its state
        # and never collides with a DB the CLI created from a hand-written config.
        return self.corpora_dir / f"{corpus_id}.db"

    # -- registry ----------------------------------------------------------
    def _read_registry(self) -> dict:
        if not self.registry_path.exists():
            return {}
        return yaml.safe_load(self.registry_path.read_text(encoding="utf-8")) or {}

    def _write_registry(self, registry: dict) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(
            yaml.safe_dump(registry, sort_keys=True), encoding="utf-8"
        )

    @property
    def active_id(self) -> str | None:
        active = self._read_registry().get("active")
        return str(active) if active else None

    def set_active(self, corpus_id: str) -> None:
        if self.get(corpus_id) is None:
            raise CorpusError(f"unknown corpus: {corpus_id}")
        registry = self._read_registry()
        registry["active"] = corpus_id
        self._write_registry(registry)

    # -- reading -----------------------------------------------------------
    def list(self) -> list[Corpus]:
        """Every corpus on disk, in stable id order."""
        if not self.corpora_dir.exists():
            return []
        corpora = []
        for path in sorted(self.corpora_dir.glob("*.yaml")):
            corpus = self._read(path)
            if corpus is not None:
                corpora.append(corpus)
        return corpora

    def get(self, corpus_id: str) -> Corpus | None:
        path = self.config_path(corpus_id)
        if not path.exists():
            return None
        return self._read(path)

    def _read(self, path: Path) -> Corpus | None:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as e:
            logger.warning("Ignoring unreadable corpus config '%s': %s", path, e)
            return None
        try:
            settings = RaggySettings(**{k: v for k, v in raw.items() if k != "name"})
        except Exception as e:  # noqa: BLE001 - a broken corpus must not break the list
            logger.warning("Ignoring invalid corpus config '%s': %s", path, e)
            return None
        corpus_id = path.stem
        return Corpus(
            id=corpus_id,
            name=str(raw.get("name") or corpus_id),
            config_path=path,
            db_directory=Path(settings.db_directory),
            sources=list(settings.sources),
            created=str(raw.get("created")) if raw.get("created") else None,
        )

    def active(self) -> Corpus | None:
        """The active corpus, falling back to the first one that exists."""
        active_id = self.active_id
        if active_id:
            corpus = self.get(active_id)
            if corpus is not None:
                return corpus
        corpora = self.list()
        return corpora[0] if corpora else None

    # -- writing -----------------------------------------------------------
    def unique_id(self, name: str) -> str:
        base = slugify(name)
        taken = {corpus.id for corpus in self.list()}
        if base not in taken:
            return base
        index = 2
        while f"{base}-{index}" in taken:
            index += 1
        return f"{base}-{index}"

    def create(
        self,
        name: str,
        sources: list[str],
        corpus_id: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> Corpus:
        """Write a new corpus config and return it.

        ``sources`` are stored resolved and absolute: a relative path in a corpus
        config would mean different files depending on the process's working
        directory, which is a footgun the CLI already documents and the GUI can
        simply avoid.
        """
        name = name.strip()
        if not name:
            raise CorpusError("a corpus needs a name")
        if not sources:
            raise CorpusError("a corpus needs at least one source folder or file")

        resolved = [_resolve_source(source) for source in sources]
        for source in resolved:
            if not Path(source).exists():
                raise CorpusError(f"source does not exist: {source}")

        corpus_id = corpus_id or self.unique_id(name)
        if self.config_path(corpus_id).exists():
            raise CorpusError(f"corpus already exists: {corpus_id}")

        source_for_db = resolved[0] if len(resolved) == 1 else None
        db_directory = self._db_directory_for(name, source_for_db, corpus_id)

        config = default_settings()
        config.pop("name", None)
        if settings:
            config.update(settings)
        config.update(
            {
                "name": name,
                "sources": resolved,
                "db_directory": str(db_directory),
            }
        )
        validated = RaggySettings(**{k: v for k, v in config.items() if k != "name"})
        _check_chunking(validated.model_dump())

        self.corpora_dir.mkdir(parents=True, exist_ok=True)
        self.config_path(corpus_id).write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        if self.active_id is None:
            self.set_active(corpus_id)
        return self.get(corpus_id)  # type: ignore[return-value]

    def _db_directory_for(self, name: str, source: str | None, corpus_id: str) -> Path:
        """Where a corpus's DB lives: always its own directory.

        Derived from the id, never from the source path, so two corpora pointed
        at the same folder still get separate manifests (and so re-adding a
        folder after removing a corpus cannot inherit that corpus's DB).
        """
        return self.db_path(corpus_id)

    def update_settings(self, corpus_id: str, settings: dict[str, Any]) -> Corpus:
        """Merge ``settings`` into a corpus config (used by the GUI's advanced panel).

        The file keeps the GUI's own keys (``name``, ``created``) alongside the
        library settings, so validation ignores exactly those — the same rule
        :func:`load_corpus_settings` applies.
        """
        corpus = self.get(corpus_id)
        if corpus is None:
            raise CorpusError(f"unknown corpus: {corpus_id}")
        raw = yaml.safe_load(corpus.config_path.read_text(encoding="utf-8")) or {}
        raw.update(settings)
        # Identity and isolation stay the GUI's business, not the caller's.
        raw["db_directory"] = str(corpus.db_directory)
        raw["sources"] = [
            _resolve_source(source) for source in raw.get("sources") or corpus.sources
        ]
        validated = RaggySettings(
            **{k: v for k, v in raw.items() if k not in _NON_SETTINGS}
        )
        _check_chunking(validated.model_dump())
        corpus.config_path.write_text(
            yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        return self.get(corpus_id)  # type: ignore[return-value]

    def add_source(self, corpus_id: str, source: str) -> Corpus:
        """Append a folder/file to a corpus and return the updated corpus."""
        corpus = self.get(corpus_id)
        if corpus is None:
            raise CorpusError(f"unknown corpus: {corpus_id}")
        resolved = _resolve_source(source)
        if not Path(resolved).exists():
            raise CorpusError(f"source does not exist: {resolved}")
        sources = list(corpus.sources)
        if resolved not in sources:
            sources.append(resolved)
        return self.update_settings(corpus_id, {"sources": sources})

    def remove(self, corpus_id: str, delete_db: bool = True) -> None:
        """Forget a corpus (and, by default, delete its DB directory)."""
        corpus = self.get(corpus_id)
        if corpus is None:
            raise CorpusError(f"unknown corpus: {corpus_id}")
        if delete_db and corpus.db_directory.exists():
            shutil.rmtree(corpus.db_directory, ignore_errors=True)
        corpus.config_path.unlink(missing_ok=True)

        registry = self._read_registry()
        if registry.get("active") == corpus_id:
            remaining = self.list()
            if remaining:
                registry["active"] = remaining[0].id
            else:
                registry.pop("active", None)
            self._write_registry(registry)


def _resolve_source(source: str) -> str:
    """Absolute, normalized path for a user-selected source."""
    return str(Path(source).expanduser().resolve())


def load_corpus_settings(
    corpus: Corpus, overrides: dict[str, Any] | None = None
) -> dict:
    """Validate and return a corpus's runtime settings, with optional overrides.

    The GUI metadata keys (``name``, ``created``) are stripped first: they belong
    to the corpus file but not to :class:`RaggySettings`, which rejects unknown
    fields. Everything else is validated exactly as the CLI validates its own
    config, so the GUI runs the settings the library would.

    ``sources`` and ``db_directory`` always come from the corpus object, never
    from the file's copy of them, so the DB isolation enforced at creation time
    holds for every run.
    """
    raw = yaml.safe_load(corpus.config_path.read_text(encoding="utf-8")) or {}
    settings = RaggySettings(
        **{key: value for key, value in raw.items() if key not in _NON_SETTINGS}
    ).model_dump()
    settings["db_directory"] = str(corpus.db_directory)
    settings["sources"] = list(corpus.sources)
    if overrides:
        settings.update(overrides)
        RaggySettings(**settings)
        _check_chunking(settings)
    return settings


def _check_chunking(settings: dict) -> None:
    """Reject a chunking config that cannot produce an index.

    The text splitter only complains at the first chunk — from langchain, mid-run,
    naming neither the file nor the corpus — so a GUI that lets a user change
    ``chunk_size`` has to catch this where the change is made.
    """
    if int(settings["chunk_overlap"]) >= int(settings["chunk_size"]):
        raise ValueError(
            f"chunk_overlap ({settings['chunk_overlap']}) must be smaller than "
            f"chunk_size ({settings['chunk_size']})"
        )


# Keys a corpus config carries for the GUI's benefit. They are not part of the
# library's settings schema, which rejects unknown fields outright.
_NON_SETTINGS = {"name", "created"}
