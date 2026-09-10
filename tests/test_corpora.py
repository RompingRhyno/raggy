"""Named corpora: config files, DB isolation, and the active-corpus registry."""

import pytest
import yaml
from pydantic import ValidationError

from raggy.corpora import (
    CorpusError,
    CorpusStore,
    default_settings,
    load_corpus_settings,
    slugify,
)


@pytest.fixture
def docs_dir(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("Alpha content", encoding="utf-8")
    return folder


@pytest.fixture
def store(tmp_path):
    return CorpusStore(tmp_path / "gui")


def test_create_writes_a_valid_config_that_the_library_can_load(store, docs_dir):
    corpus = store.create("My Papers", [str(docs_dir)])

    assert corpus.id == "my-papers"
    assert corpus.name == "My Papers"
    assert corpus.sources == [str(docs_dir.resolve())]

    settings = load_corpus_settings(corpus)
    assert settings["sources"] == [str(docs_dir.resolve())]
    assert settings["db_directory"] == str(corpus.db_directory)
    # Inherited pipeline settings come from the shipped default config, not from
    # hard-coded values that could drift away from it.
    defaults = default_settings()
    assert settings["chunk_size"] == defaults["chunk_size"]
    assert settings["embedding_model"] == defaults["embedding_model"]
    assert settings["rerank_k"] == defaults["rerank_k"]


def test_each_corpus_gets_its_own_db_directory(store, docs_dir, tmp_path):
    other = tmp_path / "other"
    other.mkdir()

    first = store.create("First", [str(docs_dir)])
    second = store.create("Second", [str(docs_dir)])

    assert first.db_directory != second.db_directory
    # Pointing two corpora at the same folder must still isolate their manifests.
    assert first.sources == second.sources


def test_first_corpus_created_becomes_active_and_switching_persists(store, docs_dir):
    first = store.create("First", [str(docs_dir)])
    second = store.create("Second", [str(docs_dir)])

    assert store.active_id == first.id
    store.set_active(second.id)

    # A fresh store reading the same home sees the same active corpus.
    assert CorpusStore(store.home).active_id == second.id


def test_active_falls_back_to_the_first_corpus_when_the_registry_is_stale(
    store, docs_dir
):
    corpus = store.create("First", [str(docs_dir)])
    store.registry_path.write_text(
        yaml.safe_dump({"active": "deleted-corpus"}), encoding="utf-8"
    )

    assert store.active().id == corpus.id


def test_create_rejects_a_missing_source(store, tmp_path):
    with pytest.raises(CorpusError, match="does not exist"):
        store.create("Empty", [str(tmp_path / "nope")])


def test_create_rejects_a_blank_name(store, docs_dir):
    with pytest.raises(CorpusError, match="name"):
        store.create("   ", [str(docs_dir)])


def test_create_rejects_an_empty_source_list(store):
    with pytest.raises(CorpusError, match="source"):
        store.create("Nothing", [])


def test_ids_are_slugged_and_deduplicated(store, docs_dir):
    first = store.create("Papers!", [str(docs_dir)])
    second = store.create("papers", [str(docs_dir)])

    assert first.id == "papers"
    assert second.id == "papers-2"
    assert slugify("  Weird / Name  ") == "weird-name"


def test_broken_config_is_listed_around_rather_than_crashing(tmp_path, docs_dir):
    store = CorpusStore(tmp_path / "gui")
    good = store.create("Good", [str(docs_dir)])
    (store.corpora_dir / "broken.yaml").write_text(
        yaml.safe_dump({"sources": [], "db_directory": "x"}), encoding="utf-8"
    )

    assert [corpus.id for corpus in store.list()] == [good.id]


def test_update_settings_keeps_identity_and_db_isolation(store, docs_dir):
    corpus = store.create("First", [str(docs_dir)])

    updated = store.update_settings(
        corpus.id, {"chunk_size": 777, "db_directory": "/tmp/evil"}
    )

    assert updated.sources == corpus.sources
    # db_directory is the GUI's business: a caller cannot point a corpus at a
    # shared DB and reintroduce cross-corpus churn.
    assert updated.db_directory == corpus.db_directory
    assert load_corpus_settings(updated)["chunk_size"] == 777


def test_update_settings_rejects_an_invalid_value(store, docs_dir):
    corpus = store.create("First", [str(docs_dir)])

    with pytest.raises(ValidationError):
        store.update_settings(corpus.id, {"chunk_size": -5})


def test_add_source_appends_once(store, docs_dir, tmp_path):
    extra = tmp_path / "extra"
    extra.mkdir()
    corpus = store.create("First", [str(docs_dir)])

    updated = store.add_source(corpus.id, str(extra))
    again = store.add_source(corpus.id, str(extra))

    assert updated.sources == [str(docs_dir.resolve()), str(extra.resolve())]
    assert again.sources == updated.sources


def test_remove_deletes_config_and_db_and_moves_active(store, docs_dir):
    first = store.create("First", [str(docs_dir)])
    second = store.create("Second", [str(docs_dir)])
    first.db_directory.mkdir(parents=True, exist_ok=True)
    (first.db_directory / "manifest.yaml").write_text("files: {}", encoding="utf-8")
    store.set_active(first.id)

    store.remove(first.id)

    assert not first.config_path.exists()
    assert not first.db_directory.exists()
    assert store.active_id == second.id


def test_remove_clears_active_when_it_was_the_last_corpus(store, docs_dir):
    corpus = store.create("Only", [str(docs_dir)])

    store.remove(corpus.id)

    assert store.list() == []
    assert store.active_id is None


def test_set_active_rejects_an_unknown_corpus(store):
    with pytest.raises(CorpusError, match="unknown corpus"):
        store.set_active("ghost")


def test_home_can_be_overridden_by_the_environment(tmp_path, monkeypatch):
    from raggy.corpora import gui_home

    monkeypatch.setenv("RAGGY_GUI_HOME", str(tmp_path / "custom"))

    assert gui_home() == tmp_path / "custom"
