"""One explicit index run, with a per-file report of what it did.

The library entry points (:func:`raggy.indexing.initialize_db` and friends)
answer "is the DB current?" — a question the CLI asks on every startup, and one
whose answer is deliberately cheap and often "yes, nothing to do". A GUI needs
the other question answered: the user pressed *Refresh*, so what happened?

:func:`refresh_index` is that explicit run. It wraps ``initialize_db`` and then
reconciles three independent sources of truth into an
:class:`~raggy.indexing.IndexReport`:

- **the walk** — which files exist under ``sources``, and which extensions
  raggy has no loader for (skipped, not failed);
- **the load pass** — which files raised, and why (failed);
- **the DB itself** — how many chunks each file contributed, which is what
  makes "loaded but contributed nothing" visible rather than a silent success.

It also handles DOCX/PPTX → PDF conversion (see :mod:`raggy.render`), so those
formats are chunked from the rendered PDF and their citation page numbers
match what the viewer displays.
"""

import logging
from pathlib import Path

from .indexing import (
    IndexOutcome,
    IndexReport,
    chunk_counts,
    initialize_db,
)
from .loaders import (
    FileFailure,
    FileSkip,
    LoadReport,
    failure_reasons,
    file_failures,
    source_files,
)
from .progress import ProgressCallback
from .render import (
    RenderCache,
    converter_map,
    make_conversion_loader,
    make_text_caching_loader,
)

logger = logging.getLogger(__name__)


def refresh_index(
    db_directory: str,
    embedding_model: str,
    sources: list[str],
    chunk_size: int,
    chunk_overlap: int,
    embed_batch_size: int,
    progress: ProgressCallback | None = None,
    convert: bool = True,
    retry_failed: bool = True,
    allow_missing: bool = False,
    file_converters: dict[str, str] | None = None,
    loader=None,
) -> tuple[object, IndexReport]:
    """Index ``sources`` into ``db_directory`` and report what happened.

    ``convert`` enables the DOCX/PPTX → PDF pre-render step where LibreOffice is
    available; without it those formats are indexed from their native text
    extraction, as they always were. ``retry_failed`` re-reads files that failed
    on an earlier run (a user pressing Refresh wants the corpus retried, not
    just the changed parts). ``allow_missing`` tolerates a source entry that has
    disappeared, which is what raises otherwise.

    ``file_converters`` and ``loader`` override the conversion setup: what the
    manifest records as the converters in play, and how a convertible file is
    read. They exist so a test (or a caller with its own converter) can pin both
    without LibreOffice installed; by default they are whatever
    :mod:`raggy.render` finds on this machine.

    Returns the vector store and the report. The store is returned so the caller
    keeps the connection it should close before the next rebuild.
    """
    render_cache: RenderCache | None = None
    if file_converters is None:
        converters: dict[str, str] = {}
        if convert:
            render_cache = RenderCache(db_directory)
            converters = converter_map()
            if converters and loader is None:
                loader = make_conversion_loader(render_cache)
            elif not converters:
                logger.info(
                    "No document converter available; DOCX/PPTX will be indexed "
                    "from their native text extraction."
                )
    else:
        converters = dict(file_converters)

    # Whatever the conversion setup decided, the text each image loader extracts
    # is cached as it is extracted — otherwise the OCR behind an image would run
    # again the first time a viewer asks to show that text (see
    # :class:`raggy.render.ExtractedTextCache`).
    loader = make_text_caching_loader(db_directory, loader)

    # Checked here rather than left to initialize_db: an entry that no longer
    # exists raises from the walk, and it should raise before any model or
    # converter work, not halfway through fingerprinting.
    missing = [source for source in sources if not Path(source).exists()]
    if missing and not allow_missing:
        raise FileNotFoundError(
            "Source document(s) not found at: "
            + ", ".join(str(path) for path in missing)
        )

    load_report = LoadReport()
    for path in missing:
        load_report.skipped.append(
            FileSkip(path=str(path), reason="missing", detail="")
        )
        logger.warning("Ignoring source '%s': it no longer exists.", path)
    outcome = initialize_db(
        db_directory=db_directory,
        embedding_model=embedding_model,
        sources=sources,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_batch_size=embed_batch_size,
        progress=progress,
        report=load_report,
        loader=loader,
        file_converters=converters,
        retry_failed=retry_failed,
        on_missing="skip" if allow_missing else "raise",
    )

    return outcome.vectorstore, reconcile(outcome, load_report, sources)


def reconcile(
    outcome: IndexOutcome, report: LoadReport, sources: list[str] | None = None
) -> IndexReport:
    """Turn one index run into a per-file :class:`IndexReport`.

    Split out from :func:`refresh_index` so it can be exercised (and tested)
    without an embedding model: everything it needs is the store's contents, the
    load report, the walk, and the outcome the run itself produced.

    The run's own plan is the source of truth for what changed — reading it back
    out of the manifest would only describe the state *after* the write, in
    which nothing has changed by definition. ``sources`` is re-walked here
    because a run with nothing to do never reads a file, so the load report
    alone cannot say which files were left untouched.
    """
    attempted, failed = file_failures(report)
    vectorstore = outcome.vectorstore

    walked = set(attempted) | {failure.path for failure in report.failed}
    if sources:
        walked |= {str(path) for path in source_files(sources, on_missing="skip")}

    failed_paths = set(failed)
    counts = chunk_counts(vectorstore)
    stored = sorted(counts)
    stored_set = set(stored)
    # Only files this run actually read can have been (re-)embedded. On a no-op
    # run nothing was read, so ``indexed`` is empty — which is precisely what
    # "nothing to do" means.
    indexed = sorted((set(attempted) - failed_paths) & stored_set)
    # "unchanged" is the files the run left alone, minus whatever is not in the
    # DB any more (deleted). Unsupported extensions never enter the walk at all.
    untouched = walked - set(attempted) - {f.path for f in report.failed}

    return IndexReport(
        # ``changed`` describes the corpus, not the run: a file that is still
        # unreadable after a retry has not changed anything, so a caller that
        # wants "did this run do work" reads ``indexed`` instead.
        full_rebuild=outcome.full_rebuild,
        changed=outcome.plan.has_changes,
        indexed=indexed,
        stored=stored,
        unchanged=sorted(path for path in untouched if path in stored_set),
        failed=_failures(report, failed),
        skipped=list(report.skipped),
        removed=list(outcome.plan.removed),
        chunks={path: counts[path] for path in stored},
        orphans=sorted(stored_set - walked),
    )


def _failures(report: LoadReport, failed: dict[str, str]) -> list[FileFailure]:
    """The run's failures: what raised, plus what loaded but yielded no text."""
    seen: dict[str, str] = {}
    for path, reason in failure_reasons(report).items():
        seen.setdefault(path, reason)
    for path in failed:
        seen.setdefault(path, "no text could be extracted from this file")
    return [FileFailure(path=path, error=reason) for path, reason in seen.items()]
