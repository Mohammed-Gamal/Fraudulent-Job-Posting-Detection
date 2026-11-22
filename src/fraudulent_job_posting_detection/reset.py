"""Clearing generated state, so the pipeline can be run again from scratch.

Everything this project produces is regenerable from the raw CSV: the cleaned
corpus, the fitted models, the saved results, the report tables and figures.
That makes a reset safe in principle -- but it is still deletion, so the rules
here are deliberately narrow.

What it will not do
-------------------
*Delete the raw dataset by default.* EMSCAD is not redistributable and cannot
be fetched without a Kaggle session, so it is the one artifact a user cannot
simply regenerate. Removing it is possible, but only when asked for
explicitly and separately from everything else.

*Delete anything outside the project's own output directories.* Each path is
resolved and checked to be inside the directory it was listed under, so a
symlink pointing elsewhere is skipped rather than followed.

*Delete the `.gitkeep` files.* They are what keeps the empty directories in
version control; removing them would quietly change the repository layout.

*Run while a job is running.* A worker thread part-way through writing a
corpus or a model would be left writing into a directory being emptied. The
caller is expected to refuse in that case.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config

logger = logging.getLogger(__name__)

# Files that are part of the repository layout, not generated output.
PROTECTED_NAMES = frozenset({".gitkeep", ".gitignore"})


@dataclass(frozen=True)
class Target:
    """One group of generated files the reset can remove."""

    key: str
    label: str
    description: str
    directory: Path
    patterns: tuple[str, ...] = ("*",)
    # Only removed when the caller explicitly opts in.
    guarded: bool = False


def targets() -> tuple[Target, ...]:
    """The removable groups.

    Built on each call rather than at import, because the config paths are
    redirected in tests and a module-level tuple would capture the real ones.
    """
    return (
        Target(
            key="corpus",
            label="Cleaned corpus",
            description="data/processed — corpus.parquet and its build record",
            directory=config.PROCESSED_DIR,
        ),
        Target(
            key="interim",
            label="Resumable results",
            description="data/interim — part-finished cross-validation runs",
            directory=config.INTERIM_DIR,
        ),
        Target(
            key="models",
            label="Trained models and results",
            description="models/ — fitted pipelines, training results, analyses",
            directory=config.MODELS_DIR,
        ),
        Target(
            key="figures",
            label="Report figures",
            description="reports/figures — generated PNGs",
            directory=config.FIGURES_DIR,
            patterns=("*.png",),
        ),
        Target(
            key="tables",
            label="Report tables",
            description="reports/tables — generated CSV and Markdown",
            directory=config.TABLES_DIR,
            patterns=("*.csv", "*.md"),
        ),
        Target(
            key="dataset",
            label="Imported dataset",
            description=(
                "data/raw — the EMSCAD CSV. Not redistributable: re-importing "
                "it needs the file again from Kaggle."
            ),
            directory=config.RAW_DIR,
            patterns=("*.csv",),
            guarded=True,
        ),
    )


def _removable(target: Target) -> list[Path]:
    """Files inside `target` that may be deleted, symlinks and keepers excluded."""
    directory = target.directory
    if not directory.exists():
        return []
    try:
        root = directory.resolve(strict=True)
    except OSError:
        return []

    found: list[Path] = []
    for pattern in target.patterns:
        for path in directory.glob(pattern):
            if path.name in PROTECTED_NAMES:
                continue
            if path.is_symlink():
                # Following one could delete something outside the project.
                logger.warning("skipping symlink %s", path)
                continue
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError):
                logger.warning("skipping %s: outside %s", path, root)
                continue
            found.append(path)
    return sorted(set(found))


def _size(path: Path) -> int:
    try:
        if path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return path.stat().st_size
    except OSError:
        return 0


def describe() -> list[dict[str, Any]]:
    """What a reset would remove, so the user can see it before confirming."""
    summary = []
    for target in targets():
        files = _removable(target)
        summary.append({
            "key": target.key,
            "label": target.label,
            "description": target.description,
            "guarded": target.guarded,
            "n_files": len(files),
            "bytes": sum(_size(path) for path in files),
            "examples": [path.name for path in files[:4]],
        })
    return summary


def clear_caches() -> None:
    """Drop every in-memory cache.

    Called after deleting files, and safe to call on its own: the caches key
    on file identity, so a deleted file would invalidate them anyway. Doing it
    explicitly means the next request cannot serve a frame read from a file
    that no longer exists.
    """
    from . import charts
    from .data import clear_cache as clear_dataset_cache
    from .models import clear_corpus_cache
    from .splits import clear_group_cache, clear_split_cache

    clear_dataset_cache()
    clear_corpus_cache()
    clear_group_cache()
    clear_split_cache()
    charts.clear_cache()


def perform(include_dataset: bool = False) -> dict[str, Any]:
    """Delete the generated artifacts and clear the caches.

    `include_dataset` is the only way to remove the raw CSV, and it is
    deliberately not implied by anything else.
    """
    removed: list[str] = []
    failed: list[dict[str, str]] = []
    freed = 0

    for target in targets():
        if target.guarded and not include_dataset:
            continue
        for path in _removable(target):
            size = _size(path)
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except OSError as exc:
                failed.append({"path": str(path), "error": str(exc)})
                logger.warning("could not remove %s: %s", path, exc)
                continue
            removed.append(str(path.relative_to(config.PROJECT_ROOT))
                           if _is_relative(path, config.PROJECT_ROOT) else path.name)
            freed += size

    clear_caches()
    # Recreate the directory tree, so the next run has somewhere to write and
    # the interface does not have to special-case a missing folder.
    config.ensure_directories()

    logger.info(
        "reset removed %s files (%.1f MB), dataset %s",
        len(removed), freed / 1_048_576,
        "removed" if include_dataset else "kept",
    )
    return {
        "removed": removed,
        "n_removed": len(removed),
        "bytes_freed": freed,
        "failed": failed,
        "dataset_removed": bool(include_dataset),
    }


def _is_relative(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
