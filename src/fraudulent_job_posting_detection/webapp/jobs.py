"""A small background-job runner for the single-page interface.

Cleaning takes ~30s and training several models takes minutes. Holding an HTTP
request open that long is fragile, so long actions run on a worker thread and
the page polls for progress.

This is deliberately small -- an in-memory registry and one thread per job --
and assumes a single user in a single process, which is how `fjpd serve` is
meant to be run. It is not suitable for a multi-worker deployment, where the
registry would differ per worker and polling would hit the wrong one.

What it does take seriously
---------------------------
*Mutual exclusion across jobs.* Cleaning and training both load the corpus and
both draw matplotlib figures, and pyplot is global state. The previous version
guarded only against starting the *same* job twice, so cleaning and training
could run at once, fight over memory, and interleave their figures. A single
global slot now means at most one long action is ever in flight.

*Reading state atomically.* Progress is written by the worker and read by the
poller. Every field is read under the job's lock into one snapshot, so the page
can never show a percentage from one moment with a label from another.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# A task receives a progress reporter and returns a JSON-serialisable result.
Task = Callable[[Any], Any]

STATUS_IDLE = "idle"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"


@dataclass
class Job:
    """State of one background task."""

    name: str
    status: str = STATUS_IDLE
    label: str = ""
    fraction: float = 0.0
    result: Any = None
    error: str = ""
    run_id: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    cancel_requested: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self, include_result: bool = False) -> dict[str, Any]:
        """A consistent view of this job, taken under the lock."""
        with self.lock:
            elapsed = (
                (self.finished_at or time.monotonic()) - self.started_at
                if self.started_at
                else 0.0
            )
            payload: dict[str, Any] = {
                "name": self.name,
                "status": self.status,
                "label": self.label,
                "percent": round(max(0.0, min(self.fraction, 1.0)) * 100, 1),
                "error": self.error,
                "run_id": self.run_id,
                "elapsed_seconds": round(elapsed, 1),
                "cancel_requested": self.cancel_requested,
            }
            if include_result:
                payload["result"] = self.result
            return payload


class JobCancelled(RuntimeError):
    """Raised inside a task when the user asked it to stop."""


class _Registry:
    """Holds every job and enforces that only one runs at a time."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._active: str | None = None

    def get(self, name: str) -> Job:
        with self._lock:
            job = self._jobs.get(name)
            if job is None:
                job = Job(name=name)
                self._jobs[name] = job
            return job

    def active(self) -> str | None:
        with self._lock:
            return self._active

    def all_snapshots(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        return {job.name: job.snapshot() for job in jobs}

    def start(self, name: str, task: Task) -> tuple[bool, str]:
        """Launch `task` on a worker thread.

        Returns (started, message). Refuses when any job is already running,
        because these jobs contend for the corpus, for memory and for
        matplotlib's global state.
        """
        job = self.get(name)
        with self._lock:
            if self._active is not None:
                busy = self._active
                return False, (
                    f"'{busy}' is already running. Wait for it to finish, "
                    "or cancel it first."
                )
            self._active = name

        run_id = uuid.uuid4().hex[:12]
        with job.lock:
            job.status = STATUS_RUNNING
            job.label = "starting"
            job.fraction = 0.0
            job.result = None
            job.error = ""
            job.run_id = run_id
            job.started_at = time.monotonic()
            job.finished_at = 0.0
            job.cancel_requested = False

        def report(fraction: float, label: str) -> None:
            with job.lock:
                # Guard against a stale reporter from a previous run writing
                # into the current one.
                if job.run_id != run_id:
                    return
                job.fraction = fraction
                job.label = label
                cancelled = job.cancel_requested
            if cancelled:
                raise JobCancelled(f"{name} cancelled")

        def run() -> None:
            from ..progress import ProgressReporter

            reporter = ProgressReporter(report)
            try:
                result = task(reporter)
                with job.lock:
                    job.result = result
                    job.status = STATUS_DONE
                    job.label = "complete"
                    job.fraction = 1.0
            except JobCancelled:
                logger.info("job %s cancelled by the user", name)
                with job.lock:
                    job.status = STATUS_CANCELLED
                    job.label = "cancelled"
                    job.error = ""
            except Exception as exc:
                logger.exception("job %s failed", name)
                with job.lock:
                    job.status = STATUS_ERROR
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.label = "failed"
            finally:
                with job.lock:
                    job.finished_at = time.monotonic()
                with self._lock:
                    if self._active == name:
                        self._active = None

        threading.Thread(target=run, name=f"job-{name}", daemon=True).start()
        return True, f"{name} started"

    def cancel(self, name: str) -> bool:
        """Ask a running job to stop at its next progress report.

        Cooperative, not forced: a task is interrupted where it reports
        progress, so it cannot be left holding a half-written file. A step that
        never reports runs to completion.
        """
        job = self.get(name)
        with job.lock:
            if job.status != STATUS_RUNNING:
                return False
            job.cancel_requested = True
            job.label = "cancelling…"
            return True


_REGISTRY = _Registry()


def get(name: str) -> Job:
    return _REGISTRY.get(name)


def start(name: str, task: Task) -> tuple[bool, str]:
    return _REGISTRY.start(name, task)


def cancel(name: str) -> bool:
    return _REGISTRY.cancel(name)


def active() -> str | None:
    return _REGISTRY.active()


def snapshots() -> dict[str, dict[str, Any]]:
    return _REGISTRY.all_snapshots()


def reset_registry() -> None:
    """Forget every job.

    Used after a reset: leaving "training: done" beside a models directory
    that no longer holds a model would describe a state that is gone.
    Refuses while something is running, so a live worker is never orphaned
    from the registry that reports it.
    """
    global _REGISTRY
    if _REGISTRY.active() is not None:
        logger.warning("not clearing the job registry while a job is running")
        return
    _REGISTRY = _Registry()
