"""A minimal background-job runner.

Preprocessing takes ~30s and training four models takes ~60s. Holding an HTTP
request open that long is fragile, so long actions run on a worker thread and
the page polls /job/<name> for progress.

This is deliberately tiny: an in-memory dict and one thread per job. It
assumes a single user on a single process, which is exactly how this app is
meant to be run (`fjpd serve` on your own machine). It is not suitable for a
multi-worker production deployment.
"""

from __future__ import annotations

import logging
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """State of one background task."""

    status: str = "idle"          # idle | running | done | error
    label: str = ""               # human-readable current step
    done: int = 0                 # completed units of work
    total: int = 0                # total units of work
    result: Any = None
    error: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def as_dict(self) -> dict[str, Any]:
        percent = int(self.done / self.total * 100) if self.total else 0
        return {
            "status": self.status,
            "label": self.label,
            "done": self.done,
            "total": self.total,
            "percent": percent,
            "error": self.error,
        }


_JOBS: dict[str, Job] = {}
_REGISTRY_LOCK = threading.Lock()


def get(name: str) -> Job:
    with _REGISTRY_LOCK:
        return _JOBS.setdefault(name, Job())


def start(name: str, target: Callable[[Callable[[int, int, str], None]], Any]) -> bool:
    """Run `target` on a worker thread. Returns False if that job is already running.

    `target` receives a `progress(done, total, label)` callback it may call to
    report where it has got to.
    """
    job = get(name)
    with job.lock:
        if job.status == "running":
            return False
        job.status, job.label = "running", "starting"
        job.done = job.total = 0
        job.result, job.error = None, ""

    def progress(done: int, total: int, label: str) -> None:
        job.done, job.total, job.label = done, total, label

    def run() -> None:
        try:
            job.result = target(progress)
            job.status, job.label = "done", "complete"
        except Exception as exc:  # surfaced to the page, not swallowed
            logger.exception("job %s failed", name)
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.label = "failed"
            logger.debug(traceback.format_exc())

    threading.Thread(target=run, name=f"job-{name}", daemon=True).start()
    return True
