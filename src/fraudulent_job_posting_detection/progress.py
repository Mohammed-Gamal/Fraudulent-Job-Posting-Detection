"""Weighted, nestable progress reporting.

The interface shows one percentage for an action that is really several
unequal steps: loading a 50 MB CSV, cleaning 17,880 documents, fitting models
of very different cost. Counting steps gives a bar that sits at 25% for a
minute and then jumps to 100%, which tells the user nothing.

A `ProgressReporter` instead owns a [0, 1] span of the overall run and hands
out sub-spans sized by the share of the work each step is expected to take. A
step reports its own 0-to-1 completion inside its span and needs to know
nothing about the rest of the run, so a model's per-epoch progress rolls up
into the overall percentage without any step doing global arithmetic.

Two separate pieces of state make that work, and conflating them is the easy
mistake:

`_cursor`
    Per-reporter. Where the next child span starts, so siblings created by
    repeated `span()` calls lay out end to end instead of overlapping.

`_floor`
    Shared by every reporter in the tree. The highest fraction reported so
    far. A child beginning its span reports a value lower than its parent's
    cursor, so without a tree-wide floor the bar would visibly jump forward
    and then snap back -- which reads as a fault.

The weights are estimates, not measurements, so the bar is not perfectly
linear in time; it is monotonic and never stalls, which is what makes it
readable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from typing import Any

logger = logging.getLogger(__name__)

# A sink receives (fraction in [0, 1], human-readable label).
Sink = Callable[[float, str], None]


def null_sink(fraction: float, label: str) -> None:
    """Discard progress. Used when nobody is watching, e.g. from the CLI."""


class _Floor:
    """The tree-wide high-water mark, shared by reference between reporters."""

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = 0.0


class ProgressReporter:
    """Reports completion of a span of work as a fraction of the whole run."""

    __slots__ = ("_sink", "_start", "_end", "_label", "_cursor", "_floor")

    def __init__(
        self,
        sink: Sink | None = None,
        start: float = 0.0,
        end: float = 1.0,
        label: str = "",
        _floor: _Floor | None = None,
    ) -> None:
        self._sink = sink if sink is not None else null_sink
        self._start = float(start)
        self._end = max(float(end), float(start))
        self._label = label
        self._cursor = self._start
        self._floor = _floor if _floor is not None else _Floor()

    # -- reporting ----------------------------------------------------------
    def __call__(self, fraction: float, label: str | None = None) -> None:
        """Report `fraction` (0-1) complete *within this span*."""
        clamped = 0.0 if fraction < 0.0 else 1.0 if fraction > 1.0 else float(fraction)
        absolute = self._start + clamped * (self._end - self._start)
        if absolute < self._floor.value:
            absolute = self._floor.value
        self._floor.value = absolute
        self._sink(absolute, label if label is not None else self._label)

    def message(self, label: str) -> None:
        """Update the label without moving the bar."""
        self._sink(self._floor.value, label)

    def done(self, label: str | None = None) -> None:
        self(1.0, label)

    @property
    def fraction(self) -> float:
        """The overall fraction currently reported, for tests and logging."""
        return self._floor.value

    # -- composition --------------------------------------------------------
    def span(self, share: float, label: str = "") -> ProgressReporter:
        """Carve the next `share` (0-1) of this span off as a child reporter.

        Advances this reporter's cursor, so repeated calls lay the children out
        end to end. The child shares this reporter's monotonic floor.
        """
        width = self._end - self._start
        if share <= 0 or width <= 0:
            return ProgressReporter(
                self._sink, self._cursor, self._cursor, label, self._floor
            )
        child_start = self._cursor
        child_end = min(child_start + share * width, self._end)
        self._cursor = child_end
        return ProgressReporter(self._sink, child_start, child_end, label, self._floor)

    def weighted(self, weights: dict[str, float]) -> dict[str, ProgressReporter]:
        """Split this span into named children proportional to `weights`.

        >>> steps = reporter.weighted({"load": 1, "clean": 9})
        >>> steps["clean"](0.5, "halfway through cleaning")
        """
        total = sum(max(float(value), 0.0) for value in weights.values())
        if total <= 0:
            return {name: self.span(0.0, name) for name in weights}
        return {
            name: self.span(max(float(weight), 0.0) / total, name)
            for name, weight in weights.items()
        }

    def steps(self, count: int, label: str = "") -> list[ProgressReporter]:
        """Split this span into `count` equal children."""
        if count <= 0:
            return []
        return [self.span(1.0 / count, label) for _ in range(count)]

    def iterate(
        self,
        items: Iterable[Any],
        label_for: Callable[[int, Any], str] | None = None,
    ) -> Iterator[Any]:
        """Wrap an iterable, reporting progress as it is consumed.

        >>> for chunk in reporter.iterate(chunks, lambda i, c: f"chunk {i + 1}"):
        ...     process(chunk)
        """
        sequence = list(items)
        total = len(sequence)
        if total == 0:
            self.done()
            return
        for index, item in enumerate(sequence):
            label = label_for(index, item) if label_for is not None else None
            self(index / total, label)
            yield item
        self.done()
