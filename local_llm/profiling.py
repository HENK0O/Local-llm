from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class OperationTiming:
    calls: int = 0
    seconds: float = 0.0


@dataclass(frozen=True)
class ProfileEntry:
    operation: str
    calls: int
    seconds: float
    milliseconds_per_call: float
    percent: float


class OperationProfiler:
    """Low-overhead accumulator enabled explicitly by the profile command."""

    def __init__(self) -> None:
        self.timings: Dict[str, OperationTiming] = {}

    def record(self, operation: str, seconds: float) -> None:
        timing = self.timings.setdefault(operation, OperationTiming())
        timing.calls += 1
        timing.seconds += seconds

    @property
    def total_seconds(self) -> float:
        return sum(timing.seconds for timing in self.timings.values())

    def entries(self) -> list[ProfileEntry]:
        total = self.total_seconds
        entries = [
            ProfileEntry(
                operation=name,
                calls=timing.calls,
                seconds=timing.seconds,
                milliseconds_per_call=(timing.seconds * 1000.0 / timing.calls),
                percent=(timing.seconds / total * 100.0 if total else 0.0),
            )
            for name, timing in self.timings.items()
        ]
        return sorted(entries, key=lambda entry: entry.seconds, reverse=True)

    def to_dict(self) -> dict:
        return {
            "tracked_seconds": self.total_seconds,
            "operations": [asdict(entry) for entry in self.entries()],
        }
