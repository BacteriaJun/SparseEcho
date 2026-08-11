from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class NullTelemetrySink:
    def emit(self, event: dict) -> None:
        return None


class JsonlTelemetrySink:
    """Append-only deployment journal with one JSON object per event."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event: dict) -> None:
        record = {"wall_time_ns": time.time_ns(), **event}
        line = json.dumps(record, separators=(",", ":"), sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
