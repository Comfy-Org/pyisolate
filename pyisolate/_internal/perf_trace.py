from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_TRACE_ENV = "PYISOLATE_TRACE_FILE"
_LOCK = threading.Lock()


def trace_path() -> str | None:
    path = os.environ.get(_TRACE_ENV)
    if not path:
        return None
    return path


def tracing_enabled() -> bool:
    return trace_path() is not None


def estimate_payload_bytes(payload: Any) -> int:
    try:
        encoded = json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        return len(encoded)
    except Exception:
        return 0


def record_event(event: dict[str, Any]) -> None:
    path = trace_path()
    if not path:
        return

    enriched = {
        "ts_ns": time.time_ns(),
        "pid": os.getpid(),
        **event,
    }

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(enriched, sort_keys=True))
        handle.write("\n")
