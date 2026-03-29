"""Utilities for sharing host path context with PyIsolate children.

``serialize_host_snapshot`` captures host ``sys.path`` and select environment variables.
``build_child_sys_path`` reconstructs ``sys.path`` in children with an optional preferred
root first and removes code subdirectories that would shadow imports.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

_DEFAULT_ENV_KEYS = (
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "HF_HUB_DISABLE_TELEMETRY",
    "DO_NOT_TRACK",
)


def serialize_host_snapshot(
    output_path: str | os.PathLike[str] | None = None,
    extra_env_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Capture the host interpreter context for use by child processes.

    Persisting the snapshot is optional; when provided, ``output_path`` will contain
    a JSON payload with ``sys.path``, the Python executable, and selected env vars.
    """
    env_keys = list(_DEFAULT_ENV_KEYS)
    if extra_env_keys:
        env_keys.extend(extra_env_keys)

    env = {key: os.environ[key] for key in env_keys if key in os.environ}

    snapshot: dict[str, Any] = {
        "sys_path": list(sys.path),
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "environment": env,
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        snapshot["snapshot_path"] = str(path)

    return snapshot


def build_child_sys_path(
    host_paths: Sequence[str],
    extra_paths: Sequence[str],
    preferred_root: str | None = None,
    filtered_subdirs: Sequence[str] | None = None,
) -> list[str]:
    """Construct ``sys.path`` for an isolated child interpreter.

    Host paths retain order, an optional preferred root is prepended, and child
    venv site-packages are appended while avoiding duplicates. When
    ``filtered_subdirs`` is provided, those named subdirectories of
    ``preferred_root`` are excluded from the reconstructed path. When
    ``filtered_subdirs`` is ``None``, no subdirectory filtering is applied.
    The caller (typically an :class:`~pyisolate.interfaces.IsolationAdapter`)
    is responsible for supplying the appropriate list via
    ``get_path_config()["filtered_subdirs"]``.
    """

    def _norm(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    result: list[str] = []
    seen: set[str] = set()

    ordered_host = list(host_paths)
    if preferred_root:
        root_norm = _norm(preferred_root)
        code_subdirs: set[str] = set()
        if filtered_subdirs is not None:
            code_subdirs = {os.path.join(root_norm, name) for name in filtered_subdirs}
        filtered_host = []
        for p in ordered_host:
            p_norm = _norm(p)
            if p_norm == root_norm:
                continue
            if p_norm in code_subdirs:
                continue
            filtered_host.append(p)
        ordered_host = [preferred_root] + filtered_host

    def add_path(path: str) -> None:
        if not path:
            return
        key = _norm(path)
        if key in seen:
            return
        seen.add(key)
        result.append(path)

    for path in ordered_host:
        add_path(path)

    for path in extra_paths:
        add_path(path)

    return result
