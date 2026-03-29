"""Child-process bootstrap for PyIsolate.

This module resolves the "config before path" paradox by applying the host's
snapshot (sys.path + adapter metadata) before any heavy imports occur in the
child process.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, cast

from ..interfaces import IsolationAdapter
from ..path_helpers import build_child_sys_path
from .serialization_registry import SerializerRegistry

logger = logging.getLogger(__name__)


def _should_apply_host_sys_path(snapshot: dict[str, Any]) -> bool:
    return bool(snapshot.get("apply_host_sys_path", True))


def _merge_sys_path_front(paths: list[str]) -> None:
    """Prepend paths to sys.path while preserving order and removing duplicates."""
    seen = set()
    merged: list[str] = []

    def add_path(p: str) -> None:
        norm = os.path.normcase(os.path.abspath(p))
        if norm in seen:
            return
        seen.add(norm)
        merged.append(p)

    for p in paths:
        add_path(p)

    for p in sys.path:
        add_path(p)

    sys.path[:] = merged


def _apply_sealed_opt_in_paths(snapshot: dict[str, Any]) -> None:
    raw_paths = snapshot.get("sealed_host_ro_paths", [])
    if not isinstance(raw_paths, list):
        return

    opt_in_paths: list[str] = []
    for path in raw_paths:
        if not isinstance(path, str) or not path.strip():
            continue
        if not os.path.isabs(path):
            continue
        if not os.path.exists(path):
            continue
        opt_in_paths.append(path)

    if not opt_in_paths:
        return

    _merge_sys_path_front(opt_in_paths)
    logger.debug("Applied %d sealed opt-in import paths", len(opt_in_paths))


def _apply_sys_path(snapshot: dict[str, Any]) -> None:
    if not _should_apply_host_sys_path(snapshot):
        _apply_sealed_opt_in_paths(snapshot)
        logger.debug("Skipping host sys.path reconstruction for sealed child")
        return

    host_paths = snapshot.get("sys_path", [])
    extra_paths = snapshot.get("additional_paths", [])

    preferred_root: str | None = snapshot.get("preferred_root")
    if not preferred_root:
        context_data = snapshot.get("context_data", {})
        module_path = context_data.get("module_path") or os.environ.get("PYISOLATE_MODULE_PATH")
        if module_path:
            preferred_root = str(Path(module_path).parent.parent)

    filtered_subdirs = snapshot.get("filtered_subdirs")
    child_paths = build_child_sys_path(host_paths, extra_paths, preferred_root, filtered_subdirs)

    if not child_paths:
        return

    # Rebuild sys.path with child paths first while preserving any existing entries
    # that are not already in the computed set.
    _merge_sys_path_front(child_paths)
    logger.debug("Applied %d paths from snapshot (preferred_root=%s)", len(child_paths), preferred_root)


def _rehydrate_adapter(start_ref: str) -> IsolationAdapter:
    """Import and instantiate adapter from string reference."""
    import importlib

    from .adapter_registry import AdapterRegistry

    try:
        module_path, class_name = start_ref.split(":", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        # Instantiate and register immediately
        adapter = cls()

        # KEY STEP: Register in child's memory space so subsequent calls work
        AdapterRegistry.register(adapter)

        return cast(IsolationAdapter, adapter)
    except Exception as exc:
        raise ValueError(f"Failed to rehydrate adapter '{start_ref}': {exc}") from exc


def bootstrap_child() -> IsolationAdapter | None:
    """Initialize child environment using host snapshot.

    Returns:
        The loaded adapter instance, or None if no snapshot/adapter present.

    Raises:
        ValueError: If snapshot is malformed or adapter cannot be loaded.
    """
    snapshot_env = os.environ.get("PYISOLATE_HOST_SNAPSHOT")
    if not snapshot_env:
        logger.debug("No PYISOLATE_HOST_SNAPSHOT set; skipping bootstrap")
        return None

    snapshot: dict[str, Any]

    # PYISOLATE_HOST_SNAPSHOT may be either a JSON string or a file path.
    # If it starts with '{', assume it's a JSON payload.
    if snapshot_env.strip().startswith("{"):
        looks_like_path = False
    else:
        looks_like_path = os.path.sep in snapshot_env or snapshot_env.endswith(".json")

    if looks_like_path:
        try:
            with open(snapshot_env, encoding="utf-8") as fh:
                snapshot_text = fh.read()
        except FileNotFoundError:
            logger.debug("Snapshot path missing (%s); skipping bootstrap", snapshot_env)
            return None

        try:
            snapshot = json.loads(snapshot_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to decode snapshot file {snapshot_env}: {exc}") from exc
    else:
        try:
            snapshot = json.loads(snapshot_env)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to decode PYISOLATE_HOST_SNAPSHOT: {exc}") from exc

    _apply_sys_path(snapshot)

    adapter: IsolationAdapter | None = None
    is_sealed = not _should_apply_host_sys_path(snapshot)

    adapter_ref = snapshot.get("adapter_ref")
    if adapter_ref:
        try:
            adapter = _rehydrate_adapter(adapter_ref)
        except Exception as exc:
            logger.warning(
                "Failed to rehydrate adapter from ref %s: %s",
                adapter_ref,
                exc,
            )

    if not adapter and adapter_ref and not is_sealed:
        raise ValueError("Snapshot contained adapter info but adapter could not be loaded")

    if adapter:
        if not is_sealed:
            adapter.setup_child_environment(snapshot)
        registry = SerializerRegistry.get_instance()
        adapter.register_serializers(registry)

    return adapter
