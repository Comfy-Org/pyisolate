import json
import sys
from typing import Any

import pytest

from pyisolate._internal import bootstrap


def test_apply_sys_path_merges_and_dedup(monkeypatch: Any, tmp_path: Any) -> None:
    original = list(sys.path)
    snapshot = {
        "sys_path": [str(tmp_path), str(tmp_path)],
        "additional_paths": [str(tmp_path / "extra")],
        "preferred_root": None,
    }
    (tmp_path / "extra").mkdir(parents=True, exist_ok=True)
    bootstrap._apply_sys_path(snapshot)
    assert sys.path[0] == str(tmp_path)
    assert sys.path[1] == str(tmp_path / "extra")
    sys.path[:] = original


def test_bootstrap_child_missing_snapshot_returns_none(monkeypatch: Any) -> None:
    monkeypatch.delenv("PYISOLATE_HOST_SNAPSHOT", raising=False)
    assert bootstrap.bootstrap_child() is None


def test_bootstrap_child_json_payload_adapter_none(monkeypatch: Any) -> None:
    payload = json.dumps(
        {
            "sys_path": [],
            "adapter_ref": "demo:Adapter",
        }
    )
    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", payload)
    # Simulate failed rehydration
    monkeypatch.setattr(
        bootstrap, "_rehydrate_adapter", lambda name: (_ for _ in ()).throw(ValueError("failed"))
    )
    with pytest.raises(ValueError):
        bootstrap.bootstrap_child()


def test_bootstrap_child_snapshot_file_errors(tmp_path: Any, monkeypatch: Any) -> None:
    snap_path = tmp_path / "bad.json"
    snap_path.write_text("not-json")
    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", str(snap_path))
    with pytest.raises(ValueError):
        bootstrap.bootstrap_child()


def test_bootstrap_child_missing_file_graceful(tmp_path: Any, monkeypatch: Any) -> None:
    snap_path = tmp_path / "missing.json"
    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", str(snap_path))
    assert bootstrap.bootstrap_child() is None
