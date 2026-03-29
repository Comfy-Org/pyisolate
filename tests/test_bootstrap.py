import json
import sys
from importlib import import_module

import pytest

from pyisolate._internal import bootstrap
from pyisolate._internal.serialization_registry import SerializerRegistry


class FakeAdapter:
    identifier = "fake"

    def __init__(self):
        self.setup_called = False
        self.registry_used = False

    def get_path_config(self, module_path):
        return None

    def setup_child_environment(self, snapshot):
        self.setup_called = True

    def register_serializers(self, registry):
        self.registry_used = True
        registry.register("FakeType", lambda x: {"v": x}, lambda x: x["v"])

    def provide_rpc_services(self):
        return []

    def handle_api_registration(self, api, rpc):
        return None


@pytest.fixture(autouse=True)
def clear_registry():
    registry = SerializerRegistry.get_instance()
    registry.clear()
    yield
    registry.clear()


def test_bootstrap_applies_snapshot(monkeypatch, tmp_path):
    fake_adapter = FakeAdapter()
    monkeypatch.setattr(bootstrap, "_rehydrate_adapter", lambda name: fake_adapter)

    snapshot = {
        "sys_path": [str(tmp_path / "foo")],
        "adapter_ref": "fake:FakeAdapter",
    }
    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", json.dumps(snapshot))

    original_sys_path = list(sys.path)
    try:
        adapter = bootstrap.bootstrap_child()
        updated_sys_path = list(sys.path)
    finally:
        sys.path[:] = original_sys_path

    assert adapter is fake_adapter
    assert fake_adapter.setup_called
    assert fake_adapter.registry_used
    assert snapshot["sys_path"][0] in updated_sys_path

    registry = SerializerRegistry.get_instance()
    assert registry.has_handler("FakeType")


def test_bootstrap_no_snapshot(monkeypatch):
    monkeypatch.delenv("PYISOLATE_HOST_SNAPSHOT", raising=False)
    assert bootstrap.bootstrap_child() is None


def test_bootstrap_bad_json(monkeypatch):
    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", "not-json")
    with pytest.raises(ValueError):
        bootstrap.bootstrap_child()


def test_bootstrap_missing_adapter(monkeypatch):
    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", json.dumps({"adapter_ref": "missing"}))
    monkeypatch.setattr(
        bootstrap, "_rehydrate_adapter", lambda name: (_ for _ in ()).throw(ValueError("nope"))
    )
    with pytest.raises(ValueError):
        bootstrap.bootstrap_child()


def test_bootstrap_skips_host_sys_path_for_sealed_worker(monkeypatch, tmp_path):
    host_only_path = str(tmp_path / "host_only")
    snapshot = {
        "sys_path": [host_only_path],
        "adapter_ref": "fake:FakeAdapter",
        "apply_host_sys_path": False,
    }
    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", json.dumps(snapshot))

    original_sys_path = list(sys.path)
    try:
        adapter = bootstrap.bootstrap_child()
        updated_sys_path = list(sys.path)
    finally:
        sys.path[:] = original_sys_path

    assert adapter is None
    assert host_only_path not in updated_sys_path


def test_bootstrap_sealed_worker_skips_adapter_rehydration(monkeypatch):
    snapshot = {
        "adapter_ref": "bad.module:BadClass",
        "apply_host_sys_path": False,
    }
    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", json.dumps(snapshot))
    monkeypatch.setattr(
        bootstrap, "_rehydrate_adapter", lambda name: (_ for _ in ()).throw(ValueError("should not run"))
    )

    assert bootstrap.bootstrap_child() is None


def test_sealed_worker_host_policy_ro_paths_enable_import_without_host_sys_path(monkeypatch, tmp_path):
    module_name = "sealed_opt_in_visible_module"
    module_root = tmp_path / "opt_in_root"
    module_root.mkdir(parents=True, exist_ok=True)
    (module_root / f"{module_name}.py").write_text("VALUE = 42\n", encoding="utf-8")

    snapshot = {
        "sys_path": [],
        "apply_host_sys_path": False,
        "sealed_host_ro_paths": [str(module_root)],
    }
    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", json.dumps(snapshot))

    original_sys_path = list(sys.path)
    try:
        bootstrap.bootstrap_child()
        imported = import_module(module_name)
    finally:
        sys.path[:] = original_sys_path
        sys.modules.pop(module_name, None)

    assert imported.VALUE == 42


def test_sealed_worker_without_opt_in_still_cannot_import_module(monkeypatch, tmp_path):
    module_name = "sealed_no_opt_in_hidden_module"
    blocked_root = tmp_path / "blocked_root"
    blocked_root.mkdir(parents=True, exist_ok=True)
    (blocked_root / f"{module_name}.py").write_text("VALUE = 7\n", encoding="utf-8")

    snapshot = {
        "sys_path": [str(blocked_root)],
        "apply_host_sys_path": False,
    }
    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", json.dumps(snapshot))

    original_sys_path = list(sys.path)
    try:
        bootstrap.bootstrap_child()
        with pytest.raises(ModuleNotFoundError):
            import_module(module_name)
    finally:
        sys.path[:] = original_sys_path
        sys.modules.pop(module_name, None)


def test_sealed_worker_attempts_adapter_rehydration_non_fatal(monkeypatch, tmp_path):
    """Sealed workers attempt adapter rehydration for serializer registration.

    If rehydration fails, it is not fatal — the sealed worker continues
    without an adapter. This changed from the previous behavior where
    sealed workers skipped rehydration entirely.
    """
    module_name = "sealed_opt_in_without_adapter"
    module_root = tmp_path / "adapter_guard_root"
    module_root.mkdir(parents=True, exist_ok=True)
    (module_root / f"{module_name}.py").write_text("VALUE = 99\n", encoding="utf-8")

    called = {"rehydrate": False}

    def _fail(_name: str):
        called["rehydrate"] = True
        raise ImportError("adapter module not available in sealed env")

    monkeypatch.setattr(bootstrap, "_rehydrate_adapter", _fail)
    snapshot = {
        "sys_path": [],
        "apply_host_sys_path": False,
        "adapter_ref": "fake.module:Adapter",
        "sealed_host_ro_paths": [str(module_root)],
    }
    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", json.dumps(snapshot))

    original_sys_path = list(sys.path)
    try:
        adapter = bootstrap.bootstrap_child()
        imported = import_module(module_name)
    finally:
        sys.path[:] = original_sys_path
        sys.modules.pop(module_name, None)

    assert adapter is None
    assert called["rehydrate"] is True  # rehydration was attempted
    assert imported.VALUE == 99


def test_sealed_worker_singleton_bootstrap_attempts_adapter_rehydration(monkeypatch):
    """Sealed workers attempt adapter rehydration. Failure is non-fatal."""
    called = {"rehydrate": False}

    def _fail(_name: str):
        called["rehydrate"] = True
        raise ImportError("sealed singleton cannot import adapter module")

    monkeypatch.setattr(bootstrap, "_rehydrate_adapter", _fail)
    monkeypatch.setenv(
        "PYISOLATE_HOST_SNAPSHOT",
        json.dumps(
            {
                "apply_host_sys_path": False,
                "adapter_ref": "comfy.isolation.adapter:ComfyIsolationAdapter",
                "sealed_host_ro_paths": ["/home/johnj/ComfyUI"],
            }
        ),
    )

    adapter = bootstrap.bootstrap_child()

    assert adapter is None
    assert called["rehydrate"] is True
