import types
from typing import Any

import pytest

from pyisolate.host import ExtensionManager


class FakeExtension:
    @classmethod
    def __class_getitem__(cls, item: Any) -> Any:
        return cls

    def __init__(
        self, module_path: str, extension_type: Any, config: dict[str, Any], venv_root_path: str
    ) -> None:
        self.module_path = module_path
        self.extension_type = extension_type
        self.config = config
        self.venv_root_path = venv_root_path
        self.started = 0
        self.proxy_obj = types.SimpleNamespace(run=lambda: "ok")
        self.rpc = object()
        self.stopped = 0
        self._process_initialized = False

    def ensure_process_started(self) -> None:
        self.started += 1
        self._process_initialized = True

    def get_proxy(self) -> Any:
        return self.proxy_obj

    def stop(self) -> None:
        self.stopped += 1


@pytest.fixture(autouse=True)
def patch_extension(monkeypatch: Any) -> None:
    monkeypatch.setattr("pyisolate.host.Extension", FakeExtension)


def make_manager(tmp_path: Any) -> Any:
    return ExtensionManager(types.SimpleNamespace, {"venv_root_path": str(tmp_path)})


def base_config(tmp_path: Any) -> Any:
    return {
        "name": "demo",
        "module_path": "/tmp/mod.py",
        "dependencies": [],
        "share_torch": True,
        "share_cuda_ipc": False,
        "apis": [],
        "venv_root_path": str(tmp_path),
    }


def test_load_extension_returns_host_extension(monkeypatch: Any, tmp_path: Any) -> None:
    mgr = make_manager(tmp_path)
    proxy = mgr.load_extension(base_config(tmp_path))
    # First access triggers start + rpc init + proxy creation
    assert proxy.proxy.run() == "ok"
    assert getattr(proxy, "_rpc", None) is mgr.extensions["demo"].rpc
    # Subsequent access uses cached proxy, no extra starts
    _ = proxy.proxy  # Access to verify caching works
    ext = mgr.extensions["demo"]
    assert isinstance(ext, FakeExtension)
    assert ext.started == 1


def test_duplicate_extension_name_raises(tmp_path: Any) -> None:
    mgr = make_manager(tmp_path)
    cfg = base_config(tmp_path)
    mgr.load_extension(cfg)
    with pytest.raises(ValueError):
        mgr.load_extension(cfg)


def test_host_extension_getattr_delegates(monkeypatch: Any, tmp_path: Any) -> None:
    mgr = make_manager(tmp_path)
    cfg = base_config(tmp_path)
    proxy = mgr.load_extension(cfg)
    # Add attr to underlying extension
    ext = mgr.extensions["demo"]
    ext.special = "hello"
    assert proxy.special == "hello"
    # Attribute missing on extension should delegate to proxy
    assert proxy.run() == "ok"


def test_stop_all_extensions_calls_stop(tmp_path: Any) -> None:
    mgr = make_manager(tmp_path)
    cfg = base_config(tmp_path)
    mgr.load_extension(cfg)
    mgr.load_extension({**cfg, "name": "demo2"})
    mgr.stop_all_extensions()
    assert mgr.extensions == {}


def test_stop_all_extensions_logs_error(caplog: Any, tmp_path: Any) -> None:
    mgr = make_manager(tmp_path)
    cfg = base_config(tmp_path)
    _proxy = mgr.load_extension(cfg)  # noqa: F841 - load to register extension
    ext = mgr.extensions["demo"]

    def boom() -> None:
        raise RuntimeError("boom")

    ext.stop = boom  # type: ignore[assignment]

    with caplog.at_level("ERROR"):
        mgr.stop_all_extensions()

    assert "Error stopping extension 'demo'" in caplog.text
    assert mgr.extensions == {}
