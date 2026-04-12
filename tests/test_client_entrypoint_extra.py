import importlib
import sys
from types import ModuleType
from typing import Any, cast

import pytest

from pyisolate._internal import client, uds_client
from pyisolate._internal.rpc_protocol import ProxiedSingleton
from pyisolate.config import ExtensionConfig
from pyisolate.shared import ExtensionBase


class DummyExtension(ExtensionBase):
    def __init__(self) -> None:
        super().__init__()
        self.before_called = False
        self.loaded_called = False

    async def before_module_loaded(self) -> None:
        self.before_called = True

    async def on_module_loaded(self, module: ModuleType) -> None:
        self.loaded_called = True
        assert hasattr(module, "VALUE")


@pytest.mark.asyncio
async def test_async_entrypoint_runs_hooks_and_registers(tmp_path: Any, monkeypatch: Any) -> Any:
    module_dir = tmp_path / "ext"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("VALUE = 42\n")

    config: ExtensionConfig = {
        "name": "demo",
        "isolated": True,
        "dependencies": [],
        "share_torch": False,
        "share_cuda_ipc": False,
        "apis": [],
    }

    class FakeRPC:
        def __init__(self, recv_queue: Any = None, send_queue: Any = None) -> None:  # noqa: ARG002
            self.registered: list[tuple[Any, Any]] = []
            self.running = False

        def register_callee(self, obj: Any, object_id: Any) -> None:
            self.registered.append((obj, object_id))

        def run(self) -> None:
            self.running = True

        async def run_until_stopped(self) -> Any:
            return None

    monkeypatch.setattr(client, "AsyncRPC", FakeRPC)

    ext = DummyExtension()

    await client.async_entrypoint(
        module_path=str(module_dir),
        extension_type=lambda: ext,  # type: ignore[arg-type]
        config=config,
        to_extension=None,
        from_extension=None,
        log_queue=None,
    )

    assert ext.before_called is True
    assert ext.loaded_called is True


@pytest.mark.asyncio
async def test_async_entrypoint_rejects_missing_dir(tmp_path: Any) -> None:
    config: ExtensionConfig = {
        "name": "demo",
        "isolated": True,
        "dependencies": [],
        "share_torch": False,
        "share_cuda_ipc": False,
        "apis": [],
    }

    bogus = tmp_path / "notadir"
    with pytest.raises(ValueError):
        await client.async_entrypoint(
            module_path=str(bogus),
            extension_type=DummyExtension,
            config=config,
            to_extension=None,
            from_extension=None,
            log_queue=None,
        )


@pytest.mark.asyncio
async def test_async_entrypoint_uses_inference_mode(monkeypatch: Any, tmp_path: Any) -> Any:
    module_dir = tmp_path / "ext2"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("VALUE = 1\n")

    entered = {"count": 0}

    class DummyInference:
        def __enter__(self) -> Any:
            entered["count"] += 1
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:  # noqa: ANN001
            return False

    class DummyTorch:
        def inference_mode(self) -> Any:
            return DummyInference()

    monkeypatch.setitem(sys.modules, "torch", DummyTorch())

    config: ExtensionConfig = {
        "name": "demo2",
        "isolated": True,
        "dependencies": [],
        "share_torch": True,
        "share_cuda_ipc": False,
        "apis": [],
    }

    class FakeRPC:
        def __init__(self, recv_queue: Any = None, send_queue: Any = None) -> None:  # noqa: ARG002
            pass

        def register_callee(self, *_: Any) -> Any:
            return None

        def run(self) -> Any:
            return None

        async def run_until_stopped(self) -> Any:
            return None

    monkeypatch.setattr(client, "AsyncRPC", FakeRPC)

    ext = DummyExtension()
    await client.async_entrypoint(
        module_path=str(module_dir),
        extension_type=lambda: ext,  # type: ignore[arg-type]
        config=config,
        to_extension=None,
        from_extension=None,
        log_queue=None,
    )

    assert entered["count"] == 1


@pytest.mark.asyncio
async def test_async_entrypoint_registers_apis_with_adapter(monkeypatch: Any, tmp_path: Any) -> Any:
    module_dir = tmp_path / "ext3"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("VALUE = 3\n")

    class DummyAPI(ProxiedSingleton):
        last_rpc: Any = None

        @classmethod
        def use_remote(cls, rpc: Any) -> None:  # noqa: ANN001
            cls.last_rpc = rpc

    class DummyAdapter:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, Any]] = []

        def handle_api_registration(self, api_instance: Any, rpc: Any) -> None:
            self.calls.append((api_instance, rpc))

    dummy_adapter = DummyAdapter()
    monkeypatch.setattr(client, "_adapter", dummy_adapter)

    class FakeRPC:
        def __init__(self, recv_queue: Any = None, send_queue: Any = None) -> None:  # noqa: ARG002
            pass

        def register_callee(self, *_: Any) -> Any:
            return None

        def run(self) -> Any:
            return None

        async def run_until_stopped(self) -> Any:
            return None

    monkeypatch.setattr(client, "AsyncRPC", FakeRPC)

    config: ExtensionConfig = {
        "name": "demo3",
        "isolated": True,
        "dependencies": [],
        "share_torch": False,
        "share_cuda_ipc": False,
        "apis": [DummyAPI],
    }

    ext = DummyExtension()
    await client.async_entrypoint(
        module_path=str(module_dir),
        extension_type=lambda: ext,  # type: ignore[arg-type]
        config=config,
        to_extension=None,
        from_extension=None,
        log_queue=None,
    )

    assert DummyAPI.last_rpc is not None
    assert dummy_adapter.calls


def test_sealed_worker_skips_api_class_import(monkeypatch: Any) -> Any:
    config = cast(
        ExtensionConfig,
        {
            "name": "demo-sealed",
            "isolated": True,
            "dependencies": [],
            "share_torch": False,
            "share_cuda_ipc": False,
            "execution_model": "sealed_worker",
            "apis": ["forbidden.module.ForbiddenAPI"],
        },
    )

    def _forbidden_import(name: str, package: str | None = None) -> Any:  # noqa: ARG001
        if name == "forbidden.module":
            raise AssertionError("sealed worker must not import API classes from config")
        return importlib.import_module(name)

    monkeypatch.setattr(importlib, "import_module", _forbidden_import)

    resolved = uds_client._resolve_api_classes_from_config(config)

    assert resolved == []
