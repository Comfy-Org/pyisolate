import asyncio
import logging
import queue
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyisolate._internal import host
from pyisolate._internal.host import Extension
from pyisolate._internal.rpc_protocol import AsyncRPC, ProxiedSingleton
from pyisolate._internal.rpc_transports import JSONSocketTransport
from pyisolate._internal.sandbox_detect import RestrictionModel, SandboxCapability


class DummyRPC:
    def __init__(self, *args, **kwargs):
        self.run_called = False

    def register_callee(self, obj, object_id):
        pass

    def run(self):
        self.run_called = True


class DummyProcess:
    def __init__(self):
        self.alive = False

    def start(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.alive = False

    def join(self, timeout=None):
        self.alive = False

    def kill(self):
        self.alive = False


class DummyContext:
    def __init__(self):
        self.q = queue.Queue()

    def Queue(self):  # noqa: N802 - matches multiprocessing API
        return queue.Queue()

    def Process(self, target, args):  # noqa: N802 - matches multiprocessing API
        return DummyProcess()


class DummyMP:
    def __init__(self):
        self.ctx = DummyContext()
        self.executable = None

    def get_context(self, mode):
        return self.ctx

    def set_executable(self, exe):
        self.executable = exe


class DummyExtension(Extension):
    def __init__(self, tmp_path: Path, config_overrides=None):
        base_config = {
            "name": "demo",
            "dependencies": [],
            "share_torch": True,
            "share_cuda_ipc": False,
            "apis": [],
        }
        if config_overrides:
            base_config.update(config_overrides)
        super().__init__(
            module_path="/tmp/mod.py",
            extension_type=SimpleNamespace,
            config=base_config,
            venv_root_path=str(tmp_path),
        )
        # patch multiprocessing
        self.mp = DummyMP()

    def _create_extension_venv(self):
        # skip actual venv creation
        return

    def _install_dependencies(self):
        return

    def __launch(self):
        return DummyProcess()


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    monkeypatch.delenv("PYISOLATE_ENABLE_CUDA_IPC", raising=False)
    monkeypatch.delenv("PYISOLATE_CHILD", raising=False)


def test_initialize_process_requires_share_torch_for_cuda_ipc(tmp_path):
    ext = DummyExtension(tmp_path, {"share_torch": False, "share_cuda_ipc": True})
    with pytest.raises(RuntimeError):
        ext._initialize_process()


def test_initialize_process_cuda_ipc_unavailable_raises(monkeypatch, tmp_path):
    ext = DummyExtension(tmp_path, {"share_torch": True, "share_cuda_ipc": True})
    from pyisolate._internal import torch_utils

    monkeypatch.setattr(torch_utils, "probe_cuda_ipc_support", lambda: (False, "no"))

    def mock_launch():
        if ext.config.get("share_cuda_ipc"):
            supported, reason = torch_utils.probe_cuda_ipc_support()
            if not supported:
                raise RuntimeError(f"CUDA IPC not available: {reason}")
        return SimpleNamespace(poll=lambda: None, terminate=lambda: None)

    monkeypatch.setattr(ext, "_Extension__launch", mock_launch)

    with pytest.raises(RuntimeError):
        ext._initialize_process()


@pytest.mark.skipif(sys.platform == "win32", reason="AF_UNIX monkeypatch requires Linux")
def test_initialize_process_sets_env_and_runs_rpc(monkeypatch, tmp_path):
    ext = DummyExtension(tmp_path, {"share_torch": True, "share_cuda_ipc": False})
    monkeypatch.setattr(host, "AsyncRPC", lambda recv_queue=None, send_queue=None, transport=None: DummyRPC())

    class MockPopen:
        def __init__(self, cmd, **kwargs):
            self.args = cmd
            self.env = kwargs.get("env", {})
            self.returncode = None

        def poll(self):
            return None

        def terminate(self):
            pass

        def kill(self):
            pass

        def wait(self, timeout=None):
            return 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def communicate(self, input=None, timeout=None):
            return (b"", b"")

    monkeypatch.setattr(host.subprocess, "Popen", MockPopen)

    # Mock sandbox detection to pass on Linux
    monkeypatch.setattr(
        host,
        "detect_sandbox_capability",
        lambda: SandboxCapability(
            available=True,
            bwrap_path="/usr/bin/bwrap",
            restriction_model=RestrictionModel.NONE,
            remediation="",
            raw_error=None,
        ),
    )

    class MockSocket:
        def __init__(self, *args, **kwargs):
            pass

        def bind(self, path):
            pass

        def listen(self, backlog):
            pass

        def accept(self):
            return (MockSocket(), "addr")

        def close(self):
            pass

        def sendall(self, data):
            pass

        def recv(self, n):
            return b""

        def shutdown(self, how):
            pass

    monkeypatch.setattr(host.socket, "socket", MockSocket)
    monkeypatch.setattr(host.socket, "AF_UNIX", 1)
    monkeypatch.setattr(host.socket, "SOCK_STREAM", 1)

    monkeypatch.setattr(host.os, "chmod", lambda path, mode, **kwargs: None)

    class MockTransport:
        def __init__(self, sock):
            pass

        def send(self, data):
            pass

        def recv(self):
            return {}

        def close(self):
            pass

    monkeypatch.setattr(host, "JSONSocketTransport", MockTransport)

    venv_path = Path(tmp_path) / "demo"
    site_packages = venv_path / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True, exist_ok=True)

    python_exe = venv_path / "bin" / "python"
    python_exe.parent.mkdir(parents=True, exist_ok=True)
    python_exe.write_text("#!/usr/bin/env python")
    python_exe.chmod(0o755)

    monkeypatch.setattr(host, "create_venv", lambda *args, **kwargs: None)
    monkeypatch.setattr(host, "install_dependencies", lambda *args, **kwargs: None)

    ext._initialize_process()
    val = ext.proc.env.get("PYISOLATE_ENABLE_CUDA_IPC")
    assert val == "0" or val is None
    assert isinstance(ext.rpc, DummyRPC)
    assert ext.rpc.run_called is True


def test_install_dependencies_no_deps_returns(monkeypatch, tmp_path):
    ext = DummyExtension(tmp_path)
    # ensure python exe exists
    venv_bin = Path(ext.venv_path / "bin")
    venv_bin.mkdir(parents=True, exist_ok=True)
    exe = venv_bin / "python"
    exe.write_text("#!/usr/bin/env python")
    ext._install_dependencies()


def test_probe_cuda_ipc_support_handles_import_error(monkeypatch):
    from pyisolate._internal import torch_utils

    monkeypatch.setattr(torch_utils.sys, "platform", "linux")
    monkeypatch.setitem(torch_utils.sys.modules, "torch", None)
    supported, reason = torch_utils.probe_cuda_ipc_support()
    assert supported is False
    assert "torch import failed" in reason


def test_install_dependencies_respects_lock_cache(monkeypatch, tmp_path):
    ext = DummyExtension(tmp_path)
    venv_bin = Path(ext.venv_path / "bin")
    venv_bin.mkdir(parents=True, exist_ok=True)
    exe = venv_bin / "python"
    exe.write_text("#!/usr/bin/env python")

    lock = ext.venv_path / ".pyisolate_deps.json"
    from pyisolate._internal import environment

    descriptor = {
        "dependencies": [],
        "share_torch": True,
        "torch_spec": None,
        "pyisolate": environment.pyisolate_version,
        "python": host.sys.version,
    }
    import hashlib
    import json

    fp = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
    lock.write_text(json.dumps({"fingerprint": fp, "descriptor": descriptor}))

    # should return early without invoking pip/uv
    ext._install_dependencies()


def test_callable_roundtrip_shutdown_is_clean(caplog, capsys):
    class HostCallbackAPI(ProxiedSingleton):
        async def invoke(self, handler, payload):
            return await handler(payload)

    async def scenario():
        left, right = socket.socketpair()
        host_transport = JSONSocketTransport(left)
        child_transport = JSONSocketTransport(right)
        host_rpc = AsyncRPC(transport=host_transport)
        child_rpc = AsyncRPC(transport=child_transport)
        HostCallbackAPI()._register(host_rpc)
        host_rpc.run()
        child_rpc.run()
        caller = child_rpc.create_caller(HostCallbackAPI, HostCallbackAPI.get_remote_id())

        try:
            def handler(payload):
                return {"value": payload["value"] + 1}

            result = await asyncio.wait_for(caller.invoke(handler, {"value": 41}), timeout=5)
            return result
        finally:
            child_rpc.shutdown()
            host_rpc.shutdown()
            await asyncio.sleep(0)
            child_transport.close()
            host_transport.close()
            await asyncio.sleep(0)

    with caplog.at_level(logging.ERROR):
        result = asyncio.run(scenario())

    captured = capsys.readouterr()
    assert result == {"value": 42}
    assert "InvalidStateError" not in captured.err
    assert "RPC recv failed" not in caplog.text
