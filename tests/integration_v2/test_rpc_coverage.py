"""Integration tests for RPC protocol coverage.

These tests exercise the full AsyncRPC lifecycle (run, _send_thread,
_recv_thread, _dispatch_callee_call) by loading real extensions via
ReferenceHost and making cross-process RPC calls.

Subordinate to issue #102 Slice 5 via issue #104.
"""

import asyncio
import sys

import pytest

from tests.harness.host import ReferenceHost


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def host():
    h = ReferenceHost()
    h.setup()
    yield h
    _run(h.cleanup())


@pytest.mark.skipif(sys.platform == "win32", reason="Extension loading requires Linux UDS")
class TestTorchShareRPC:
    def test_torch_share_singleton_roundtrip(self, host: ReferenceHost):
        """Load a torch_share extension and verify RPC round-trip.

        Exercises AsyncRPC.run, _send_thread, _recv_thread, and
        _dispatch_callee_call — the paths at 0% unit test coverage.
        """
        ext = host.load_test_extension(
            name="rpc_cov_torch",
            share_torch=True,
            share_cuda=False,
        )
        proxy = ext.get_proxy()
        result = _run(proxy.ping())
        assert result == "pong", f"Expected 'pong', got {result!r}"


@pytest.mark.skipif(sys.platform == "win32", reason="Extension loading requires Linux UDS")
class TestSealedWorkerRPC:
    def test_sealed_worker_singleton_roundtrip(self, host: ReferenceHost):
        """Load a sealed_worker extension and verify RPC round-trip.

        Exercises the same RPC paths as torch_share but with
        execution_model=sealed_worker and share_torch=False.
        """
        from pathlib import Path

        from pyisolate.config import ExtensionConfig, SandboxMode
        from pyisolate.host import Extension
        from tests.harness.test_package import ReferenceTestExtension

        pyisolate_root = Path(__file__).resolve().parents[2]
        package_path = Path(ReferenceTestExtension.__module__.replace(".", "/")).resolve()
        # Use the test_package path from the harness
        import tests.harness.test_package as tp

        package_path = str(Path(tp.__file__).parent.resolve())

        config = ExtensionConfig(
            name="rpc_cov_sealed",
            module_path=package_path,
            isolated=True,
            dependencies=[str(pyisolate_root)],
            apis=[],
            share_torch=False,
            share_cuda_ipc=False,
            execution_model="sealed_worker",
            sandbox_mode=SandboxMode.DISABLED,
            env={"PYISOLATE_SIGNAL_CLEANUP": "1"},
        )

        ext = Extension(
            module_path=package_path,
            extension_type=ReferenceTestExtension,
            config=config,
            venv_root_path=str(host.venv_root),
        )
        ext.ensure_process_started()
        host.extensions.append(ext)

        proxy = ext.get_proxy()
        result = _run(proxy.ping())
        assert result == "pong", f"Expected 'pong', got {result!r}"


@pytest.mark.skipif(sys.platform != "linux", reason="bwrap requires Linux")
class TestBwrapTorchShareRPC:
    def test_bwrap_torch_share_roundtrip(self, host: ReferenceHost):
        """Load a bwrap + torch_share extension and verify RPC round-trip."""
        from pyisolate._internal.sandbox_detect import detect_sandbox_capability

        cap = detect_sandbox_capability()
        if not cap.available:
            pytest.skip(f"bwrap not available: {cap.restriction_model}")

        ext = host.load_test_extension(
            name="rpc_cov_bwrap_torch",
            share_torch=True,
            share_cuda=False,
        )
        proxy = ext.get_proxy()
        result = _run(proxy.ping())
        assert result == "pong"
