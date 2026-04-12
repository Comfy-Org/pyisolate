from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

from pyisolate._internal.host import Extension
from pyisolate.config import ExtensionConfig
from pyisolate.shared import ExtensionBase


class DummyProxy:
    pass


DummyProxy.__module__ = "tests.test_exact_proxy_bootstrap"


def _make_extension(config: ExtensionConfig) -> Extension:
    ext = Extension.__new__(Extension)
    ext.name = config["name"]
    ext.normalized_name = config["name"]
    ext.config = config
    ext.venv_path = Path("/fake/venv")
    ext.module_path = "/fake/module"
    ext.extension_type = ExtensionBase
    ext._cuda_ipc_enabled = False
    ext._uds_path = None
    ext._uds_listener = None
    ext._client_sock = None
    ext._host_rpc_services = []
    return ext


def _capture_bootstrap_payload(config: ExtensionConfig) -> dict[str, Any]:
    ext = _make_extension(config)

    listener = MagicMock()
    listener.accept.return_value = (MagicMock(), None)
    transport = MagicMock()
    proc = MagicMock()
    proc.pid = 1234

    with (
        patch("pyisolate._internal.host.socket") as mock_socket,
        patch(
            "pyisolate._internal.host.tempfile.mktemp", return_value="/run/user/1000/pyisolate/ext_test.sock"
        ),
        patch("pyisolate._internal.host.subprocess.Popen", return_value=proc),
        patch(
            "pyisolate._internal.host.detect_sandbox_capability",
            return_value=MagicMock(available=True, restriction_model="none"),
        ),
        patch("pyisolate._internal.host.build_bwrap_command", return_value=["bwrap", "--clearenv", "python"]),
        patch("pyisolate._internal.host.JSONSocketTransport", return_value=transport),
        patch("pyisolate._internal.host.AsyncRPC"),
        patch("pyisolate._internal.socket_utils.has_af_unix", return_value=True),
        patch(
            "pyisolate._internal.socket_utils.ensure_ipc_socket_dir",
            return_value=Path("/run/user/1000/pyisolate"),
        ),
        patch(
            "pyisolate._internal.host.build_extension_snapshot",
            return_value={"sys_path": ["/host/path"], "apply_host_sys_path": True},
        ),
        patch("os.chmod"),
        patch("sys.platform", "linux"),
    ):
        mock_socket.socket.return_value = listener
        mock_socket.AF_UNIX = 1
        mock_socket.SOCK_STREAM = 1
        ext._launch_with_uds()

    return cast(dict[str, Any], transport.send.call_args[0][0])


def test_sealed_worker_exact_proxy_binding() -> None:
    payload = _capture_bootstrap_payload(
        cast(
            ExtensionConfig,
            {
            "name": "test_ext",
            "module": "test_module",
            "module_path": "/fake/module",
            "dependencies": [],
            "share_torch": False,
            "share_cuda_ipc": False,
            "package_manager": "uv",
            "execution_model": "sealed_worker",
            "apis": [DummyProxy],
            },
        )
    )

    assert payload["snapshot"]["apply_host_sys_path"] is False
    assert payload["config"]["apis"] == ["tests.test_exact_proxy_bootstrap.DummyProxy"]
