"""Tests for uv + sealed_worker host dispatch under bwrap (Issue 8 Slice 2)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from pyisolate._internal.sandbox_detect import RestrictionModel
from pyisolate.config import ExtensionConfig


def _make_extension(config: ExtensionConfig) -> Any:
    from pyisolate._internal.host import Extension
    from pyisolate.shared import ExtensionBase

    ext = Extension.__new__(Extension)
    ext.name = "test_ext"
    ext.config = config
    ext.venv_path = Path("/fake/venv")
    ext.module_path = "/fake/module"
    ext.extension_type = ExtensionBase
    ext._cuda_ipc_enabled = False
    return ext


def _uv_python_path(venv_path: Path) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


class TestLaunchDispatchSealedWorker:
    @patch("pyisolate._internal.host.validate_backend_config")
    @patch("pyisolate._internal.host.create_conda_env")
    @patch("pyisolate._internal.host.create_venv")
    @patch("pyisolate._internal.host.install_dependencies")
    def test_uv_sealed_worker_uses_uv_env_path(
        self,
        mock_install_deps: MagicMock,
        mock_create_venv: MagicMock,
        mock_create_conda: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        config: ExtensionConfig = {
            "name": "test_ext",
            "module": "test_module",
            "isolated": True,
            "dependencies": [],
            "apis": [],
            "share_torch": False,
            "share_cuda_ipc": False,
            "package_manager": "uv",
            "execution_model": "sealed_worker",
        }

        ext = _make_extension(config)

        with patch.object(ext, "_launch_with_uds", return_value=MagicMock()):
            ext._Extension__launch()

        mock_validate.assert_called_once_with(config)
        mock_create_venv.assert_called_once()
        mock_install_deps.assert_called_once()
        mock_create_conda.assert_not_called()

    def test_uv_sealed_worker_uses_json_tensor_transport(self) -> None:
        config: ExtensionConfig = {
            "name": "test_ext",
            "module": "test_module",
            "isolated": True,
            "dependencies": [],
            "apis": [],
            "share_torch": False,
            "share_cuda_ipc": False,
            "package_manager": "uv",
            "execution_model": "sealed_worker",
        }

        ext = _make_extension(config)

        assert ext._tensor_transport_mode() == "json"

    @patch("pyisolate._internal.host.build_bwrap_command")
    @patch("pyisolate._internal.host.subprocess.Popen")
    def test_uv_sealed_worker_launches_through_bwrap_with_strict_policy(
        self,
        mock_popen: MagicMock,
        mock_build_bwrap: MagicMock,
    ) -> None:
        config: ExtensionConfig = {
            "name": "test_ext",
            "module": "test_module",
            "isolated": True,
            "dependencies": [],
            "apis": [],
            "share_torch": False,
            "share_cuda_ipc": False,
            "package_manager": "uv",
            "execution_model": "sealed_worker",
        }

        ext = _make_extension(config)
        ext._uds_path = None
        ext._uds_listener = None
        ext._client_sock = None

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc
        mock_build_bwrap.return_value = [
            "bwrap",
            "--clearenv",
            "--setenv",
            "PYISOLATE_UDS_ADDRESS",
            "/run/ext.sock",
        ]

        transport = MagicMock()
        transport.send = MagicMock()

        with (
            patch("pyisolate._internal.host.socket") as mock_socket,
            patch("pyisolate._internal.host.tempfile"),
            patch("pyisolate._internal.host.detect_sandbox_capability") as mock_detect,
            patch("sys.platform", "linux"),
            patch("pyisolate._internal.host.JSONSocketTransport", return_value=transport),
            patch("pyisolate._internal.host.AsyncRPC"),
        ):
            mock_detect.return_value = MagicMock(
                available=True,
                restriction_model=RestrictionModel.NONE,
            )
            mock_listener = MagicMock()
            mock_listener.accept.return_value = (MagicMock(), None)
            mock_socket.socket.return_value = mock_listener
            mock_socket.AF_UNIX = 1
            mock_socket.SOCK_STREAM = 1

            with (
                patch("pyisolate._internal.socket_utils.has_af_unix", return_value=True),
                patch("pyisolate._internal.socket_utils.ensure_ipc_socket_dir", return_value=Path("/run")),
                patch("pyisolate._internal.host.build_extension_snapshot", return_value={}),
                patch("os.chmod"),
            ):
                ext._launch_with_uds()

        mock_build_bwrap.assert_called_once()
        kwargs = mock_build_bwrap.call_args.kwargs
        assert kwargs["execution_model"] == "sealed_worker"
        assert kwargs["sandbox_config"] == {}
        assert kwargs["python_exe"] == str(_uv_python_path(ext.venv_path))
        assert kwargs["module_path"] == ext.module_path
        transport.send.assert_called_once()
        bootstrap_data = transport.send.call_args[0][0]
        assert bootstrap_data["snapshot"]["apply_host_sys_path"] is False

    def test_uv_host_coupled_keeps_shared_memory_tensor_transport(self) -> None:
        config: ExtensionConfig = {
            "name": "test_ext",
            "module": "test_module",
            "isolated": True,
            "dependencies": [],
            "apis": [],
            "share_torch": False,
            "share_cuda_ipc": False,
            "package_manager": "uv",
        }

        ext = _make_extension(config)

        assert ext._tensor_transport_mode() == "shared_memory"
