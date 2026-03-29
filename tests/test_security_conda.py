"""Tests for conda sealed_worker sandbox launch under bwrap (Issue 8 Slice 4)."""

from __future__ import annotations

import contextlib
import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

from pyisolate._internal.sandbox_detect import RestrictionModel


def _make_extension():
    from pyisolate._internal.host import Extension
    from pyisolate.shared import ExtensionBase

    config = {
        "name": "test_conda",
        "module": "test_module",
        "dependencies": [],
        "share_torch": False,
        "share_cuda_ipc": False,
        "package_manager": "conda",
        "execution_model": "sealed_worker",
        "conda_channels": ["conda-forge"],
        "conda_dependencies": ["numpy"],
    }

    ext = Extension.__new__(Extension)
    ext.name = "test_conda"
    ext.config = config
    ext.venv_path = Path("/fake/venv")
    ext.module_path = "/fake/module"
    ext.extension_type = ExtensionBase
    ext._cuda_ipc_enabled = False
    ext._uds_path = None
    ext._uds_listener = None
    ext._client_sock = None
    return ext


def _pixi_python_path() -> Path:
    if os.name == "nt":
        return Path("/fake/venv/.pixi/envs/default/python.exe")
    return Path("/fake/venv/.pixi/envs/default/bin/python")


def _launch_extension(ext, mock_popen: MagicMock) -> MagicMock:
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_proc.args = [
        "bwrap",
        "--clearenv",
        str(_pixi_python_path()),
        "-m",
        "pyisolate._internal.uds_client",
    ]
    mock_popen.return_value = mock_proc

    transport = MagicMock()
    transport.send = MagicMock()

    with (
        patch(
            "pyisolate._internal.host._resolve_pixi_python",
            return_value=_pixi_python_path(),
        ),
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
            patch(
                "pyisolate._internal.socket_utils.has_af_unix",
                return_value=True,
            ),
            patch(
                "pyisolate._internal.socket_utils.ensure_ipc_socket_dir",
                return_value=Path("/run"),
            ),
            patch("pyisolate._internal.host.build_extension_snapshot", return_value={}),
            patch("os.chmod"),
            contextlib.suppress(Exception),
        ):
            ext._launch_with_uds()

    return transport


def _setenv_map(cmd: list[str]) -> dict[str, str]:
    env_map: dict[str, str] = {}
    index = 0
    while index < len(cmd):
        if cmd[index] == "--setenv":
            env_map[cmd[index + 1]] = cmd[index + 2]
            index += 3
            continue
        index += 1
    return env_map


class TestCondaSealedWorkerSandboxLaunch:
    @patch("pyisolate._internal.host.subprocess.Popen")
    def test_conda_sealed_worker_launches_via_bwrap(self, mock_popen: MagicMock) -> None:
        ext = _make_extension()

        _launch_extension(ext, mock_popen)

        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "bwrap"

    @patch("pyisolate._internal.host.subprocess.Popen")
    def test_conda_sealed_worker_uses_explicit_env_allowlist(self, mock_popen: MagicMock) -> None:
        ext = _make_extension()

        with patch.dict(
            "os.environ",
            {
                "PATH": "/usr/bin",
                "LANG": "C.UTF-8",
                "PYTHONPATH": "/host/leak",
                "SECRET_TOKEN": "should_not_leak",
            },
            clear=True,
        ):
            _launch_extension(ext, mock_popen)

        cmd = mock_popen.call_args[0][0]
        env_map = _setenv_map(cmd)

        assert "--clearenv" in cmd
        assert env_map["PATH"] == "/usr/bin"
        assert env_map["LANG"] == "C.UTF-8"
        assert env_map["HOME"] == "/tmp"
        assert env_map["TMPDIR"] == "/tmp"
        assert env_map["PYTHONNOUSERSITE"] == "1"
        assert "PYTHONPATH" not in env_map
        assert "SECRET_TOKEN" not in env_map

    @patch("pyisolate._internal.host.subprocess.Popen")
    def test_conda_sealed_worker_uses_pixi_python_inside_bwrap(self, mock_popen: MagicMock) -> None:
        ext = _make_extension()

        _launch_extension(ext, mock_popen)

        cmd = mock_popen.call_args[0][0]
        assert str(_pixi_python_path()) in cmd

    @patch("pyisolate._internal.host.subprocess.Popen")
    def test_conda_sealed_worker_does_not_inject_credential_like_vars(self, mock_popen: MagicMock) -> None:
        credential_pattern = re.compile(
            r".*(_TOKEN|_SECRET|_KEY|_PASSWORD|_CREDENTIAL)$",
            re.IGNORECASE,
        )
        ext = _make_extension()

        with patch.dict(
            "os.environ",
            {
                "PATH": "/usr/bin",
                "HOME": "/home/test",
                "API_TOKEN": "topsecret",
                "GITHUB_SECRET": "still_secret",
            },
            clear=True,
        ):
            _launch_extension(ext, mock_popen)

        cmd = mock_popen.call_args[0][0]
        env_map = _setenv_map(cmd)
        injected_creds = [key for key in env_map if credential_pattern.match(key)]
        assert injected_creds == []


class TestCondaSealedWorkerBootstrapGuards:
    @patch("pyisolate._internal.host.subprocess.Popen")
    def test_conda_sealed_worker_snapshot_disables_host_sys_path(self, mock_popen: MagicMock) -> None:
        ext = _make_extension()

        transport = _launch_extension(ext, mock_popen)

        transport.send.assert_called_once()
        bootstrap_data = transport.send.call_args[0][0]
        assert bootstrap_data["snapshot"]["apply_host_sys_path"] is False
