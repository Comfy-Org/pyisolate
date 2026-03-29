"""Generic conda/uv sealed-worker contract tests."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyisolate._internal import bootstrap
from pyisolate._internal.environment_conda import _generate_pixi_toml, _resolve_pixi_python
from pyisolate._internal.host import Extension
from pyisolate._internal.sandbox_detect import RestrictionModel
from pyisolate.shared import ExtensionBase


def _make_conda_config(**overrides):
    config = {
        "name": "contract_ext",
        "module": "contract_module",
        "dependencies": ["requests>=2.0"],
        "share_torch": False,
        "share_cuda_ipc": False,
        "package_manager": "conda",
        "conda_channels": ["conda-forge"],
        "conda_dependencies": ["numpy>=1.26"],
        "conda_platforms": ["linux-64"],
    }
    config.update(overrides)
    return config


def _make_extension(config: dict) -> Extension:
    ext = Extension.__new__(Extension)
    ext.name = config["name"]
    ext.config = config
    ext.venv_path = Path("/fake/venv")
    ext.module_path = "/fake/module"
    ext.extension_type = ExtensionBase
    ext._cuda_ipc_enabled = False
    ext._uds_path = None
    ext._uds_listener = None
    ext._client_sock = None
    return ext


def _capture_bootstrap_payload(config: dict) -> dict:
    ext = _make_extension(config)

    listener = MagicMock()
    listener.accept.return_value = (MagicMock(), None)
    transport = MagicMock()
    proc = MagicMock()
    proc.pid = 1234

    with (
        patch(
            "pyisolate._internal.host._resolve_pixi_python",
            return_value=Path("/fake/venv/.pixi/envs/default/bin/python"),
        ),
        patch("pyisolate._internal.host.socket") as mock_socket,
        patch(
            "pyisolate._internal.host.tempfile.mktemp", return_value="/run/user/1000/pyisolate/ext_test.sock"
        ),
        patch("pyisolate._internal.host.subprocess.Popen", return_value=proc),
        patch(
            "pyisolate._internal.host.detect_sandbox_capability",
            return_value=MagicMock(available=True, restriction_model=RestrictionModel.NONE),
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

    return transport.send.call_args[0][0]


def test_conda_dependency_split():
    config = _make_conda_config(
        conda_dependencies=["numpy>=1.26", "scipy"],
        dependencies=["requests>=2.0", "pandas"],
    )

    toml_text = _generate_pixi_toml(config)

    assert "[dependencies]" in toml_text
    assert 'numpy = ">=1.26"' in toml_text
    assert 'scipy = "*"' in toml_text
    assert "[pypi-dependencies]" in toml_text
    assert 'requests = ">=2.0"' in toml_text
    assert 'pandas = "*"' in toml_text


def test_conda_channels_platforms_pass_through():
    config = _make_conda_config(
        conda_channels=["conda-forge", "nvidia"],
        conda_platforms=["linux-64", "win-64"],
    )

    toml_text = _generate_pixi_toml(config)

    assert 'channels = ["conda-forge", "nvidia"]' in toml_text
    assert 'platforms = ["linux-64", "win-64"]' in toml_text


def test_uv_defaults_unchanged():
    config = _make_conda_config(package_manager="uv")
    ext = _make_extension(config)

    assert ext._execution_model() == "host-coupled"
    assert ext._tensor_transport_mode() == "shared_memory"


def test_no_host_fallback(tmp_path: Path):
    env_path = tmp_path / "conda_env"
    if os.name == "nt":
        python_path = env_path / ".pixi" / "envs" / "default" / "python.exe"
    else:
        python_path = env_path / ".pixi" / "envs" / "default" / "bin" / "python"
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.touch()

    resolved = _resolve_pixi_python(env_path)

    assert str(resolved) != sys.executable
    assert ".pixi" in str(resolved)


def test_no_host_sys_path():
    payload = _capture_bootstrap_payload(
        _make_conda_config(package_manager="conda", execution_model="sealed_worker")
    )

    snapshot = payload["snapshot"]
    assert snapshot["apply_host_sys_path"] is False
    assert snapshot["additional_paths"] == []
    assert snapshot["preferred_root"] is None


def test_no_extension_wrapper_import():
    payload = _capture_bootstrap_payload(
        _make_conda_config(package_manager="conda", execution_model="sealed_worker")
    )

    snapshot = payload["snapshot"]
    assert snapshot["adapter_ref"] is None
    assert snapshot["adapter_name"] is None
    assert "extension_wrapper" not in str(payload)


def test_sealed_worker_host_policy_ro_paths_default_block_and_opt_in_allow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload_default = _capture_bootstrap_payload(
        _make_conda_config(package_manager="conda", execution_model="sealed_worker")
    )
    payload_opt_in = _capture_bootstrap_payload(
        _make_conda_config(
            package_manager="conda",
            execution_model="sealed_worker",
            sealed_host_ro_paths=["/opt/example/app"],
        )
    )

    assert payload_default["snapshot"].get("sealed_host_ro_paths", []) == []
    assert payload_opt_in["snapshot"]["sealed_host_ro_paths"] == ["/opt/example/app"]

    app_framework_root = tmp_path / "app_framework_root"
    app_api_dir = app_framework_root / "app_framework_api"
    app_api_dir.mkdir(parents=True, exist_ok=True)
    (app_api_dir / "__init__.py").write_text("", encoding="utf-8")
    (app_api_dir / "latest.py").write_text("MARKER = 'ok'\n", encoding="utf-8")

    module_name = "app_framework_api.latest"
    original_sys_path = list(sys.path)
    try:
        monkeypatch.setenv(
            "PYISOLATE_HOST_SNAPSHOT",
            json.dumps(
                {
                    "sys_path": [str(app_framework_root)],
                    "apply_host_sys_path": False,
                }
            ),
        )
        bootstrap.bootstrap_child()
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)

        monkeypatch.setenv(
            "PYISOLATE_HOST_SNAPSHOT",
            json.dumps(
                {
                    "sys_path": [str(app_framework_root)],
                    "apply_host_sys_path": False,
                    "sealed_host_ro_paths": [str(app_framework_root)],
                }
            ),
        )
        bootstrap.bootstrap_child()
        imported = importlib.import_module(module_name)
    finally:
        sys.path[:] = original_sys_path
        sys.modules.pop("app_framework_api.latest", None)
        sys.modules.pop("app_framework_api", None)

    assert imported.MARKER == "ok"
