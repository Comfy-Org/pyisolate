"""Tests for execution_model validation and backward-compatible defaults."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from pyisolate._internal.environment import validate_backend_config


def _make_config(**overrides: Any) -> Any:
    base = {
        "name": "test_ext",
        "module_path": "/fake/path",
        "isolated": True,
        "dependencies": [],
        "apis": [],
        "share_torch": False,
        "share_cuda_ipc": False,
        "sandbox": {},
        "sandbox_mode": "disabled",
        "env": {},
        "package_manager": "uv",
    }
    base.update(overrides)
    return base


def test_uv_defaults_to_host_coupled() -> None:
    config = _make_config()
    validate_backend_config(config)


def test_uv_explicit_sealed_worker_passes() -> None:
    config = _make_config(execution_model="sealed_worker")
    validate_backend_config(config)


def test_sealed_worker_rejects_share_torch_true() -> None:
    config = _make_config(execution_model="sealed_worker", share_torch=True)
    with pytest.raises(ValueError, match="sealed_worker execution_model requires share_torch=False"):
        validate_backend_config(config)


@pytest.mark.parametrize(
    ("package_manager", "execution_model"),
    [
        ("uv", "host-coupled"),
        ("uv", "sealed_worker"),
        ("conda", "sealed_worker"),
    ],
)
@patch("shutil.which", return_value="/usr/bin/pixi")
def test_rejects_cuda_ipc_without_share_torch(
    mock_which: Any, package_manager: str, execution_model: str
) -> None:
    config = _make_config(
        package_manager=package_manager,
        execution_model=execution_model,
        share_torch=False,
        share_cuda_ipc=True,
        conda_channels=["conda-forge"] if package_manager == "conda" else None,
        conda_dependencies=["numpy"] if package_manager == "conda" else None,
    )
    with pytest.raises(ValueError, match="share_cuda_ipc=True requires share_torch=True"):
        validate_backend_config(config)


@patch("pyisolate._internal.pixi_provisioner.ensure_pixi", return_value="/usr/bin/pixi")
def test_accepts_valid_mode_matrix(mock_ensure_pixi: Any) -> None:
    valid_configs = [
        _make_config(execution_model="host-coupled", share_torch=True, share_cuda_ipc=True),
        _make_config(execution_model="host-coupled", share_torch=True, share_cuda_ipc=False),
        _make_config(execution_model="host-coupled", share_torch=False, share_cuda_ipc=False),
        _make_config(
            package_manager="conda",
            execution_model="sealed_worker",
            share_torch=False,
            share_cuda_ipc=False,
            conda_channels=["conda-forge"],
            conda_dependencies=["numpy"],
        ),
    ]

    for config in valid_configs:
        validate_backend_config(config)


def test_sealed_host_ro_paths_defaults_off_and_validation() -> None:
    config = _make_config(execution_model="sealed_worker")
    validate_backend_config(config)
    assert config.get("sealed_host_ro_paths") is None

    valid = _make_config(
        execution_model="sealed_worker",
        sealed_host_ro_paths=["/home/johnj/ComfyUI"],
    )
    validate_backend_config(valid)

    wrong_mode = _make_config(
        execution_model="host-coupled",
        sealed_host_ro_paths=["/home/johnj/ComfyUI"],
    )
    with pytest.raises(ValueError, match="sealed_host_ro_paths requires execution_model='sealed_worker'"):
        validate_backend_config(wrong_mode)

    non_list = _make_config(execution_model="sealed_worker", sealed_host_ro_paths="/home/johnj/ComfyUI")
    with pytest.raises(ValueError, match="sealed_host_ro_paths must be a list of absolute paths"):
        validate_backend_config(non_list)

    relative = _make_config(execution_model="sealed_worker", sealed_host_ro_paths=["relative/path"])
    with pytest.raises(ValueError, match="sealed_host_ro_paths entries must be absolute paths"):
        validate_backend_config(relative)


@patch("pyisolate._internal.pixi_provisioner.ensure_pixi", return_value="/usr/bin/pixi")
def test_conda_defaults_to_sealed_worker(mock_ensure_pixi: Any) -> None:
    config = _make_config(
        package_manager="conda",
        conda_channels=["conda-forge"],
        conda_dependencies=["numpy"],
    )
    validate_backend_config(config)


@patch("shutil.which", return_value="/usr/bin/pixi")
def test_conda_rejects_host_coupled_execution_model(mock_which: Any) -> None:
    config = _make_config(
        package_manager="conda",
        execution_model="host-coupled",
        conda_channels=["conda-forge"],
        conda_dependencies=["numpy"],
    )
    with pytest.raises(ValueError, match="conda backend requires execution_model='sealed_worker'"):
        validate_backend_config(config)
