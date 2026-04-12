"""Tests for conda backend configuration and validation."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from pyisolate._internal.environment import validate_backend_config


def _make_config(**overrides: Any) -> Any:
    """Build a minimal ExtensionConfig dict with conda defaults."""
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


class TestDefaultPackageManager:
    def test_default_package_manager_is_uv(self) -> None:
        """Config without package_manager should default to 'uv' and pass validation."""
        config = _make_config()
        del config["package_manager"]
        # Should not raise — uv is the default
        validate_backend_config(config)


class TestCondaShareTorchRaises:
    def test_conda_share_torch_raises(self) -> None:
        """conda + share_torch=True must raise ValueError."""
        config = _make_config(
            package_manager="conda",
            share_torch=True,
            conda_channels=["conda-forge"],
        )
        with pytest.raises(ValueError, match="share_torch=False"):
            validate_backend_config(config)


class TestCondaCudaWheelsAllowed:
    @patch("pyisolate._internal.pixi_provisioner.ensure_pixi", return_value="/usr/bin/pixi")
    def test_conda_cuda_wheels_allowed(self, _mock_ensure_pixi: Any) -> None:
        """conda + cuda_wheels is valid — pixi resolves via [pypi-options]."""
        config = _make_config(
            package_manager="conda",
            conda_channels=["conda-forge"],
            cuda_wheels=["cu121"],
        )
        # Should not raise — conda supports cuda_wheels via extra-index-urls
        validate_backend_config(config)


class TestCondaMissingChannelsRaises:
    def test_conda_missing_channels_raises(self) -> None:
        """conda + empty/missing conda_channels must raise ValueError."""
        config = _make_config(
            package_manager="conda",
        )
        with pytest.raises(ValueError, match="conda_channels"):
            validate_backend_config(config)

    def test_conda_empty_channels_raises(self) -> None:
        """conda + empty conda_channels list must raise ValueError."""
        config = _make_config(
            package_manager="conda",
            conda_channels=[],
        )
        with pytest.raises(ValueError, match="conda_channels"):
            validate_backend_config(config)


class TestCondaMissingPixiRaises:
    @patch(
        "pyisolate._internal.pixi_provisioner.ensure_pixi",
        side_effect=RuntimeError("pixi bootstrap failed"),
    )
    def test_conda_missing_pixi_raises(self, mock_ensure_pixi: Any) -> None:
        """conda + failed pixi bootstrap must raise ValueError."""
        config = _make_config(
            package_manager="conda",
            conda_channels=["conda-forge"],
        )
        with pytest.raises(
            ValueError, match="pixi is required for conda backend but could not be provisioned"
        ):
            validate_backend_config(config)


class TestCondaValidConfigPasses:
    @patch("pyisolate._internal.pixi_provisioner.ensure_pixi", return_value="/usr/bin/pixi")
    def test_conda_valid_config_passes(self, mock_ensure_pixi: Any) -> None:
        """Valid conda config must pass validation without error."""
        config = _make_config(
            package_manager="conda",
            conda_channels=["conda-forge"],
            conda_dependencies=["numpy"],
        )
        # Should not raise
        validate_backend_config(config)

    @patch("pyisolate._internal.pixi_provisioner.ensure_pixi", return_value="/usr/bin/pixi")
    def test_conda_with_platforms_passes(self, mock_ensure_pixi: Any) -> None:
        """Valid conda config with platforms must pass."""
        config = _make_config(
            package_manager="conda",
            conda_channels=["conda-forge"],
            conda_dependencies=["numpy"],
            conda_platforms=["linux-64"],
        )
        validate_backend_config(config)
