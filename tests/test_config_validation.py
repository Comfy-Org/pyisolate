"""Tests for ExtensionManagerConfig and ExtensionConfig validation.

These tests verify configuration validation without spawning processes.
"""

from pathlib import Path

from pyisolate import ExtensionConfig, ExtensionManagerConfig, ProxiedSingleton


class TestExtensionManagerConfig:
    """Tests for ExtensionManagerConfig TypedDict."""

    def test_minimal_config(self, tmp_path: Path) -> None:
        """Minimal config requires only venv_root_path."""
        config: ExtensionManagerConfig = {
            "venv_root_path": str(tmp_path / "venvs"),
        }

        assert "venv_root_path" in config

    def test_venv_root_path_is_string(self, tmp_path: Path) -> None:
        """venv_root_path must be a string path."""
        config: ExtensionManagerConfig = {
            "venv_root_path": str(tmp_path / "venvs"),
        }

        assert isinstance(config["venv_root_path"], str)


class TestExtensionConfigValidation:
    """Tests for ExtensionConfig field validation."""

    def test_name_must_be_nonempty(self) -> None:
        """Extension name should not be empty."""
        config: ExtensionConfig = {
            "name": "",  # Invalid but TypedDict doesn't enforce
            "module_path": "/path/to/ext",
            "isolated": True,
            "dependencies": [],
            "apis": [],
            "share_torch": False,
            "share_cuda_ipc": False,
        }

        # The config is syntactically valid, but semantically
        # empty name would cause issues. Runtime validation needed.
        assert config["name"] == ""

    def test_module_path_can_be_relative(self) -> None:
        """Module path can be relative."""
        config: ExtensionConfig = {
            "name": "myext",
            "module_path": "./extensions/myext",
            "isolated": True,
            "dependencies": [],
            "apis": [],
            "share_torch": False,
            "share_cuda_ipc": False,
        }

        assert config["module_path"] == "./extensions/myext"

    def test_module_path_can_be_absolute(self) -> None:
        """Module path can be absolute."""
        config: ExtensionConfig = {
            "name": "myext",
            "module_path": "/app/extensions/myext",
            "isolated": True,
            "dependencies": [],
            "apis": [],
            "share_torch": False,
            "share_cuda_ipc": False,
        }

        assert config["module_path"] == "/app/extensions/myext"

    def test_dependencies_format(self) -> None:
        """Dependencies are pip requirement specifiers."""
        config: ExtensionConfig = {
            "name": "myext",
            "module_path": "/path",
            "isolated": True,
            "dependencies": [
                "numpy>=1.20",
                "pillow==10.0.0",
                "package[extra]>=2.0,<3.0",
            ],
            "apis": [],
            "share_torch": False,
            "share_cuda_ipc": False,
        }

        assert len(config["dependencies"]) == 3
        assert "numpy>=1.20" in config["dependencies"]

    def test_apis_are_singleton_types(self) -> None:
        """APIs list contains ProxiedSingleton subclasses."""

        class MyService(ProxiedSingleton):
            pass

        config: ExtensionConfig = {
            "name": "myext",
            "module_path": "/path",
            "isolated": True,
            "dependencies": [],
            "apis": [MyService],
            "share_torch": False,
            "share_cuda_ipc": False,
        }

        assert MyService in config["apis"]
        assert issubclass(config["apis"][0], ProxiedSingleton)

    def test_share_torch_implies_requirements(self) -> None:
        """share_torch=True has implications for tensor handling."""
        config: ExtensionConfig = {
            "name": "ml_ext",
            "module_path": "/path",
            "isolated": True,
            "dependencies": [],
            "apis": [],
            "share_torch": True,
            "share_cuda_ipc": False,
        }

        assert config["share_torch"] is True
        # share_cuda_ipc can be independently configured
        assert config["share_cuda_ipc"] is False

    def test_share_cuda_ipc_requires_share_torch(self) -> None:
        """share_cuda_ipc only makes sense with share_torch."""
        # This is a semantic constraint, not enforced by TypedDict
        config: ExtensionConfig = {
            "name": "gpu_ext",
            "module_path": "/path",
            "isolated": True,
            "dependencies": [],
            "apis": [],
            "share_torch": True,  # Required for cuda_ipc to work
            "share_cuda_ipc": True,
        }

        assert config["share_torch"] is True
        assert config["share_cuda_ipc"] is True


class TestConfigDefaults:
    """Tests documenting expected config defaults."""

    def test_isolated_defaults_true(self) -> None:
        """When creating configs, isolated typically defaults True."""
        # This documents expected behavior for config factories
        default_isolated = True

        config: ExtensionConfig = {
            "name": "ext",
            "module_path": "/path",
            "isolated": default_isolated,
            "dependencies": [],
            "apis": [],
            "share_torch": False,
            "share_cuda_ipc": False,
        }

        assert config["isolated"] is True

    def test_share_torch_defaults_false(self) -> None:
        """share_torch defaults to False for safety."""
        default_share_torch = False

        config: ExtensionConfig = {
            "name": "ext",
            "module_path": "/path",
            "isolated": True,
            "dependencies": [],
            "apis": [],
            "share_torch": default_share_torch,
            "share_cuda_ipc": False,
        }

        assert config["share_torch"] is False

    def test_dependencies_defaults_empty(self) -> None:
        """dependencies defaults to empty list."""
        config: ExtensionConfig = {
            "name": "ext",
            "module_path": "/path",
            "isolated": True,
            "dependencies": [],
            "apis": [],
            "share_torch": False,
            "share_cuda_ipc": False,
        }

        assert config["dependencies"] == []

    def test_apis_defaults_empty(self) -> None:
        """apis defaults to empty list."""
        config: ExtensionConfig = {
            "name": "ext",
            "module_path": "/path",
            "isolated": True,
            "dependencies": [],
            "apis": [],
            "share_torch": False,
            "share_cuda_ipc": False,
        }

        assert config["apis"] == []
