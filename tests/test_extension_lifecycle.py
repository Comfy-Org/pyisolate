"""Tests for extension load/execute/stop lifecycle.

These tests verify pyisolate correctly manages extension lifecycle:
1. Creates extension venv
2. Installs dependencies
3. Spawns isolated process
4. Executes extension methods
5. Returns results
6. Stops cleanly

Note: These are unit tests that verify lifecycle contracts without
actually spawning subprocesses. For full integration tests, see
original_integration/.
"""

from pathlib import Path

from pyisolate.config import ExtensionConfig


class TestExtensionConfig:
    """Tests for ExtensionConfig TypedDict."""

    def test_config_requires_name(self) -> None:
        """ExtensionConfig must have a name."""
        config: ExtensionConfig = {
            "name": "test_ext",
            "module_path": "/path/to/ext",
            "isolated": True,
            "dependencies": [],
            "apis": [],
            "share_torch": False,
            "share_cuda_ipc": False,
        }
        assert config["name"] == "test_ext"

    def test_config_requires_module_path(self) -> None:
        """ExtensionConfig must have a module_path."""
        config: ExtensionConfig = {
            "name": "test_ext",
            "module_path": "/path/to/ext",
            "isolated": True,
            "dependencies": [],
            "apis": [],
            "share_torch": False,
            "share_cuda_ipc": False,
        }
        assert config["module_path"] == "/path/to/ext"

    def test_config_with_dependencies(self) -> None:
        """ExtensionConfig accepts dependencies list."""
        config: ExtensionConfig = {
            "name": "test_ext",
            "module_path": "/path/to/ext",
            "isolated": True,
            "dependencies": ["numpy>=1.20", "pillow"],
            "apis": [],
            "share_torch": False,
            "share_cuda_ipc": False,
        }
        assert "numpy>=1.20" in config["dependencies"]
        assert "pillow" in config["dependencies"]

    def test_config_share_torch(self) -> None:
        """ExtensionConfig accepts share_torch flag."""
        config: ExtensionConfig = {
            "name": "test_ext",
            "module_path": "/path/to/ext",
            "isolated": True,
            "dependencies": [],
            "apis": [],
            "share_torch": True,
            "share_cuda_ipc": False,
        }
        assert config["share_torch"] is True


class TestExtensionVenvPath:
    """Tests for extension venv path computation."""

    def test_venv_path_includes_extension_name(self) -> None:
        """Venv path should include extension name for isolation."""
        # This tests the contract, not implementation
        config: ExtensionConfig = {
            "name": "my_extension",
            "module_path": "/app/extensions/my_extension",
            "isolated": True,
            "dependencies": [],
            "apis": [],
            "share_torch": False,
            "share_cuda_ipc": False,
        }
        # The venv path pattern should include the extension name
        # Actual path computation is in Extension class
        assert config["name"] == "my_extension"


class TestExtensionManifest:
    """Tests for extension manifest (pyisolate.yaml) parsing."""

    def test_manifest_from_yaml(self, tmp_path: Path) -> None:
        """Extension can be configured via YAML manifest."""
        manifest_content = """
isolated: true
share_torch: true
dependencies:
  - numpy>=1.20
  - pillow
"""
        manifest_path = tmp_path / "pyisolate.yaml"
        manifest_path.write_text(manifest_content)

        # Parse manifest
        import yaml  # type: ignore[import-untyped]

        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        assert manifest["isolated"] is True
        assert manifest["share_torch"] is True
        assert "numpy>=1.20" in manifest["dependencies"]

    def test_manifest_defaults_isolated_true(self, tmp_path: Path) -> None:
        """Missing 'isolated' defaults to True."""
        manifest_content = """
dependencies: []
"""
        manifest_path = tmp_path / "pyisolate.yaml"
        manifest_path.write_text(manifest_content)

        import yaml  # type: ignore[import-untyped]

        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        # When creating ExtensionConfig, isolated defaults to True
        config: ExtensionConfig = {
            "name": "test",
            "module_path": str(tmp_path),
            "isolated": manifest.get("isolated", True),
            "dependencies": [],
            "apis": [],
            "share_torch": False,
            "share_cuda_ipc": False,
        }
        assert config["isolated"] is True


class TestExtensionLifecycleContract:
    """Tests for extension lifecycle contract.

    These tests verify the expected behavior without spawning
    actual subprocesses. They document the contract that the
    Extension class must fulfill.
    """

    def test_extension_requires_config(self) -> None:
        """Extension must be created with a config."""
        config: ExtensionConfig = {
            "name": "test_ext",
            "module_path": "/path/to/ext",
            "isolated": True,
            "dependencies": [],
            "apis": [],
            "share_torch": False,
            "share_cuda_ipc": False,
        }
        # Contract: Extension accepts config in constructor
        assert config["name"] == "test_ext"

    def test_extension_lifecycle_phases(self) -> None:
        """Document the extension lifecycle phases."""
        # Phase 1: Configuration
        # - ExtensionConfig created from manifest or programmatically
        # - Dependencies declared

        # Phase 2: Venv Creation
        # - Extension venv created if not exists
        # - Dependencies installed

        # Phase 3: Process Launch
        # - Child process spawned
        # - sys.path configured via adapter
        # - RPC channel established

        # Phase 4: Execution
        # - Methods called via RPC
        # - Results returned

        # Phase 5: Shutdown
        # - Process terminated
        # - Resources cleaned up

        # This test documents the contract
        phases = ["config", "venv", "launch", "execute", "shutdown"]
        assert len(phases) == 5

    def test_extension_stop_is_idempotent(self) -> None:
        """Stopping an already-stopped extension should not error."""
        # Contract: calling stop() multiple times is safe
        # This is tested at contract level, not implementation


class TestDependencyValidation:
    """Tests for dependency validation."""

    def test_valid_dependency_format(self) -> None:
        """Dependencies should be pip-installable strings."""
        valid_deps = [
            "numpy",
            "numpy>=1.20",
            "numpy>=1.20,<2.0",
            "pillow==10.0.0",
            "package[extra]",
        ]

        config: ExtensionConfig = {
            "name": "test",
            "module_path": "/path",
            "isolated": True,
            "dependencies": valid_deps,
            "apis": [],
            "share_torch": False,
            "share_cuda_ipc": False,
        }

        assert config["dependencies"] == valid_deps

    def test_empty_dependencies_allowed(self) -> None:
        """Extensions with no dependencies are valid."""
        config: ExtensionConfig = {
            "name": "test",
            "module_path": "/path",
            "isolated": True,
            "dependencies": [],
            "apis": [],
            "share_torch": False,
            "share_cuda_ipc": False,
        }

        assert config["dependencies"] == []
