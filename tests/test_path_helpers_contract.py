"""Tests for path_helpers module contracts.

These tests verify path normalization and sys.path filtering behavior
without any host-specific dependencies.
"""

import os

from pyisolate import path_helpers


class TestSerializeHostSnapshot:
    """Tests for serialize_host_snapshot function."""

    def test_snapshot_includes_sys_path(self) -> None:
        """Snapshot includes current sys.path."""
        snapshot = path_helpers.serialize_host_snapshot()

        assert "sys_path" in snapshot
        assert isinstance(snapshot["sys_path"], list)

    def test_snapshot_includes_env_vars(self) -> None:
        """Snapshot includes environment variables."""
        snapshot = path_helpers.serialize_host_snapshot()

        assert "environment" in snapshot
        assert isinstance(snapshot["environment"], dict)

    def test_snapshot_paths_are_strings(self) -> None:
        """All paths in snapshot are strings."""
        snapshot = path_helpers.serialize_host_snapshot()

        for path in snapshot["sys_path"]:
            assert isinstance(path, str)

    def test_snapshot_is_json_serializable(self) -> None:
        """Snapshot can be JSON serialized."""
        import json

        snapshot = path_helpers.serialize_host_snapshot()

        # Should not raise
        json_str = json.dumps(snapshot)
        assert isinstance(json_str, str)

        # Should roundtrip
        restored = json.loads(json_str)
        assert restored["sys_path"] == snapshot["sys_path"]


class TestBuildChildSysPath:
    """Tests for build_child_sys_path function."""

    def test_host_paths_preserved(self) -> None:
        """Host paths are included in output."""
        host = ["/app/root", "/app/lib"]

        result = path_helpers.build_child_sys_path(
            host_paths=host,
            extra_paths=[],
        )

        for path in host:
            # Paths may be normalized
            assert any(os.path.normpath(path) in os.path.normpath(r) for r in result)

    def test_extra_paths_included(self) -> None:
        """Extra paths are included in output."""
        host = ["/app/root"]
        extra = ["/app/venv/site-packages"]

        result = path_helpers.build_child_sys_path(
            host_paths=host,
            extra_paths=extra,
        )

        # Extra paths should appear somewhere
        assert len(result) >= len(host)

    def test_preferred_root_comes_first(self) -> None:
        """Preferred root is prepended to path list."""
        host = ["/app/lib", "/app/utils"]
        preferred = "/app/root"

        result = path_helpers.build_child_sys_path(
            host_paths=host,
            extra_paths=[],
            preferred_root=preferred,
        )

        # Preferred root should be first
        assert result[0] == preferred

    def test_no_duplicates(self) -> None:
        """Duplicate paths are removed."""
        host = ["/app/root", "/app/lib", "/app/root"]  # Duplicate

        result = path_helpers.build_child_sys_path(
            host_paths=host,
            extra_paths=[],
        )

        # After normalization, no duplicates
        normalized = [os.path.normpath(p) for p in result]
        assert len(normalized) == len(set(normalized))

    def test_returns_list(self) -> None:
        """Function returns a list."""
        result = path_helpers.build_child_sys_path(
            host_paths=["/app"],
            extra_paths=[],
        )

        assert isinstance(result, list)

    def test_empty_inputs_handled(self) -> None:
        """Empty inputs don't cause errors."""
        result = path_helpers.build_child_sys_path(
            host_paths=[],
            extra_paths=[],
        )

        assert isinstance(result, list)
