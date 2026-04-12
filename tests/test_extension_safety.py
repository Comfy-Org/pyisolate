"""Tests for extension naming, dependency validation, and path safety."""

from typing import Any

import pytest

from pyisolate._internal import host


class TestNormalizeExtensionName:
    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            host.normalize_extension_name("")

    def test_strips_dangerous_chars(self) -> None:
        name = "../My Extension| rm -rf /"
        normalized = host.normalize_extension_name(name)
        assert ".." not in normalized
        assert "/" not in normalized
        assert " " not in normalized
        assert normalized == "My_Extension_rm_-rf"

    def test_preserves_unicode_and_collapses_underscores(self) -> None:
        name = "你好   世界"
        normalized = host.normalize_extension_name(name)
        assert normalized == "你好_世界"

    def test_raises_when_all_chars_invalid(self) -> None:
        with pytest.raises(ValueError):
            host.normalize_extension_name("////")


class TestValidateDependency:
    def test_allows_editable_flag(self) -> None:
        host.validate_dependency("-e")  # should not raise

    @pytest.mark.parametrize(
        "dependency",
        ["--option", "pkg|whoami", "pkg&&evil", "pkg`cmd`"],
    )
    def test_rejects_dangerous_patterns(self, dependency: Any) -> None:
        with pytest.raises(ValueError):
            host.validate_dependency(dependency)


class TestValidatePathWithinRoot:
    def test_allows_path_inside_root(self, tmp_path: Any) -> None:
        root = tmp_path
        inside = root / "child" / "module"
        inside.mkdir(parents=True)
        host.validate_path_within_root(inside, root)  # should not raise

    def test_rejects_path_outside_root(self, tmp_path: Any) -> None:
        root = tmp_path / "root"
        other = tmp_path / "other"
        root.mkdir()
        other.mkdir()
        with pytest.raises(ValueError):
            host.validate_path_within_root(other, root)
