"""Security tests for the pyisolate library."""

import pytest

from pyisolate._internal.host import normalize_extension_name, validate_dependency


class TestSecurityValidation:
    """Test security validation functions."""

    def test_normalize_extension_name_basic(self) -> None:
        """Test basic extension name normalization."""
        test_cases = [
            # (input, expected_output)
            ("extension1", "extension1"),  # Already safe
            ("my_extension", "my_extension"),  # Already safe
            ("ext-name", "ext-name"),  # Hyphens are safe
            ("extension.v2", "extension.v2"),  # Dots are safe
            ("MyExtension123", "MyExtension123"),  # Mixed case is safe
            ("test_ext_v1.2.3", "test_ext_v1.2.3"),  # Already safe
        ]
        for input_name, expected in test_cases:
            assert normalize_extension_name(input_name) == expected

    def test_normalize_extension_name_with_spaces(self) -> None:
        """Test that spaces are replaced with underscores."""
        test_cases = [
            ("my extension", "my_extension"),
            ("ext with spaces", "ext_with_spaces"),
            ("multiple  spaces", "multiple_spaces"),  # Multiple spaces become single underscore
            (" leading space", "leading_space"),
            ("trailing space ", "trailing_space"),
            ("tab\ttab", "tab_tab"),
            ("new\nline", "new_line"),
        ]
        for input_name, expected in test_cases:
            assert normalize_extension_name(input_name) == expected

    def test_normalize_extension_name_unicode(self) -> None:
        """Test that Unicode characters are preserved."""
        test_cases = [
            ("扩展名", "扩展名"),  # Chinese
            ("расширение", "расширение"),  # Russian
            ("café_extension", "café_extension"),  # Accented characters
            ("🔥hot_extension", "🔥hot_extension"),  # Emoji
            ("extension_日本語", "extension_日本語"),  # Mixed
        ]
        for input_name, expected in test_cases:
            assert normalize_extension_name(input_name) == expected

    def test_normalize_extension_name_dangerous_chars(self) -> None:
        """Test that dangerous characters are replaced."""
        test_cases = [
            ("ext|pipe", "ext_pipe"),
            ("ext`backtick`", "ext_backtick"),
            ("ext$(command)", "ext_command"),
            ("ext&background", "ext_background"),
            ("ext>redirect", "ext_redirect"),
            ("ext<redirect", "ext_redirect"),
            ("ext'quote'", "ext_quote"),
            ('ext"quote"', "ext_quote"),
            ("ext!history", "ext_history"),
            ("ext{brace}", "ext_brace"),
            ("ext[glob]", "ext_glob"),
            ("ext*star", "ext_star"),
            ("ext?question", "ext_question"),
            ("ext#comment", "ext_comment"),
            ("ext=equals", "ext_equals"),
            ("ext,comma", "ext_comma"),
        ]
        for input_name, expected in test_cases:
            assert normalize_extension_name(input_name) == expected

    def test_normalize_extension_name_path_traversal(self) -> None:
        """Test that path traversal attempts are neutralized."""
        test_cases = [
            ("../evil", "evil"),  # Dots at start removed
            ("./hidden", "hidden"),  # Dots at start removed
            ("/absolute/path", "absolute_path"),  # Slashes replaced
            ("..\\windows\\path", "windows_path"),  # Backslashes replaced
            ("ext/../../../tmp/test", "ext_tmp_test"),
            ("...dots", "dots"),  # Leading dots removed
        ]
        for input_name, expected in test_cases:
            assert normalize_extension_name(input_name) == expected

    def test_normalize_extension_name_edge_cases(self) -> None:
        """Test edge cases in normalization."""
        # Empty name should raise
        with pytest.raises(ValueError, match="Extension name cannot be empty"):
            normalize_extension_name("")

        # Only dots should be removed and result in empty string
        with pytest.raises(ValueError, match="contains only invalid characters"):
            normalize_extension_name("...")

        with pytest.raises(ValueError, match="contains only invalid characters"):
            normalize_extension_name("   ")  # Only spaces

        # Multiple dots become empty after normalization
        with pytest.raises(ValueError, match="contains only invalid characters"):
            normalize_extension_name("....")

    def test_validate_dependency_valid(self) -> None:
        """Test that valid dependencies pass validation."""
        valid_deps = [
            "numpy",
            "numpy>=1.21.0",
            "numpy>=1.21.0,<2.0.0",
            "requests[security]>=2.25.0",
            "torch==2.0.0+cpu",
            "my-package==1.0.0",
            "git+https://github.com/user/repo.git@v1.0",
            "-e",  # Special case for editable installs
            "package[extra1,extra2]>=1.0",
        ]
        for dep in valid_deps:
            validate_dependency(dep)  # Should not raise

    def test_validate_dependency_invalid(self) -> None:
        """Test that invalid dependencies are rejected."""
        invalid_cases = [
            ("--extra-index-url", "cannot start with '-'"),
            ("--trusted-host", "cannot start with '-'"),
            ("-f http://example.com", "cannot start with '-'"),
            ("numpy && echo test", "dangerous character: '&&'"),
            ("numpy || echo test", "dangerous character: '||'"),
            ("numpy | echo test", r"dangerous character: '\|'"),
            ("numpy`echo test`", "dangerous character: '`'"),
            ("numpy$(echo test)", r"dangerous character: '\$'"),
            ("numpy\ntest-package", "dangerous character: '\\n'"),
            ("numpy\rtest", "dangerous character: '\\r'"),
            ("numpy\x00test", "dangerous character: '\\x00'"),
        ]

        for dep, expected_msg in invalid_cases:
            with pytest.raises(ValueError, match=expected_msg):
                validate_dependency(dep)

    def test_dependency_injection_attempts(self) -> None:
        """Test various dependency injection attempts are blocked."""
        # Test that we can't inject extra index URLs
        with pytest.raises(ValueError):
            validate_dependency("--extra-index-url http://example.com")

        # Test that we can't inject trusted hosts
        with pytest.raises(ValueError):
            validate_dependency("--trusted-host example.com")

        # Test that we can't use shell operators
        with pytest.raises(ValueError):
            validate_dependency("numpy && echo test")

        # Test that we can't use command substitution
        with pytest.raises(ValueError):
            validate_dependency("numpy`echo test`")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
