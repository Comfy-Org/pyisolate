"""Unit tests for sandbox capability detection.

Tests cover:
- Sysctl file reading
- RHEL/Ubuntu restriction detection
- SELinux and hardened kernel checks
- bwrap binary invocation
- Error classification
- Full detection flow
"""

import subprocess
import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest

from pyisolate._internal.sandbox_detect import (
    RestrictionModel,
    SandboxCapability,
    _check_hardened_kernel,
    _check_rhel_restriction,
    _check_selinux_enforcing,
    _check_ubuntu_apparmor_restriction,
    _classify_error,
    _read_sysctl,
    _test_bwrap,
    _test_bwrap_degraded,
    detect_sandbox_capability,
)


class TestSysctlReaders:
    """Test low-level sysctl reading functions."""

    def test_read_sysctl_success(self) -> None:
        """Test successful sysctl read."""
        m = mock_open(read_data="15000\n")
        with patch("builtins.open", m):
            assert _read_sysctl("/proc/sys/user/max_user_namespaces") == 15000

    def test_read_sysctl_file_missing(self) -> None:
        """Test when sysctl file doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert _read_sysctl("/proc/sys/nonexistent") is None

    def test_read_sysctl_permission_denied(self) -> None:
        """Test when sysctl file is not readable."""
        with patch("builtins.open", side_effect=PermissionError):
            assert _read_sysctl("/proc/sys/restricted") is None

    def test_read_sysctl_invalid_value(self) -> None:
        """Test when sysctl contains non-integer."""
        m = mock_open(read_data="not_a_number\n")
        with patch("builtins.open", m):
            assert _read_sysctl("/proc/sys/something") is None

    def test_rhel_restriction_detected(self) -> None:
        """Test RHEL sysctl restriction (max_user_namespaces=0)."""
        m = mock_open(read_data="0")
        with patch("builtins.open", m):
            assert _check_rhel_restriction() is True

    def test_rhel_restriction_not_present(self) -> None:
        """Test when RHEL restriction is not present."""
        m = mock_open(read_data="15000")
        with patch("builtins.open", m):
            assert _check_rhel_restriction() is False

    def test_rhel_restriction_file_missing(self) -> None:
        """Test when sysctl file doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert _check_rhel_restriction() is False

    def test_ubuntu_apparmor_detected(self) -> None:
        """Test Ubuntu AppArmor restriction."""
        m = mock_open(read_data="1")
        with patch("builtins.open", m):
            assert _check_ubuntu_apparmor_restriction() is True

    def test_ubuntu_apparmor_not_present(self) -> None:
        """Test when Ubuntu AppArmor is not enabled."""
        m = mock_open(read_data="0")
        with patch("builtins.open", m):
            assert _check_ubuntu_apparmor_restriction() is False

    def test_ubuntu_apparmor_file_missing(self) -> None:
        """Test when AppArmor sysctl doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert _check_ubuntu_apparmor_restriction() is False


class TestKernelChecks:
    """Test kernel feature detection."""

    def test_selinux_enforcing(self) -> None:
        """Test SELinux enforcing detection."""
        mock_result = MagicMock()
        mock_result.stdout = b"Enforcing\n"
        with patch("subprocess.run", return_value=mock_result):
            assert _check_selinux_enforcing() is True

    def test_selinux_permissive(self) -> None:
        """Test SELinux permissive mode."""
        mock_result = MagicMock()
        mock_result.stdout = b"Permissive\n"
        with patch("subprocess.run", return_value=mock_result):
            assert _check_selinux_enforcing() is False

    def test_selinux_disabled(self) -> None:
        """Test SELinux disabled mode."""
        mock_result = MagicMock()
        mock_result.stdout = b"Disabled\n"
        with patch("subprocess.run", return_value=mock_result):
            assert _check_selinux_enforcing() is False

    def test_selinux_not_installed(self) -> None:
        """Test when getenforce command doesn't exist."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _check_selinux_enforcing() is False

    def test_selinux_timeout(self) -> None:
        """Test when getenforce times out."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("getenforce", 5)):
            assert _check_selinux_enforcing() is False

    def test_hardened_kernel_detected(self) -> None:
        """Test hardened kernel detection."""
        m = mock_open(read_data="Linux version 5.15.0-hardened-x86_64")
        with patch("builtins.open", m):
            assert _check_hardened_kernel() is True

    def test_hardened_kernel_not_present(self) -> None:
        """Test standard kernel."""
        m = mock_open(read_data="Linux version 5.15.0-generic-x86_64")
        with patch("builtins.open", m):
            assert _check_hardened_kernel() is False

    def test_hardened_kernel_file_missing(self) -> None:
        """Test when /proc/version doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert _check_hardened_kernel() is False


class TestBwrapInvocation:
    """Test bwrap binary invocation and error handling."""

    def test_bwrap_test_success(self) -> None:
        """Test successful bwrap invocation."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            success, error = _test_bwrap("/usr/bin/bwrap")
            assert success is True
            assert error == ""

    def test_bwrap_test_uses_unshare_user_try(self) -> None:
        """Test that bwrap test uses --unshare-user-try flag."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            _test_bwrap("/usr/bin/bwrap")
            args = mock_run.call_args[0][0]
            assert "--unshare-user-try" in args

    def test_bwrap_test_failure_permission(self) -> None:
        """Test bwrap failure with permission denied."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = b"Permission denied: uid map"
        with patch("subprocess.run", return_value=mock_result):
            success, error = _test_bwrap("/usr/bin/bwrap")
            assert success is False
            assert "Permission denied" in error

    def test_bwrap_test_timeout(self) -> None:
        """Test bwrap test timeout."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("bwrap", 10)):
            success, error = _test_bwrap("/usr/bin/bwrap")
            assert success is False
            assert "timed out" in error.lower()

    def test_bwrap_test_exception(self) -> None:
        """Test bwrap test with unexpected exception."""
        with patch("subprocess.run", side_effect=Exception("Unexpected error")):
            success, error = _test_bwrap("/usr/bin/bwrap")
            assert success is False
            assert "Unexpected error" in error

    def test_bwrap_degraded_test_success(self) -> None:
        """Test successful degraded bwrap invocation."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            success, error = _test_bwrap_degraded("/usr/bin/bwrap")
            assert success is True
            assert error == ""


class TestErrorClassification:
    """Test error message classification."""

    def test_classify_apparmor_error(self) -> None:
        """Test AppArmor error classification."""
        with patch(
            "pyisolate._internal.sandbox_detect._check_ubuntu_apparmor_restriction",
            return_value=True,
        ):
            model = _classify_error("Permission denied: uid map")
            assert model == RestrictionModel.UBUNTU_APPARMOR

    def test_classify_selinux_error(self) -> None:
        """Test SELinux error classification."""
        with (
            patch(
                "pyisolate._internal.sandbox_detect._check_ubuntu_apparmor_restriction",
                return_value=False,
            ),
            patch(
                "pyisolate._internal.sandbox_detect._check_selinux_enforcing",
                return_value=True,
            ),
        ):
            model = _classify_error("Permission denied")
            assert model == RestrictionModel.SELINUX

    def test_classify_rhel_sysctl_error(self) -> None:
        """Test RHEL sysctl error classification."""
        model = _classify_error("No space left on device")
        assert model == RestrictionModel.RHEL_SYSCTL

    def test_classify_rhel_enospc_error(self) -> None:
        """Test RHEL ENOSPC error classification."""
        model = _classify_error("ENOSPC")
        assert model == RestrictionModel.RHEL_SYSCTL

    def test_classify_hardened_kernel_error(self) -> None:
        """Test hardened kernel error classification."""
        with patch(
            "pyisolate._internal.sandbox_detect._check_hardened_kernel",
            return_value=True,
        ):
            model = _classify_error("Operation not permitted")
            assert model == RestrictionModel.ARCH_HARDENED

    def test_classify_operation_not_permitted_non_hardened(self) -> None:
        """Test operation not permitted on non-hardened kernel."""
        with patch(
            "pyisolate._internal.sandbox_detect._check_hardened_kernel",
            return_value=False,
        ):
            model = _classify_error("Operation not permitted")
            assert model == RestrictionModel.UNKNOWN

    def test_classify_unknown_error(self) -> None:
        """Test unknown error classification."""
        model = _classify_error("Some weird error")
        assert model == RestrictionModel.UNKNOWN

    def test_classify_permission_denied_neither_apparmor_nor_selinux(self) -> None:
        """Test permission denied when neither AppArmor nor SELinux."""
        with (
            patch(
                "pyisolate._internal.sandbox_detect._check_ubuntu_apparmor_restriction",
                return_value=False,
            ),
            patch(
                "pyisolate._internal.sandbox_detect._check_selinux_enforcing",
                return_value=False,
            ),
        ):
            model = _classify_error("Permission denied")
            assert model == RestrictionModel.UNKNOWN


class TestFullDetection:
    """Integration tests for full detection flow."""

    def test_platform_check_non_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that non-Linux platforms return PLATFORM_UNSUPPORTED."""
        monkeypatch.setattr(sys, "platform", "darwin")
        cap = detect_sandbox_capability()
        assert cap.available is False
        assert cap.restriction_model == RestrictionModel.PLATFORM_UNSUPPORTED
        assert "darwin" in cap.remediation

    def test_platform_check_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that Windows returns PLATFORM_UNSUPPORTED."""
        monkeypatch.setattr(sys, "platform", "win32")
        cap = detect_sandbox_capability()
        assert cap.available is False
        assert cap.restriction_model == RestrictionModel.PLATFORM_UNSUPPORTED

    def test_bwrap_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that missing bwrap binary returns BWRAP_MISSING."""
        monkeypatch.setattr(sys, "platform", "linux")
        with patch("shutil.which", return_value=None):
            cap = detect_sandbox_capability()
            assert cap.available is False
            assert cap.restriction_model == RestrictionModel.BWRAP_MISSING
            assert "bubblewrap" in cap.remediation.lower()

    def test_rhel_restriction_blocks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test RHEL sysctl blocks before bwrap test."""
        monkeypatch.setattr(sys, "platform", "linux")
        with (
            patch("shutil.which", return_value="/usr/bin/bwrap"),
            patch(
                "pyisolate._internal.sandbox_detect._check_rhel_restriction",
                return_value=True,
            ),
        ):
            cap = detect_sandbox_capability()
            assert cap.available is False
            assert cap.restriction_model == RestrictionModel.RHEL_SYSCTL
            assert cap.bwrap_path == "/usr/bin/bwrap"

    def test_full_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test full detection success path."""
        monkeypatch.setattr(sys, "platform", "linux")
        with (
            patch("shutil.which", return_value="/usr/bin/bwrap"),
            patch(
                "pyisolate._internal.sandbox_detect._check_rhel_restriction",
                return_value=False,
            ),
            patch(
                "pyisolate._internal.sandbox_detect._test_bwrap",
                return_value=(True, ""),
            ),
        ):
            cap = detect_sandbox_capability()
            assert cap.available is True
            assert cap.restriction_model == RestrictionModel.NONE
            assert cap.bwrap_path == "/usr/bin/bwrap"
            assert cap.remediation == ""

    def test_ubuntu_apparmor_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Ubuntu AppArmor detection with degraded-mode fallback."""
        monkeypatch.setattr(sys, "platform", "linux")
        with (
            patch("shutil.which", return_value="/usr/bin/bwrap"),
            patch(
                "pyisolate._internal.sandbox_detect._check_rhel_restriction",
                return_value=False,
            ),
            patch(
                "pyisolate._internal.sandbox_detect._test_bwrap",
                return_value=(False, "Permission denied: uid map"),
            ),
            patch(
                "pyisolate._internal.sandbox_detect._check_ubuntu_apparmor_restriction",
                return_value=True,
            ),
            patch(
                "pyisolate._internal.sandbox_detect._test_bwrap_degraded",
                return_value=(True, ""),
            ),
        ):
            cap = detect_sandbox_capability()
            assert cap.available is True
            assert cap.restriction_model == RestrictionModel.UBUNTU_APPARMOR
            assert "apparmor" in cap.remediation.lower()
            assert cap.raw_error == "Permission denied: uid map"

    def test_ubuntu_apparmor_failure_when_degraded_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Ubuntu AppArmor detection when degraded fallback also fails."""
        monkeypatch.setattr(sys, "platform", "linux")
        with (
            patch("shutil.which", return_value="/usr/bin/bwrap"),
            patch(
                "pyisolate._internal.sandbox_detect._check_rhel_restriction",
                return_value=False,
            ),
            patch(
                "pyisolate._internal.sandbox_detect._test_bwrap",
                return_value=(False, "Permission denied: uid map"),
            ),
            patch(
                "pyisolate._internal.sandbox_detect._check_ubuntu_apparmor_restriction",
                return_value=True,
            ),
            patch(
                "pyisolate._internal.sandbox_detect._test_bwrap_degraded",
                return_value=(False, "still blocked"),
            ),
        ):
            cap = detect_sandbox_capability()
            assert cap.available is False
            assert cap.restriction_model == RestrictionModel.UBUNTU_APPARMOR
            assert "degraded: still blocked" in (cap.raw_error or "")

    def test_unknown_error_includes_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that unknown errors include the raw error in remediation."""
        monkeypatch.setattr(sys, "platform", "linux")
        with (
            patch("shutil.which", return_value="/usr/bin/bwrap"),
            patch(
                "pyisolate._internal.sandbox_detect._check_rhel_restriction",
                return_value=False,
            ),
            patch(
                "pyisolate._internal.sandbox_detect._test_bwrap",
                return_value=(False, "Some weird unknown error"),
            ),
            patch(
                "pyisolate._internal.sandbox_detect._classify_error",
                return_value=RestrictionModel.UNKNOWN,
            ),
            patch(
                "pyisolate._internal.sandbox_detect._test_bwrap_degraded",
                return_value=(False, "still blocked"),
            ),
        ):
            cap = detect_sandbox_capability()
            assert cap.available is False
            assert cap.restriction_model == RestrictionModel.UNKNOWN
            assert "weird unknown" in cap.remediation

    def test_capability_dataclass_fields(self) -> None:
        """Test SandboxCapability dataclass has expected fields."""
        cap = SandboxCapability(
            available=True,
            bwrap_path="/usr/bin/bwrap",
            restriction_model=RestrictionModel.NONE,
            remediation="",
            raw_error=None,
        )
        assert cap.available is True
        assert cap.bwrap_path == "/usr/bin/bwrap"
        assert cap.restriction_model == RestrictionModel.NONE
        assert cap.remediation == ""
        assert cap.raw_error is None


class TestRestrictionModelEnum:
    """Test RestrictionModel enum values."""

    def test_all_models_have_remediation(self) -> None:
        """Ensure all restriction models have remediation messages."""
        from pyisolate._internal.sandbox_detect import _REMEDIATION_MESSAGES

        for model in RestrictionModel:
            assert model in _REMEDIATION_MESSAGES, f"Missing remediation for {model}"

    def test_none_has_empty_remediation(self) -> None:
        """Test that NONE restriction has empty remediation."""
        from pyisolate._internal.sandbox_detect import _REMEDIATION_MESSAGES

        assert _REMEDIATION_MESSAGES[RestrictionModel.NONE] == ""
