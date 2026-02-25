"""Multi-distro sandbox capability detection for PyIsolate.

This module detects whether bubblewrap (bwrap) sandboxing is available on the
current system, identifying the specific restriction model in use and providing
distro-specific remediation instructions.

Supported restriction models:
- RHEL/CentOS: user.max_user_namespaces = 0
- Ubuntu 24.04+: kernel.apparmor_restrict_unprivileged_userns = 1
- Fedora: SELinux denials
- Arch Hardened: Hardened kernel without bwrap-suid
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SandboxMode(Enum):
    """Sandbox enforcement mode."""

    DISABLED = "disabled"  # Never use sandbox
    PREFERRED = "preferred"  # Use if available, warn and fallback if not
    REQUIRED = "required"  # Fail-loud if unavailable


class RestrictionModel(Enum):
    """Detected namespace restriction model."""

    NONE = "none"  # No restrictions detected
    RHEL_SYSCTL = "rhel_sysctl"  # max_user_namespaces = 0
    UBUNTU_APPARMOR = "ubuntu_apparmor"  # apparmor_restrict_unprivileged_userns = 1
    SELINUX = "selinux"  # SELinux denial
    ARCH_HARDENED = "arch_hardened"  # Hardened kernel
    PLATFORM_UNSUPPORTED = "platform"  # Non-Linux platform
    BWRAP_MISSING = "bwrap_missing"  # bwrap binary not found
    UNKNOWN = "unknown"  # Unknown restriction


# Distro-specific remediation messages
_REMEDIATION_MESSAGES: dict[RestrictionModel, str] = {
    RestrictionModel.RHEL_SYSCTL: (
        "Namespace limit is 0. Fix: "
        "echo 'user.max_user_namespaces=15000' | sudo tee /etc/sysctl.d/99-userns.conf && "
        "sudo sysctl -p"
    ),
    RestrictionModel.UBUNTU_APPARMOR: (
        "AppArmor restricts namespaces. Fix: "
        "sudo apt install apparmor-profiles && "
        "sudo ln -s /usr/share/apparmor/extra-profiles/bwrap-userns-restrict /etc/apparmor.d/bwrap && "
        "sudo apparmor_parser -r /etc/apparmor.d/bwrap"
    ),
    RestrictionModel.SELINUX: (
        "SELinux blocks namespace operations. Check: ausearch -m avc -ts recent | audit2allow"
    ),
    RestrictionModel.ARCH_HARDENED: ("Hardened kernel detected. Install: pacman -S bubblewrap-suid"),
    RestrictionModel.PLATFORM_UNSUPPORTED: ("Sandbox isolation requires Linux. Current platform: {platform}"),
    RestrictionModel.BWRAP_MISSING: (
        "bwrap binary not found. Install: apt install bubblewrap (Debian/Ubuntu) "
        "or dnf install bubblewrap (Fedora/RHEL)"
    ),
    RestrictionModel.UNKNOWN: ("Unknown restriction. bwrap test failed: {error}"),
    RestrictionModel.NONE: "",
}


@dataclass
class SandboxCapability:
    """Result of sandbox capability detection."""

    available: bool
    bwrap_path: str | None
    restriction_model: RestrictionModel
    remediation: str
    raw_error: str | None = None


def _read_sysctl(path: str) -> int | None:
    """Read an integer sysctl value from /proc/sys path."""
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError, PermissionError):
        return None


def _check_rhel_restriction() -> bool:
    """Check if RHEL-style namespace limit is blocking.

    Returns True if max_user_namespaces is 0 (blocked).
    """
    value = _read_sysctl("/proc/sys/user/max_user_namespaces")
    return value == 0


def _check_ubuntu_apparmor_restriction() -> bool:
    """Check if Ubuntu AppArmor namespace restriction is enabled.

    Returns True if restriction is enabled (default on Ubuntu 24.04+).
    """
    value = _read_sysctl("/proc/sys/kernel/apparmor_restrict_unprivileged_userns")
    return value == 1


def _check_selinux_enforcing() -> bool:
    """Check if SELinux is in enforcing mode."""
    try:
        # S607: getenforce is a standard SELinux utility, path varies by distro
        result = subprocess.run(
            ["getenforce"],  # noqa: S607
            capture_output=True,
            timeout=5,
        )
        return result.stdout.decode().strip().lower() == "enforcing"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _check_hardened_kernel() -> bool:
    """Check if running a hardened kernel (e.g., linux-hardened on Arch)."""
    try:
        with open("/proc/version") as f:
            version = f.read().lower()
            return "hardened" in version
    except FileNotFoundError:
        return False


def _test_bwrap(bwrap_path: str) -> tuple[bool, str]:
    """Test if bwrap actually works on this system.

    Returns (success, error_message).
    """
    try:
        # S603: bwrap_path comes from shutil.which(), not user input
        result = subprocess.run(  # noqa: S603
            [
                bwrap_path,
                "--unshare-user-try",
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--ro-bind",
                "/usr",
                "/usr",
                "--ro-bind",
                "/bin",
                "/bin",
                "--ro-bind",
                "/lib",
                "/lib",
                "--ro-bind",
                "/lib64",
                "/lib64",
                "/usr/bin/true",
            ],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True, ""
        return False, result.stderr.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return False, "bwrap test timed out"
    except Exception as exc:
        return False, str(exc)


def _test_bwrap_degraded(bwrap_path: str) -> tuple[bool, str]:
    """Test if bwrap works without user namespace isolation.

    This allows degraded sandbox mode on systems that block unprivileged
    user namespaces (for example Ubuntu AppArmor defaults).
    """
    try:
        # S603: bwrap_path comes from shutil.which(), not user input
        result = subprocess.run(  # noqa: S603
            [
                bwrap_path,
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--ro-bind",
                "/usr",
                "/usr",
                "--ro-bind",
                "/bin",
                "/bin",
                "--ro-bind",
                "/lib",
                "/lib",
                "--ro-bind",
                "/lib64",
                "/lib64",
                "/usr/bin/true",
            ],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True, ""
        return False, result.stderr.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return False, "bwrap degraded test timed out"
    except Exception as exc:
        return False, str(exc)


def _classify_error(error: str) -> RestrictionModel:
    """Classify a bwrap error message to determine restriction model."""
    error_lower = error.lower()

    if "permission denied" in error_lower or "uid map" in error_lower:
        # Could be AppArmor or SELinux
        if _check_ubuntu_apparmor_restriction():
            return RestrictionModel.UBUNTU_APPARMOR
        if _check_selinux_enforcing():
            return RestrictionModel.SELINUX
        return RestrictionModel.UNKNOWN

    if "no space left" in error_lower or "enospc" in error_lower:
        return RestrictionModel.RHEL_SYSCTL

    if "operation not permitted" in error_lower:
        if _check_hardened_kernel():
            return RestrictionModel.ARCH_HARDENED
        return RestrictionModel.UNKNOWN

    return RestrictionModel.UNKNOWN


def detect_sandbox_capability() -> SandboxCapability:
    """Detect sandbox capability with distro-specific diagnostics.

    Returns a SandboxCapability with:
    - available: True if bwrap sandbox can be used
    - bwrap_path: Path to bwrap binary (or None)
    - restriction_model: The type of restriction detected
    - remediation: Distro-specific fix instructions
    - raw_error: The raw error message from bwrap (if any)
    """
    # 1. Platform check
    if sys.platform != "linux":
        model = RestrictionModel.PLATFORM_UNSUPPORTED
        return SandboxCapability(
            available=False,
            bwrap_path=None,
            restriction_model=model,
            remediation=_REMEDIATION_MESSAGES[model].format(platform=sys.platform),
        )

    # 2. Find bwrap binary
    bwrap_path = shutil.which("bwrap")
    if bwrap_path is None:
        model = RestrictionModel.BWRAP_MISSING
        return SandboxCapability(
            available=False,
            bwrap_path=None,
            restriction_model=model,
            remediation=_REMEDIATION_MESSAGES[model],
        )

    # 3. Pre-flight checks (fast, avoids subprocess if obviously blocked)
    if _check_rhel_restriction():
        model = RestrictionModel.RHEL_SYSCTL
        return SandboxCapability(
            available=False,
            bwrap_path=bwrap_path,
            restriction_model=model,
            remediation=_REMEDIATION_MESSAGES[model],
        )

    # 4. Test actual bwrap invocation
    success, error = _test_bwrap(bwrap_path)

    if success:
        return SandboxCapability(
            available=True,
            bwrap_path=bwrap_path,
            restriction_model=RestrictionModel.NONE,
            remediation="",
        )

    # 5. Classify the failure
    model = _classify_error(error)
    remediation = _REMEDIATION_MESSAGES[model]

    # Try degraded mode on platforms that can still use mount-namespace sandboxing
    # even when user namespace creation is blocked.
    if model in {
        RestrictionModel.UBUNTU_APPARMOR,
        RestrictionModel.SELINUX,
        RestrictionModel.ARCH_HARDENED,
        RestrictionModel.UNKNOWN,
    }:
        degraded_success, degraded_error = _test_bwrap_degraded(bwrap_path)
        if degraded_success:
            return SandboxCapability(
                available=True,
                bwrap_path=bwrap_path,
                restriction_model=model,
                remediation=remediation,
                raw_error=error,
            )
        if degraded_error:
            error = f"{error} | degraded: {degraded_error}"

    if model == RestrictionModel.UNKNOWN:
        remediation = remediation.format(error=error[:200])

    return SandboxCapability(
        available=False,
        bwrap_path=bwrap_path,
        restriction_model=model,
        remediation=remediation,
        raw_error=error,
    )
