"""Auto-provision the pixi binary for conda backend support.

Downloads, caches, and integrity-verifies the pixi binary from prefix-dev
GitHub releases. The binary is cached at ~/.cache/pyisolate/pixi/{version}/
and reused across runs.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import shutil
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

PIXI_VERSION = "0.67.0"

_PLATFORM_MAP = {
    ("Linux", "x86_64"): "x86_64-unknown-linux-musl",
    ("Linux", "aarch64"): "aarch64-unknown-linux-musl",
    ("Darwin", "x86_64"): "x86_64-apple-darwin",
    ("Darwin", "arm64"): "aarch64-apple-darwin",
    ("Windows", "AMD64"): "x86_64-pc-windows-msvc",
    ("Windows", "ARM64"): "aarch64-pc-windows-msvc",
}


def _binary_name() -> str:
    return "pixi.exe" if platform.system() == "Windows" else "pixi"


def _archive_extension() -> str:
    return ".zip" if platform.system() == "Windows" else ".tar.gz"


def _release_asset_name(target: str) -> str:
    return f"pixi-{target}{_archive_extension()}"


def _release_url(version: str, target: str) -> str:
    asset = _release_asset_name(target)
    return f"https://github.com/prefix-dev/pixi/releases/download/v{version}/{asset}"


def _checksum_url(version: str, target: str) -> str:
    asset = _release_asset_name(target)
    return f"https://github.com/prefix-dev/pixi/releases/download/v{version}/{asset}.sha256"


def _get_target() -> str:
    system = platform.system()
    machine = platform.machine()
    key = (system, machine)
    target = _PLATFORM_MAP.get(key)
    if not target:
        raise RuntimeError(
            f"Unsupported platform for pixi auto-provisioning: {system}/{machine}. "
            f"Supported: {', '.join(f'{s}/{m}' for s, m in _PLATFORM_MAP)}"
        )
    return target


def _cache_dir(version: str) -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "pyisolate" / "pixi" / version


def _fetch_url(url: str) -> bytes:
    """Download a URL using urllib (stdlib, no extra deps)."""
    import urllib.error
    import urllib.request

    if not url.startswith("https://"):
        raise RuntimeError(f"Unsupported pixi download URL scheme: {url}")

    req = urllib.request.Request(url, headers={"User-Agent": "pyisolate"})  # noqa: S310
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        data = resp.read()
        if not isinstance(data, bytes):
            raise RuntimeError(f"Unexpected pixi download payload type: {type(data).__name__}")
        return data


def _verify_checksum(data: bytes, expected_hex: str) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_hex:
        raise RuntimeError(f"pixi binary checksum mismatch: expected {expected_hex}, got {actual}")


def _safe_extract_member(
    *,
    cache: Path,
    member_name: str,
    binary_name: str,
    data: bytes,
    mode: int | None = None,
) -> Path:
    relative_member = Path(member_name)
    target = (cache / relative_member.name).resolve()
    cache_root = cache.resolve()
    if target.parent != cache_root:
        raise RuntimeError(f"Unsafe pixi archive path: {member_name}")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    if mode is not None:
        target.chmod(mode)
    elif target.name == binary_name:
        target.chmod(target.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return target


def ensure_pixi(version: str | None = None) -> str:
    """Return path to pixi binary, downloading if necessary.

    1. Check if pixi is already on PATH and matches the pinned version.
    2. Check the cache directory for a previously downloaded binary.
    3. Download from GitHub releases, verify checksum, cache, and return.
    """
    version = version or PIXI_VERSION

    # Check PATH first
    existing = shutil.which("pixi")
    if existing:
        try:
            result = subprocess.run([existing, "--version"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and version in result.stdout:
                return existing
        except (subprocess.TimeoutExpired, OSError):
            pass

    # Check cache
    cache = _cache_dir(version)
    cached_binary = cache / _binary_name()
    if cached_binary.exists():
        return str(cached_binary)

    # Download
    target = _get_target()
    archive_url = _release_url(version, target)
    checksum_url = _checksum_url(version, target)
    archive_extension = _archive_extension()

    logger.info("Downloading pixi %s for %s...", version, target)

    checksum_data = _fetch_url(checksum_url)
    expected_hash = checksum_data.decode().strip().split()[0]

    archive_data = _fetch_url(archive_url)
    _verify_checksum(archive_data, expected_hash)

    # Extract
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=archive_extension, delete=False) as tmp:
        tmp.write(archive_data)
        tmp_path = tmp.name

    try:
        binary_name = _binary_name()
        if archive_extension == ".zip":
            with zipfile.ZipFile(tmp_path) as zf:
                members = zf.namelist()
                member_name = binary_name
                if member_name not in members:
                    for member in members:
                        if member.endswith(binary_name):
                            member_name = member
                            break
                with zf.open(member_name) as member_fp:
                    extracted = _safe_extract_member(
                        cache=cache,
                        member_name=member_name,
                        binary_name=binary_name,
                        data=member_fp.read(),
                    )
        else:
            with tarfile.open(tmp_path, "r:gz") as tf:
                members = tf.getnames()
                member_name = binary_name
                if member_name not in members:
                    for member in members:
                        if member.endswith(binary_name):
                            member_name = member
                            break
                tar_member = tf.getmember(member_name)
                tar_member_fp = tf.extractfile(tar_member)
                if tar_member_fp is None:
                    raise RuntimeError(f"pixi archive member has no data: {member_name}")
                with tar_member_fp:
                    extracted = _safe_extract_member(
                        cache=cache,
                        member_name=member_name,
                        binary_name=binary_name,
                        data=tar_member_fp.read(),
                        mode=tar_member.mode,
                    )

        if extracted != cached_binary:
            extracted.rename(cached_binary)
    finally:
        os.unlink(tmp_path)

    # Make executable
    cached_binary.chmod(cached_binary.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    logger.info("pixi %s cached at %s", version, cached_binary)
    return str(cached_binary)
