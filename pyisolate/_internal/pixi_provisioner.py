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
from pathlib import Path

logger = logging.getLogger(__name__)

PIXI_VERSION = "0.67.0"

_PLATFORM_MAP = {
    ("Linux", "x86_64"): "x86_64-unknown-linux-musl",
    ("Linux", "aarch64"): "aarch64-unknown-linux-musl",
    ("Darwin", "x86_64"): "x86_64-apple-darwin",
    ("Darwin", "arm64"): "aarch64-apple-darwin",
    ("Windows", "AMD64"): "x86_64-pc-windows-msvc",
}

_RELEASE_URL = "https://github.com/prefix-dev/pixi/releases/download/v{version}/pixi-{target}.tar.gz"
_CHECKSUM_URL = "https://github.com/prefix-dev/pixi/releases/download/v{version}/pixi-{target}.tar.gz.sha256"


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
    cached_binary = cache / ("pixi.exe" if platform.system() == "Windows" else "pixi")
    if cached_binary.exists():
        return str(cached_binary)

    # Download
    target = _get_target()
    tarball_url = _RELEASE_URL.format(version=version, target=target)
    checksum_url = _CHECKSUM_URL.format(version=version, target=target)

    logger.info("Downloading pixi %s for %s...", version, target)

    checksum_data = _fetch_url(checksum_url)
    expected_hash = checksum_data.decode().strip().split()[0]

    tarball_data = _fetch_url(tarball_url)
    _verify_checksum(tarball_data, expected_hash)

    # Extract
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp.write(tarball_data)
        tmp_path = tmp.name

    try:
        with tarfile.open(tmp_path, "r:gz") as tf:
            members = tf.getnames()
            binary_name = "pixi.exe" if platform.system() == "Windows" else "pixi"
            if binary_name not in members:
                for m in members:
                    if m.endswith(binary_name):
                        binary_name = m
                        break
            tf.extract(binary_name, path=str(cache))
            extracted = cache / binary_name
            if extracted != cached_binary:
                extracted.rename(cached_binary)
    finally:
        os.unlink(tmp_path)

    # Make executable
    cached_binary.chmod(cached_binary.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    logger.info("pixi %s cached at %s", version, cached_binary)
    return str(cached_binary)
