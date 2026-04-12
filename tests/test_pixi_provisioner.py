"""Tests for pixi binary auto-provisioner."""

from __future__ import annotations

import hashlib
import io
import os
import platform
import tarfile
import tempfile
import zipfile
from contextlib import closing
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pyisolate._internal.pixi_provisioner import (
    PIXI_VERSION,
    _archive_extension,
    _binary_name,
    _get_target,
    _release_url,
    _verify_checksum,
    ensure_pixi,
)


class TestGetTarget:
    def test_linux_x86_64(self) -> None:
        with patch("platform.system", return_value="Linux"), patch("platform.machine", return_value="x86_64"):
            assert _get_target() == "x86_64-unknown-linux-musl"

    def test_darwin_arm64(self) -> None:
        with patch("platform.system", return_value="Darwin"), patch("platform.machine", return_value="arm64"):
            assert _get_target() == "aarch64-apple-darwin"

    def test_unsupported_raises(self) -> None:
        with (
            patch("platform.system", return_value="FreeBSD"),
            patch("platform.machine", return_value="sparc"),
            pytest.raises(RuntimeError, match="Unsupported platform"),
        ):
            _get_target()


class TestPlatformCoverage:
    """All 5 supported platform/arch combinations."""

    @pytest.mark.parametrize(
        "system,machine,expected_target",
        [
            ("Linux", "x86_64", "x86_64-unknown-linux-musl"),
            ("Linux", "aarch64", "aarch64-unknown-linux-musl"),
            ("Darwin", "x86_64", "x86_64-apple-darwin"),
            ("Darwin", "arm64", "aarch64-apple-darwin"),
            ("Windows", "AMD64", "x86_64-pc-windows-msvc"),
        ],
    )
    def test_platform_target_string(self, system: Any, machine: Any, expected_target: Any) -> None:
        with patch("platform.system", return_value=system), patch("platform.machine", return_value=machine):
            result = _get_target()
            assert result == expected_target
            print(f"PLATFORM={system}/{machine} TARGET={result}")

    def test_windows_binary_name(self, tmp_path: Any) -> None:
        """ensure_pixi() uses pixi.exe on Windows."""
        version = PIXI_VERSION
        cache = tmp_path / "pyisolate" / "pixi" / version
        cache.mkdir(parents=True)
        cached_bin = cache / "pixi.exe"
        cached_bin.write_bytes(b"fake windows binary")

        with (
            patch("shutil.which", return_value=None),
            patch("platform.system", return_value="Windows"),
            patch("platform.machine", return_value="AMD64"),
            patch("pyisolate._internal.pixi_provisioner._cache_dir", return_value=cache),
        ):
            result = ensure_pixi(version)
            assert result == str(cached_bin)
            assert result.endswith("pixi.exe")
            print(f"WINDOWS_BINARY_PATH={result}")

    @pytest.mark.parametrize(
        "system,machine,expected_target",
        [
            ("Linux", "x86_64", "x86_64-unknown-linux-musl"),
            ("Linux", "aarch64", "aarch64-unknown-linux-musl"),
            ("Darwin", "x86_64", "x86_64-apple-darwin"),
            ("Darwin", "arm64", "aarch64-apple-darwin"),
            ("Windows", "AMD64", "x86_64-pc-windows-msvc"),
        ],
    )
    def test_url_construction(self, system: Any, machine: Any, expected_target: Any) -> None:
        """Download URL matches GitHub release asset naming convention."""
        with patch("platform.system", return_value=system), patch("platform.machine", return_value=machine):
            url = _release_url(PIXI_VERSION, _get_target())
            ext = ".zip" if system == "Windows" else ".tar.gz"
            expected_url = (
                f"https://github.com/prefix-dev/pixi/releases/download/v{PIXI_VERSION}/"
                f"pixi-{expected_target}{ext}"
            )
            assert url == expected_url
            print(f"URL={url}")


class TestVerifyChecksum:
    def test_valid_checksum(self) -> None:
        data = b"test binary content"
        expected = hashlib.sha256(data).hexdigest()
        _verify_checksum(data, expected)

    def test_invalid_checksum_raises(self) -> None:
        data = b"test binary content"
        with pytest.raises(RuntimeError, match="checksum"):
            _verify_checksum(data, "0" * 64)


class TestEnsurePixi:
    def test_returns_path_on_system(self, tmp_path: Any) -> None:
        """If pixi is on PATH and version matches, return that path."""
        fake_pixi = tmp_path / "pixi"
        fake_pixi.touch()
        fake_pixi.chmod(0o755)

        with patch("shutil.which", return_value=str(fake_pixi)), patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=f"pixi {PIXI_VERSION}\n")
            result = ensure_pixi()
            assert result == str(fake_pixi)
            print(f"RESOLVED_PATH={result}")

    def test_returns_cached_binary(self, tmp_path: Any) -> None:
        """If cached binary exists, return it without downloading."""
        version = PIXI_VERSION
        cache = tmp_path / "pyisolate" / "pixi" / version
        cache.mkdir(parents=True)
        cached_bin = cache / _binary_name()
        cached_bin.write_bytes(b"cached binary")

        with (
            patch("shutil.which", return_value=None),
            patch("pyisolate._internal.pixi_provisioner._cache_dir", return_value=cache),
        ):
            result = ensure_pixi(version)
            assert result == str(cached_bin)

    def test_cache_hit_no_http(self, tmp_path: Any) -> None:
        """ensure_pixi called twice: second call makes zero HTTP requests."""
        version = PIXI_VERSION
        cache = tmp_path / "pyisolate" / "pixi" / version
        cache.mkdir(parents=True)
        cached_bin = cache / _binary_name()
        cached_bin.write_bytes(b"cached binary")

        fetch_mock = MagicMock()

        with (
            patch("shutil.which", return_value=None),
            patch("pyisolate._internal.pixi_provisioner._cache_dir", return_value=cache),
            patch("pyisolate._internal.pixi_provisioner._fetch_url", fetch_mock),
        ):
            # First call — cache hit, no fetch
            result1 = ensure_pixi(version)
            assert result1 == str(cached_bin)
            assert fetch_mock.call_count == 0

            # Second call — still cache hit
            result2 = ensure_pixi(version)
            assert result2 == str(cached_bin)
            assert fetch_mock.call_count == 0
            print("HTTP_REQUESTS_ON_SECOND_CALL=0")

    def test_corrupted_download_raises(self, tmp_path: Any) -> None:
        """Corrupted download triggers checksum RuntimeError."""
        version = PIXI_VERSION
        cache = tmp_path / "pyisolate" / "pixi" / version

        # Simulate: no pixi on PATH, no cache, download gives bad data
        archive_name = f"pixi{_archive_extension()}"
        with (
            patch("shutil.which", return_value=None),
            patch("pyisolate._internal.pixi_provisioner._cache_dir", return_value=cache),
            patch("pyisolate._internal.pixi_provisioner._fetch_url") as fetch_mock,
        ):
            # First call returns checksum, second returns tarball with wrong content
            good_hash = hashlib.sha256(b"good data").hexdigest()
            fetch_mock.side_effect = [
                f"{good_hash}  {archive_name}".encode(),  # checksum file
                b"corrupted tarball data",  # tarball (wrong content)
            ]

            with pytest.raises(RuntimeError, match="checksum"):
                ensure_pixi(version)

    def test_downloads_and_caches(self, tmp_path: Any) -> None:
        """Full download path: fetch, verify, extract, cache."""
        version = PIXI_VERSION
        cache = tmp_path / "pyisolate" / "pixi" / version

        # Create a real archive with a fake pixi binary.
        fake_binary = b"#!/bin/sh\necho pixi"
        binary_name = _binary_name()
        archive_extension = _archive_extension()
        with closing(tempfile.NamedTemporaryFile(suffix=archive_extension, delete=False)) as archive_buf:
            archive_path = Path(archive_buf.name)
        if platform.system() == "Windows":
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr(binary_name, fake_binary)
        else:
            with tarfile.open(archive_path, "w:gz") as tf:
                info = tarfile.TarInfo(name=binary_name)
                info.size = len(fake_binary)
                info.mode = 0o755
                tf.addfile(info, io.BytesIO(fake_binary))

        archive_data = archive_path.read_bytes()
        os.unlink(archive_path)

        archive_hash = hashlib.sha256(archive_data).hexdigest()
        archive_name = f"pixi{archive_extension}"

        with (
            patch("shutil.which", return_value=None),
            patch("pyisolate._internal.pixi_provisioner._cache_dir", return_value=cache),
            patch("pyisolate._internal.pixi_provisioner._fetch_url") as fetch_mock,
        ):
            fetch_mock.side_effect = [
                f"{archive_hash}  {archive_name}".encode(),
                archive_data,
            ]
            result = ensure_pixi(version)
            assert Path(result).exists()
            assert Path(result).read_bytes() == fake_binary
            print(f"RESOLVED_PATH={result}")


class TestVersionPin:
    def test_version_is_pinned(self) -> None:
        """PIXI_VERSION is a hardcoded string, not dynamic."""
        assert isinstance(PIXI_VERSION, str)
        assert len(PIXI_VERSION.split(".")) >= 2
        assert PIXI_VERSION == "0.67.0"
