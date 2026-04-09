"""Tests for pixi binary auto-provisioner."""

from __future__ import annotations

import hashlib
import os
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyisolate._internal.pixi_provisioner import (
    PIXI_VERSION,
    _cache_dir,
    _get_target,
    _verify_checksum,
    ensure_pixi,
)


class TestGetTarget:
    def test_linux_x86_64(self):
        with patch("platform.system", return_value="Linux"), \
             patch("platform.machine", return_value="x86_64"):
            assert _get_target() == "x86_64-unknown-linux-musl"

    def test_darwin_arm64(self):
        with patch("platform.system", return_value="Darwin"), \
             patch("platform.machine", return_value="arm64"):
            assert _get_target() == "aarch64-apple-darwin"

    def test_unsupported_raises(self):
        with patch("platform.system", return_value="FreeBSD"), \
             patch("platform.machine", return_value="sparc"), \
             pytest.raises(RuntimeError, match="Unsupported platform"):
            _get_target()


class TestVerifyChecksum:
    def test_valid_checksum(self):
        data = b"test binary content"
        expected = hashlib.sha256(data).hexdigest()
        _verify_checksum(data, expected)

    def test_invalid_checksum_raises(self):
        data = b"test binary content"
        with pytest.raises(RuntimeError, match="checksum"):
            _verify_checksum(data, "0" * 64)


class TestEnsurePixi:
    def test_returns_path_on_system(self, tmp_path):
        """If pixi is on PATH and version matches, return that path."""
        fake_pixi = tmp_path / "pixi"
        fake_pixi.touch()
        fake_pixi.chmod(0o755)

        with patch("shutil.which", return_value=str(fake_pixi)), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=f"pixi {PIXI_VERSION}\n"
            )
            result = ensure_pixi()
            assert result == str(fake_pixi)
            print(f"RESOLVED_PATH={result}")

    def test_returns_cached_binary(self, tmp_path):
        """If cached binary exists, return it without downloading."""
        version = PIXI_VERSION
        cache = tmp_path / "pyisolate" / "pixi" / version
        cache.mkdir(parents=True)
        cached_bin = cache / "pixi"
        cached_bin.write_bytes(b"cached binary")

        with patch("shutil.which", return_value=None), \
             patch("pyisolate._internal.pixi_provisioner._cache_dir", return_value=cache):
            result = ensure_pixi(version)
            assert result == str(cached_bin)

    def test_cache_hit_no_http(self, tmp_path):
        """ensure_pixi called twice: second call makes zero HTTP requests."""
        version = PIXI_VERSION
        cache = tmp_path / "pyisolate" / "pixi" / version
        cache.mkdir(parents=True)
        cached_bin = cache / "pixi"
        cached_bin.write_bytes(b"cached binary")

        fetch_mock = MagicMock()

        with patch("shutil.which", return_value=None), \
             patch("pyisolate._internal.pixi_provisioner._cache_dir", return_value=cache), \
             patch("pyisolate._internal.pixi_provisioner._fetch_url", fetch_mock):
            # First call — cache hit, no fetch
            result1 = ensure_pixi(version)
            assert result1 == str(cached_bin)
            assert fetch_mock.call_count == 0

            # Second call — still cache hit
            result2 = ensure_pixi(version)
            assert result2 == str(cached_bin)
            assert fetch_mock.call_count == 0
            print(f"HTTP_REQUESTS_ON_SECOND_CALL=0")

    def test_corrupted_download_raises(self, tmp_path):
        """Corrupted download triggers checksum RuntimeError."""
        version = PIXI_VERSION
        cache = tmp_path / "pyisolate" / "pixi" / version

        # Simulate: no pixi on PATH, no cache, download gives bad data
        with patch("shutil.which", return_value=None), \
             patch("pyisolate._internal.pixi_provisioner._cache_dir", return_value=cache), \
             patch("pyisolate._internal.pixi_provisioner._fetch_url") as fetch_mock:
            # First call returns checksum, second returns tarball with wrong content
            good_hash = hashlib.sha256(b"good data").hexdigest()
            fetch_mock.side_effect = [
                f"{good_hash}  pixi.tar.gz".encode(),  # checksum file
                b"corrupted tarball data",               # tarball (wrong content)
            ]

            with pytest.raises(RuntimeError, match="checksum"):
                ensure_pixi(version)

    def test_downloads_and_caches(self, tmp_path):
        """Full download path: fetch, verify, extract, cache."""
        version = PIXI_VERSION
        cache = tmp_path / "pyisolate" / "pixi" / version

        # Create a real tarball with a fake pixi binary
        fake_binary = b"#!/bin/sh\necho pixi"
        tarball_buf = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
        with tarfile.open(tarball_buf.name, "w:gz") as tf:
            import io
            info = tarfile.TarInfo(name="pixi")
            info.size = len(fake_binary)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(fake_binary))

        tarball_data = Path(tarball_buf.name).read_bytes()
        os.unlink(tarball_buf.name)

        tarball_hash = hashlib.sha256(tarball_data).hexdigest()

        with patch("shutil.which", return_value=None), \
             patch("pyisolate._internal.pixi_provisioner._cache_dir", return_value=cache), \
             patch("pyisolate._internal.pixi_provisioner._fetch_url") as fetch_mock:
            fetch_mock.side_effect = [
                f"{tarball_hash}  pixi.tar.gz".encode(),
                tarball_data,
            ]
            result = ensure_pixi(version)
            assert Path(result).exists()
            assert Path(result).read_bytes() == fake_binary
            print(f"RESOLVED_PATH={result}")


class TestVersionPin:
    def test_version_is_pinned(self):
        """PIXI_VERSION is a hardcoded string, not dynamic."""
        assert isinstance(PIXI_VERSION, str)
        assert len(PIXI_VERSION.split(".")) >= 2
        assert PIXI_VERSION == "0.67.0"
