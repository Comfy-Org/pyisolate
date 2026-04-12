"""Tests for conda/pixi environment creation (environment_conda.py)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from pyisolate._internal.environment_conda import (
    _generate_pixi_toml,
    _install_cuda_wheels_into_pixi,
    _install_local_wheels,
    _parse_dep,
    _pyisolate_source_path,
    _resolve_pixi_python,
    _resolve_uv_exe,
    _toml_path_string,
    create_conda_env,
)


def _make_conda_config(**overrides: object) -> dict:
    """Minimal valid conda config for tests."""
    base: dict = {
        "package_manager": "conda",
        "conda_channels": ["conda-forge"],
        "conda_dependencies": ["numpy"],
        "dependencies": ["requests"],
        "share_torch": False,
        "module": "test_ext",
    }
    base.update(overrides)
    return base


def _pixi_python_path(env_path: Path) -> Path:
    if os.name == "nt":
        return env_path / ".pixi" / "envs" / "default" / "python.exe"
    return env_path / ".pixi" / "envs" / "default" / "bin" / "python"


# ── _generate_pixi_toml ──────────────────────────────────────────────


class TestGeneratePixiToml:
    def test_basic_toml_structure(self) -> None:
        config = _make_conda_config()
        toml_str = _generate_pixi_toml(config)
        assert "[workspace]" in toml_str
        assert "[project]" not in toml_str
        assert "[dependencies]" in toml_str
        assert 'python = "*"' in toml_str
        assert "numpy" in toml_str

    def test_generate_pixi_toml_uses_workspace_header(self) -> None:
        config = _make_conda_config()
        toml_str = _generate_pixi_toml(config)
        assert "[workspace]" in toml_str
        assert "[project]" not in toml_str

    def test_conda_deps_in_dependencies_section(self) -> None:
        config = _make_conda_config(conda_dependencies=["numpy", "scipy>=1.10"])
        toml_str = _generate_pixi_toml(config)
        assert "numpy" in toml_str
        assert "scipy" in toml_str

    def test_pip_deps_in_pypi_dependencies(self) -> None:
        config = _make_conda_config(dependencies=["requests>=2.0", "flask"])
        toml_str = _generate_pixi_toml(config)
        assert "[pypi-dependencies]" in toml_str
        assert "requests" in toml_str
        assert "flask" in toml_str

    def test_generate_pixi_toml_excludes_cuda_wheel_packages_from_pypi_dependencies(
        self,
    ) -> None:
        config = _make_conda_config(
            dependencies=["requests>=2.0", "spconv", "cumm", "flash-attn"],
            cuda_wheels={
                "index_url": "https://example.invalid/cuda-wheels/",
                "packages": ["spconv", "cumm", "flash-attn"],
            },
        )
        toml_str = _generate_pixi_toml(config)
        assert "[pypi-dependencies]" in toml_str
        assert 'requests = ">=2.0"' in toml_str
        assert "spconv =" not in toml_str
        assert "cumm =" not in toml_str
        assert "flash-attn =" not in toml_str

    def test_conda_manifest_installs_local_pyisolate(self) -> None:
        config = _make_conda_config()
        toml_str = _generate_pixi_toml(config)
        assert f'pyisolate = {{ path = "{_toml_path_string(_pyisolate_source_path())}" }}' in toml_str

    def test_windows_pixi_manifest_path_is_toml_safe(self) -> None:
        config = _make_conda_config()
        toml_str = _generate_pixi_toml(config)
        assert f'pyisolate = {{ path = "{_toml_path_string(_pyisolate_source_path())}" }}' in toml_str
        if os.name == "nt":
            assert f'pyisolate = {{ path = "{_pyisolate_source_path()}" }}' not in toml_str

    def test_channels_included(self) -> None:
        config = _make_conda_config(conda_channels=["conda-forge", "nvidia"])
        toml_str = _generate_pixi_toml(config)
        assert "conda-forge" in toml_str
        assert "nvidia" in toml_str

    def test_platforms_included(self) -> None:
        config = _make_conda_config(conda_platforms=["linux-64", "win-64"])
        toml_str = _generate_pixi_toml(config)
        assert "linux-64" in toml_str
        assert "win-64" in toml_str

    def test_no_pip_deps_omits_pypi_section(self) -> None:
        config = _make_conda_config(dependencies=[])
        toml_str = _generate_pixi_toml(config)
        assert "[pypi-dependencies]" in toml_str
        assert f'pyisolate = {{ path = "{_toml_path_string(_pyisolate_source_path())}" }}' in toml_str

    def test_no_conda_deps_omits_dependencies_section(self) -> None:
        config = _make_conda_config(conda_dependencies=[])
        toml_str = _generate_pixi_toml(config)
        assert "[workspace]" in toml_str
        assert "[dependencies]" in toml_str
        assert 'python = "*"' in toml_str

    def test_generate_pixi_toml_preserves_extras(self) -> None:
        config = _make_conda_config(dependencies=["jax[cuda12]>=0.4.30", "numpy>=2.2"])
        toml_str = _generate_pixi_toml(config)
        assert 'jax = { version = ">=0.4.30", extras = ["cuda12"] }' in toml_str
        assert 'numpy = ">=2.2"' in toml_str

    def test_generate_pixi_toml_marker_passthrough(self) -> None:
        config = _make_conda_config(
            dependencies=[
                "jax[cuda12]>=0.4.30; sys_platform == 'linux'",
                "jax>=0.4.30; sys_platform == 'win32'",
            ]
        )
        toml_str = _generate_pixi_toml(config)
        assert "sys_platform == 'linux'" in toml_str
        assert "sys_platform == 'win32'" in toml_str

    def test_generate_pixi_toml_marker_not_in_version(self) -> None:
        config = _make_conda_config(dependencies=["jax[cuda12]>=0.4.30; sys_platform == 'linux'"])
        toml_str = _generate_pixi_toml(config)
        # The marker must NOT appear inside the version field
        assert 'version = ">=0.4.30; sys_platform' not in toml_str
        # It must appear in a separate markers field
        assert 'markers = "sys_platform ==' in toml_str

    def test_generate_pixi_toml_marker_version_clean(self) -> None:
        config = _make_conda_config(dependencies=["jax[cuda12]>=0.4.30; sys_platform == 'linux'"])
        toml_str = _generate_pixi_toml(config)
        assert 'version = ">=0.4.30"' in toml_str

    def test_generate_pixi_toml_uses_version_pin_when_no_pyproject(self, tmp_path: Path) -> None:
        config = _make_conda_config()
        with patch(
            "pyisolate._internal.environment_conda._pyisolate_source_path",
            return_value=tmp_path,
        ):
            toml_str = _generate_pixi_toml(config)
        assert 'pyisolate = "==' in toml_str
        assert "pyisolate = { path =" not in toml_str

    def test_generate_pixi_toml_uses_path_when_pyproject_exists(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'pyisolate'\n")
        config = _make_conda_config()
        with patch(
            "pyisolate._internal.environment_conda._pyisolate_source_path",
            return_value=tmp_path,
        ):
            toml_str = _generate_pixi_toml(config)
        assert "pyisolate = { path =" in toml_str
        assert 'pyisolate = "==' not in toml_str

    def test_generate_pixi_toml_pypi_fallback_produces_parseable_toml(self, tmp_path: Path) -> None:
        import tomllib

        config = _make_conda_config(dependencies=["jax[cuda12]>=0.4.30", "numpy>=2.2"])
        with patch(
            "pyisolate._internal.environment_conda._pyisolate_source_path",
            return_value=tmp_path,
        ):
            toml_str = _generate_pixi_toml(config)
        parsed = tomllib.loads(toml_str)
        pyisolate_dep = parsed["pypi-dependencies"]["pyisolate"]
        assert isinstance(
            pyisolate_dep, str
        ), f"Expected string, got {type(pyisolate_dep)}: {pyisolate_dep}"
        assert pyisolate_dep.startswith(
            "=="
        ), f"Expected version pin starting with '==', got: {pyisolate_dep}"


# ── _parse_dep ──────────────────────────────────────────────────────


class TestParseDep:
    def test_parse_dep_preserves_extras(self) -> None:
        name, sep, ver, extras, marker = _parse_dep("jax[cuda12]>=0.4.30")
        assert name == "jax"
        assert sep == ">="
        assert ver == ">=0.4.30"
        assert extras == ["cuda12"]
        assert marker == ""

    def test_parse_dep_preserves_easy_extra(self) -> None:
        name, sep, ver, extras, marker = _parse_dep("trimesh[easy]>=4.0.0")
        assert name == "trimesh"
        assert sep == ">="
        assert ver == ">=4.0.0"
        assert extras == ["easy"]
        assert marker == ""

    def test_parse_dep_no_extras(self) -> None:
        name, sep, ver, extras, marker = _parse_dep("numpy>=2.0")
        assert name == "numpy"
        assert sep == ">="
        assert ver == ">=2.0"
        assert extras == []
        assert marker == ""

    def test_parse_dep_bare_name(self) -> None:
        name, sep, ver, extras, marker = _parse_dep("requests")
        assert name == "requests"
        assert sep == ""
        assert ver == ""
        assert extras == []
        assert marker == ""

    def test_parse_dep_url(self) -> None:
        name, sep, ver, extras, marker = _parse_dep("pkg @ https://example.com/pkg.whl")
        assert name == "pkg"
        assert sep == "@"
        assert ver == "https://example.com/pkg.whl"
        assert extras == []
        assert marker == ""

    def test_parse_dep_marker_extras(self) -> None:
        name, sep, ver, extras, marker = _parse_dep("jax[cuda12]>=0.4.30; sys_platform == 'linux'")
        assert name == "jax"
        assert sep == ">="
        assert ver == ">=0.4.30"
        assert extras == ["cuda12"]
        assert marker == "sys_platform == 'linux'"

    def test_parse_dep_marker_version_only(self) -> None:
        name, sep, ver, extras, marker = _parse_dep("numpy>=2.0; platform_system != 'Windows'")
        assert name == "numpy"
        assert sep == ">="
        assert ver == ">=2.0"
        assert extras == []
        assert marker == "platform_system != 'Windows'"

    def test_parse_dep_marker_url(self) -> None:
        name, sep, ver, extras, marker = _parse_dep(
            "pkg @ https://example.com/pkg.whl ; python_version >= '3.12'"
        )
        assert name == "pkg"
        assert sep == "@"
        assert ver == "https://example.com/pkg.whl"
        assert extras == []
        assert marker == "python_version >= '3.12'"


# ── create_conda_env ─────────────────────────────────────────────────


class TestCreateCondaEnv:
    def test_pixi_not_found_raises(self, tmp_path: Path) -> None:
        config = _make_conda_config()
        with patch("shutil.which", return_value=None), pytest.raises(RuntimeError, match="pixi.*not found"):
            create_conda_env(tmp_path / "env", config, "test_ext")

    def test_pixi_install_called(self, tmp_path: Path) -> None:
        env_path = tmp_path / "env"
        config = _make_conda_config()
        pixi_python = _pixi_python_path(env_path)

        with (
            patch("shutil.which", return_value="/usr/bin/pixi"),
            patch("subprocess.check_call") as mock_call,
            patch.object(Path, "exists", return_value=True),
        ):
            # Make the pixi python appear to exist
            pixi_python.parent.mkdir(parents=True, exist_ok=True)
            pixi_python.touch()
            create_conda_env(env_path, config, "test_ext")

        # pixi install should have been called
        assert mock_call.called
        call_args = mock_call.call_args[0][0]
        assert "pixi" in call_args[0]
        assert "install" in call_args

    def test_pixi_install_failure_raises(self, tmp_path: Path) -> None:
        import subprocess

        config = _make_conda_config()
        with (
            patch("shutil.which", return_value="/usr/bin/pixi"),
            patch(
                "subprocess.check_call",
                side_effect=subprocess.CalledProcessError(1, "pixi"),
            ),
            pytest.raises(subprocess.CalledProcessError),
        ):
            create_conda_env(tmp_path / "env", config, "test_ext")

    def test_create_conda_env_installs_cuda_wheels_post_pixi(self, tmp_path: Path) -> None:
        env_path = tmp_path / "env"
        config = _make_conda_config(
            dependencies=["requests>=2.0", "spconv", "cumm"],
            cuda_wheels={
                "index_url": "https://example.invalid/cuda-wheels/",
                "packages": ["spconv", "cumm"],
            },
        )
        pixi_python = _pixi_python_path(env_path)

        with (
            patch("shutil.which", return_value="/usr/bin/pixi"),
            patch("subprocess.check_call") as mock_check_call,
            patch(
                "pyisolate._internal.environment_conda._install_cuda_wheels_into_pixi"
            ) as mock_install_cuda_wheels,
        ):
            pixi_python.parent.mkdir(parents=True, exist_ok=True)
            pixi_python.touch()
            create_conda_env(env_path, config, "test_ext")

        assert mock_check_call.called
        mock_install_cuda_wheels.assert_called_once_with(
            pixi_python,
            config,
            config["cuda_wheels"],
            "test_ext",
        )

    def test_writes_pixi_toml(self, tmp_path: Path) -> None:
        env_path = tmp_path / "env"
        config = _make_conda_config()
        pixi_python = _pixi_python_path(env_path)

        with (
            patch("shutil.which", return_value="/usr/bin/pixi"),
            patch("subprocess.check_call"),
        ):
            pixi_python.parent.mkdir(parents=True, exist_ok=True)
            pixi_python.touch()
            create_conda_env(env_path, config, "test_ext")

        toml_path = env_path / "pixi.toml"
        assert toml_path.exists()
        content = toml_path.read_text()
        assert "[workspace]" in content

    def test_sanitizes_invalid_tmpdir_for_pixi_install(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_path = tmp_path / "env"
        config = _make_conda_config()
        stale_tmpdir = tmp_path / "deleted" / "ipc_shared"
        monkeypatch.setenv("TMPDIR", str(stale_tmpdir))

        with (
            patch("shutil.which", return_value="/usr/bin/pixi"),
            patch("subprocess.check_call") as mock_call,
            patch(
                "pyisolate._internal.environment_conda._resolve_pixi_python",
                return_value=_pixi_python_path(env_path),
            ),
        ):
            create_conda_env(env_path, config, "test_ext")

        call_kwargs = mock_call.call_args.kwargs
        passed_env = call_kwargs["env"]
        assert passed_env["TMPDIR"] != str(stale_tmpdir)
        assert Path(passed_env["TMPDIR"]).exists()

    def test_fingerprint_skip(self, tmp_path: Path) -> None:
        """If fingerprint matches, pixi install should be skipped."""
        env_path = tmp_path / "env"
        env_path.mkdir(parents=True)
        config = _make_conda_config()

        # Pre-create a matching fingerprint
        import hashlib

        toml_content = _generate_pixi_toml(config)
        descriptor = {
            "conda_dependencies": config.get("conda_dependencies", []),
            "pip_dependencies": config.get("dependencies", []),
            "channels": config.get("conda_channels", []),
            "platforms": config.get("conda_platforms", []),
            "cuda_wheels": config.get("cuda_wheels"),
            "find_links": config.get("find_links", []),
            "pixi_toml": toml_content,
        }
        fingerprint = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
        lock_path = env_path / ".pyisolate_deps.json"
        lock_path.write_text(json.dumps({"fingerprint": fingerprint, "descriptor": descriptor}))

        pixi_python = _pixi_python_path(env_path)
        pixi_python.parent.mkdir(parents=True, exist_ok=True)
        pixi_python.touch()

        with (
            patch("shutil.which", return_value="/usr/bin/pixi"),
            patch("subprocess.check_call") as mock_call,
        ):
            create_conda_env(env_path, config, "test_ext")

        # pixi install should NOT have been called
        assert not mock_call.called


# ── _resolve_pixi_python ─────────────────────────────────────────────


class TestResolvePixiPython:
    def test_returns_pixi_env_python(self, tmp_path: Path) -> None:
        env_path = tmp_path / "env"
        expected = _pixi_python_path(env_path)
        expected.parent.mkdir(parents=True, exist_ok=True)
        expected.touch()
        result = _resolve_pixi_python(env_path)
        assert result == expected

    def test_missing_python_raises(self, tmp_path: Path) -> None:
        env_path = tmp_path / "env"
        env_path.mkdir(parents=True)
        with pytest.raises(RuntimeError, match="Python.*not found"):
            _resolve_pixi_python(env_path)

    def test_never_returns_host_python(self, tmp_path: Path) -> None:
        env_path = tmp_path / "env"
        expected = _pixi_python_path(env_path)
        expected.parent.mkdir(parents=True, exist_ok=True)
        expected.touch()
        result = _resolve_pixi_python(env_path)
        assert str(result) != sys.executable
        assert ".pixi" in str(result)


# ── _install_cuda_wheels_into_pixi target_python threading ─────────────


def test_install_cuda_wheels_passes_target_python(monkeypatch, tmp_path):
    """AC-1: conda_python='3.12.*' is parsed and passed as target_python=(3, 12)."""
    captured_kwargs: list[dict] = []

    def mock_resolve(deps, config, **kwargs):
        captured_kwargs.append(kwargs)
        return deps  # return unchanged (no wheel resolution)

    monkeypatch.setattr(
        "pyisolate._internal.cuda_wheels.resolve_cuda_wheel_requirements",
        mock_resolve,
    )

    python_exe = tmp_path / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.touch()

    config = _make_conda_config(
        conda_python="3.12.*",
        dependencies=["flash-attn"],
        cuda_wheels={
            "index_url": "https://example.invalid/",
            "packages": ["flash-attn"],
        },
    )

    _install_cuda_wheels_into_pixi(python_exe, config, config["cuda_wheels"], "test")

    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["target_python"] == (3, 12)


def test_install_cuda_wheels_wildcard_python_uses_host_tags(monkeypatch, tmp_path):
    """AC-2: conda_python='*' passes target_python=None (host tags fallback)."""
    captured_kwargs: list[dict] = []

    def mock_resolve(deps, config, **kwargs):
        captured_kwargs.append(kwargs)
        return deps

    monkeypatch.setattr(
        "pyisolate._internal.cuda_wheels.resolve_cuda_wheel_requirements",
        mock_resolve,
    )

    python_exe = tmp_path / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.touch()

    config = _make_conda_config(
        conda_python="*",
        dependencies=["flash-attn"],
        cuda_wheels={
            "index_url": "https://example.invalid/",
            "packages": ["flash-attn"],
        },
    )

    _install_cuda_wheels_into_pixi(python_exe, config, config["cuda_wheels"], "test")

    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["target_python"] is None


def test_install_cuda_wheels_parses_311(monkeypatch, tmp_path):
    """AC-3: conda_python='3.11.*' parses to target_python=(3, 11)."""
    captured_kwargs: list[dict] = []

    def mock_resolve(deps, config, **kwargs):
        captured_kwargs.append(kwargs)
        return deps

    monkeypatch.setattr(
        "pyisolate._internal.cuda_wheels.resolve_cuda_wheel_requirements",
        mock_resolve,
    )

    python_exe = tmp_path / "bin" / "python"
    python_exe.parent.mkdir(parents=True)
    python_exe.touch()

    config = _make_conda_config(
        conda_python="3.11.*",
        dependencies=["flash-attn"],
        cuda_wheels={
            "index_url": "https://example.invalid/",
            "packages": ["flash-attn"],
        },
    )

    _install_cuda_wheels_into_pixi(python_exe, config, config["cuda_wheels"], "test")

    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["target_python"] == (3, 11)


# ── _resolve_uv_exe / uv path fallback ─────────────────────────────


class TestResolveUvExe:
    def test_install_cuda_wheels_uv_exe_fallback(self, monkeypatch, tmp_path):
        """When python_exe.parent/uv does not exist, falls back to shutil.which."""
        # Create python_exe in a dir without uv
        python_exe = tmp_path / "no_uv_here" / "python"
        python_exe.parent.mkdir(parents=True)
        python_exe.touch()

        resolved = _resolve_uv_exe(python_exe)
        # Should have fallen back to shutil.which since no local uv exists
        import shutil

        system_uv = shutil.which("uv")
        assert resolved == system_uv

    def test_install_local_wheels_uv_exe_fallback(self, monkeypatch, tmp_path):
        """_install_local_wheels uses _resolve_uv_exe and falls back correctly."""
        python_exe = tmp_path / "no_uv_here" / "python"
        python_exe.parent.mkdir(parents=True)
        python_exe.touch()

        # Create a fake wheel file
        wheel_dir = tmp_path / "wheels"
        wheel_dir.mkdir()
        (wheel_dir / "fake-1.0-py3-none-any.whl").touch()

        captured_cmds: list[list[str]] = []

        def mock_check_call(cmd, **kwargs):
            captured_cmds.append(cmd)

        monkeypatch.setattr("subprocess.check_call", mock_check_call)

        config = _make_conda_config(module_path=str(tmp_path))
        _install_local_wheels(python_exe, config, [str(wheel_dir)], "test")

        assert len(captured_cmds) == 1
        import shutil

        system_uv = shutil.which("uv")
        assert captured_cmds[0][0] == system_uv

    def test_install_cuda_wheels_uv_exe_prefers_local(self, tmp_path):
        """When python_exe.parent/uv exists, it is preferred over shutil.which."""
        python_exe = tmp_path / "bin" / "python"
        python_exe.parent.mkdir(parents=True)
        python_exe.touch()
        local_uv = tmp_path / "bin" / "uv"
        local_uv.touch()

        resolved = _resolve_uv_exe(python_exe)
        assert resolved == str(local_uv)

    def test_install_cuda_wheels_uv_exe_windows_layout(self, tmp_path):
        """Windows pixi layout: python at envs/default/python.exe, no bin/ dir."""
        # Simulate Windows pixi path structure
        pixi_default = tmp_path / ".pixi" / "envs" / "default"
        pixi_default.mkdir(parents=True)
        python_exe = pixi_default / "python.exe"
        python_exe.touch()
        # No uv in python_exe.parent (Windows layout)

        resolved = _resolve_uv_exe(python_exe)
        import shutil

        system_uv = shutil.which("uv")
        assert resolved == system_uv
