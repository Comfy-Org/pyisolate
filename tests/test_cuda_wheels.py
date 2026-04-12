"""Synthetic/unit coverage for CUDA wheel resolution.

These tests intentionally use monkeypatches and fake indexes. They do not
perform a real wheel download or a real install.
"""

import builtins
import io
import sys
from types import SimpleNamespace

import pytest
from packaging.tags import sys_tags

from pyisolate._internal import environment
from pyisolate._internal.cuda_wheels import (
    CUDAWheelResolutionError,
    _normalize_cuda_wheel_config,
    get_cuda_wheel_runtime,
    resolve_cuda_wheel_requirements,
    resolve_cuda_wheel_url,
)


def _runtime() -> dict[str, object]:
    return {
        "torch": "2.8",
        "torch_nodot": "28",
        "cuda": "12.8",
        "cuda_nodot": "128",
        "python_tags": [str(tag) for tag in sys_tags()],
    }


def _wheel_filename(distribution: str, version: str) -> str:
    tag = next(iter(sys_tags()))
    return f"{distribution}-{version}-{tag.interpreter}-{tag.abi}-{tag.platform}.whl"


def _simple_index_html(*filenames: str) -> str:
    links = [f'<a href="{filename}">{filename}</a>' for filename in filenames]
    return "<html><body>" + "".join(links) + "</body></html>"


def test_resolve_cuda_wheel_requirement_to_direct_url(monkeypatch):
    runtime = _runtime()
    wheel = _wheel_filename("flash_attn", "1.1.0+cu128torch28")
    page_url = "https://example.invalid/cuda-wheels/flash-attn/"

    monkeypatch.setattr("pyisolate._internal.cuda_wheels.get_cuda_wheel_runtime", lambda **kw: runtime)
    monkeypatch.setattr(
        "pyisolate._internal.cuda_wheels._fetch_index_html",
        lambda url: _simple_index_html(wheel) if url == page_url else None,
    )

    resolved = resolve_cuda_wheel_requirements(
        ["flash-attn>=1.0"],
        {
            "index_url": "https://example.invalid/cuda-wheels/",
            "packages": ["flash-attn"],
            "package_map": {},
        },
    )

    assert resolved == [page_url + wheel]


def test_resolve_cuda_wheel_requirement_supports_underscore_index(monkeypatch):
    runtime = _runtime()
    wheel = _wheel_filename("torch_generic_nms", "0.2.0+cu128torch28")
    page_url = "https://example.invalid/cuda-wheels/torch_generic_nms/"

    monkeypatch.setattr("pyisolate._internal.cuda_wheels.get_cuda_wheel_runtime", lambda **kw: runtime)
    monkeypatch.setattr(
        "pyisolate._internal.cuda_wheels._fetch_index_html",
        lambda url: _simple_index_html(wheel) if url == page_url else None,
    )

    resolved = resolve_cuda_wheel_requirements(
        ["torch-generic-nms>=0.1"],
        {
            "index_url": "https://example.invalid/cuda-wheels/",
            "packages": ["torch-generic-nms"],
            "package_map": {},
        },
    )

    assert resolved == [page_url + wheel]


def test_resolve_cuda_wheel_requirement_supports_percent_encoded_links(monkeypatch):
    runtime = _runtime()
    wheel = _wheel_filename("torch_generic_nms", "0.1+cu128torch28")
    encoded_wheel = wheel.replace("+", "%2B")
    page_url = "https://example.invalid/cuda-wheels/torch-generic-nms/"

    monkeypatch.setattr("pyisolate._internal.cuda_wheels.get_cuda_wheel_runtime", lambda **kw: runtime)
    monkeypatch.setattr(
        "pyisolate._internal.cuda_wheels._fetch_index_html",
        lambda url: _simple_index_html(encoded_wheel) if url == page_url else None,
    )

    resolved = resolve_cuda_wheel_requirements(
        ["torch-generic-nms==0.1"],
        {
            "index_url": "https://example.invalid/cuda-wheels/",
            "packages": ["torch-generic-nms"],
            "package_map": {},
        },
    )

    assert resolved == [page_url + wheel]


def test_resolve_cuda_wheel_requirement_honors_package_map(monkeypatch):
    runtime = _runtime()
    wheel = _wheel_filename("flash_attn", "1.2.0+cu128torch28")
    page_url = "https://example.invalid/cuda-wheels/flash_attn_special/"

    monkeypatch.setattr("pyisolate._internal.cuda_wheels.get_cuda_wheel_runtime", lambda **kw: runtime)
    monkeypatch.setattr(
        "pyisolate._internal.cuda_wheels._fetch_index_html",
        lambda url: _simple_index_html(wheel) if url == page_url else None,
    )

    resolved = resolve_cuda_wheel_requirements(
        ["flash-attn>=1.0"],
        {
            "index_url": "https://example.invalid/cuda-wheels/",
            "packages": ["flash-attn"],
            "package_map": {"flash-attn": "flash_attn_special"},
        },
    )

    assert resolved == [page_url + wheel]


def test_resolve_cuda_wheel_requirement_picks_highest_matching_version(monkeypatch):
    runtime = _runtime()
    compatible_old = _wheel_filename("flash_attn", "1.1.0+cu128torch28")
    compatible_new = _wheel_filename("flash_attn", "1.3.0+pt28cu128")
    incompatible_cuda = _wheel_filename("flash_attn", "1.4.0+cu127torch28")
    out_of_range = _wheel_filename("flash_attn", "2.0.0+cu128torch28")
    page_url = "https://example.invalid/cuda-wheels/flash-attn/"

    monkeypatch.setattr("pyisolate._internal.cuda_wheels.get_cuda_wheel_runtime", lambda **kw: runtime)
    monkeypatch.setattr(
        "pyisolate._internal.cuda_wheels._fetch_index_html",
        lambda url: (
            _simple_index_html(
                compatible_old,
                compatible_new,
                incompatible_cuda,
                out_of_range,
            )
            if url == page_url
            else None
        ),
    )

    resolved = resolve_cuda_wheel_requirements(
        ["flash-attn>=1.0,<2.0"],
        {
            "index_url": "https://example.invalid/cuda-wheels/",
            "packages": ["flash-attn"],
            "package_map": {},
        },
    )

    assert resolved == [page_url + compatible_new]


def test_resolve_cuda_wheel_requirement_prefers_better_supported_tag(monkeypatch):
    all_tags = list(sys_tags())
    manylinux_tags = [t for t in all_tags if "manylinux" in t.platform and "x86_64" in t.platform]
    linux_tags = [t for t in all_tags if t.platform == "linux_x86_64"]
    if not manylinux_tags or not linux_tags:
        pytest.skip("manylinux/linux_x86_64 tags not available on this platform")
    ml_tag = manylinux_tags[0]
    lx_tag = linux_tags[0]
    runtime = _runtime()
    hyphen_page = "https://example.invalid/cuda-wheels/torch-generic-nms/"
    underscore_page = "https://example.invalid/cuda-wheels/torch_generic_nms/"
    preferred = f"torch_generic_nms-0.1+cu128torch28-{ml_tag.interpreter}-{ml_tag.abi}-{ml_tag.platform}.whl"
    fallback = f"torch_generic_nms-0.1+cu128torch28-{lx_tag.interpreter}-{lx_tag.abi}-{lx_tag.platform}.whl"

    monkeypatch.setattr("pyisolate._internal.cuda_wheels.get_cuda_wheel_runtime", lambda **kw: runtime)
    monkeypatch.setattr(
        "pyisolate._internal.cuda_wheels._fetch_index_html",
        lambda url: (
            _simple_index_html(preferred)
            if url == hyphen_page
            else _simple_index_html(fallback)
            if url == underscore_page
            else None
        ),
    )

    resolved = resolve_cuda_wheel_requirements(
        ["torch-generic-nms==0.1"],
        {
            "index_url": "https://example.invalid/cuda-wheels/",
            "packages": ["torch-generic-nms"],
            "package_map": {},
        },
    )

    assert resolved == [hyphen_page + preferred]


def test_resolve_cuda_wheel_requirement_raises_when_no_match(monkeypatch):
    runtime = _runtime()
    wheel = _wheel_filename("flash_attn", "1.1.0+cu127torch28")
    page_url = "https://example.invalid/cuda-wheels/flash-attn/"

    monkeypatch.setattr("pyisolate._internal.cuda_wheels.get_cuda_wheel_runtime", lambda **kw: runtime)
    monkeypatch.setattr(
        "pyisolate._internal.cuda_wheels._fetch_index_html",
        lambda url: _simple_index_html(wheel) if url == page_url else None,
    )

    with pytest.raises(CUDAWheelResolutionError, match="No compatible CUDA wheel found"):
        resolve_cuda_wheel_requirements(
            ["flash-attn>=1.0"],
            {
                "index_url": "https://example.invalid/cuda-wheels/",
                "packages": ["flash-attn"],
                "package_map": {},
            },
        )


def test_get_cuda_wheel_runtime_raises_without_torch(monkeypatch):
    real_import = builtins.__import__

    def missing_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("missing torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_torch)

    with pytest.raises(CUDAWheelResolutionError, match="host torch"):
        get_cuda_wheel_runtime()


def test_get_cuda_wheel_runtime_raises_without_cuda(monkeypatch):
    fake_torch = SimpleNamespace(
        __version__="2.8.1",
        version=SimpleNamespace(cuda=None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(CUDAWheelResolutionError, match="CUDA-enabled host torch"):
        get_cuda_wheel_runtime()


def test_install_dependencies_cache_invalidation_tracks_cuda_runtime(monkeypatch, tmp_path):
    import os

    venv_path = tmp_path / "venv"
    python_exe = venv_path / "Scripts" / "python.exe" if os.name == "nt" else venv_path / "bin" / "python"
    python_exe.parent.mkdir(parents=True, exist_ok=True)
    python_exe.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    monkeypatch.setattr(environment.shutil, "which", lambda binary: "/usr/bin/uv")
    monkeypatch.setattr(
        environment,
        "exclude_satisfied_requirements",
        lambda config, requirements, python_exe: requirements,
    )
    monkeypatch.setattr(
        environment,
        "resolve_cuda_wheel_requirements",
        lambda requirements, config: ["https://example.invalid/flash_attn.whl"],
    )

    current_runtime = {"value": {"torch": "2.8", "cuda": "12.8", "python_tags": ["cp312"]}}
    monkeypatch.setattr(
        environment,
        "get_cuda_wheel_runtime_descriptor",
        lambda: current_runtime["value"],
    )

    popen_calls: list[list[str]] = []

    class MockPopen:
        def __init__(self, cmd, **kwargs):
            popen_calls.append(cmd)
            self.stdout = io.StringIO("installed\n")

        def wait(self):
            return 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(environment.subprocess, "Popen", MockPopen)

    config = {
        "dependencies": ["flash-attn>=1.0"],
        "share_torch": True,
        "cuda_wheels": {
            "index_url": "https://example.invalid/cuda-wheels/",
            "packages": ["flash-attn"],
            "package_map": {},
        },
    }

    environment.install_dependencies(venv_path, config, "demo")
    environment.install_dependencies(venv_path, config, "demo")

    current_runtime["value"] = {"torch": "2.8", "cuda": "12.9", "python_tags": ["cp312"]}
    environment.install_dependencies(venv_path, config, "demo")

    assert len(popen_calls) == 2


# ── target_python parameter tests ─────────────────────────────────────


def _wheel_filename_for_cpython(distribution: str, version: str, cpython: str) -> str:
    """Build a wheel filename with a specific cpython tag (e.g. 'cp312')."""
    tag = next(iter(sys_tags()))
    return f"{distribution}-{version}-{cpython}-{cpython}-{tag.platform}.whl"


def test_resolve_cuda_wheel_url_accepts_target_python_parameter(monkeypatch):
    """AC-1: resolve_cuda_wheel_url accepts target_python=(3, 12) without TypeError."""
    runtime = _runtime()
    wheel_312 = _wheel_filename_for_cpython("flash_attn", "1.0.0+cu128torch28", "cp312")
    page_url = "https://example.invalid/cuda-wheels/flash-attn/"

    monkeypatch.setattr("pyisolate._internal.cuda_wheels.get_cuda_wheel_runtime", lambda **kw: runtime)
    monkeypatch.setattr(
        "pyisolate._internal.cuda_wheels._fetch_index_html",
        lambda url: _simple_index_html(wheel_312) if url == page_url else None,
    )

    from packaging.requirements import Requirement

    url = resolve_cuda_wheel_url(
        Requirement("flash-attn"),
        {
            "index_url": "https://example.invalid/cuda-wheels/",
            "packages": ["flash-attn"],
            "package_map": {},
        },
        runtime,
        target_python=(3, 12),
    )
    assert url.endswith(".whl")
    assert "cp312" in url


def test_resolve_cuda_wheel_uses_target_python_tags(monkeypatch):
    """AC-2: target_python=(3, 12) selects cp312 wheel, not cp313."""
    runtime = _runtime()
    wheel_312 = _wheel_filename_for_cpython("flash_attn", "1.0.0+cu128torch28", "cp312")
    wheel_313 = _wheel_filename_for_cpython("flash_attn", "1.0.0+cu128torch28", "cp313")
    page_url = "https://example.invalid/cuda-wheels/flash-attn/"

    monkeypatch.setattr("pyisolate._internal.cuda_wheels.get_cuda_wheel_runtime", lambda **kw: runtime)
    monkeypatch.setattr(
        "pyisolate._internal.cuda_wheels._fetch_index_html",
        lambda url: _simple_index_html(wheel_312, wheel_313) if url == page_url else None,
    )

    from packaging.requirements import Requirement

    url = resolve_cuda_wheel_url(
        Requirement("flash-attn"),
        {
            "index_url": "https://example.invalid/cuda-wheels/",
            "packages": ["flash-attn"],
            "package_map": {},
        },
        runtime,
        target_python=(3, 12),
    )
    assert "cp312" in url
    assert "cp313" not in url


def test_resolve_cuda_wheel_requirements_threads_target_python(monkeypatch):
    """AC-3: resolve_cuda_wheel_requirements threads target_python to resolve_cuda_wheel_url."""
    runtime = _runtime()
    wheel_312 = _wheel_filename_for_cpython("flash_attn", "1.0.0+cu128torch28", "cp312")
    wheel_313 = _wheel_filename_for_cpython("flash_attn", "1.0.0+cu128torch28", "cp313")
    page_url = "https://example.invalid/cuda-wheels/flash-attn/"

    monkeypatch.setattr("pyisolate._internal.cuda_wheels.get_cuda_wheel_runtime", lambda **kw: runtime)
    monkeypatch.setattr(
        "pyisolate._internal.cuda_wheels._fetch_index_html",
        lambda url: _simple_index_html(wheel_312, wheel_313) if url == page_url else None,
    )

    resolved = resolve_cuda_wheel_requirements(
        ["flash-attn>=1.0"],
        {
            "index_url": "https://example.invalid/cuda-wheels/",
            "packages": ["flash-attn"],
            "package_map": {},
        },
        target_python=(3, 12),
    )
    assert len(resolved) == 1
    assert "cp312" in resolved[0]
    assert "cp313" not in resolved[0]


def test_resolve_cuda_wheel_target_python_rejects_host_only_wheel(monkeypatch):
    """AC-4: target_python=(3, 12) raises when only cp313 wheel available."""
    runtime = _runtime()
    wheel_313_only = _wheel_filename_for_cpython("flash_attn", "1.0.0+cu128torch28", "cp313")
    page_url = "https://example.invalid/cuda-wheels/flash-attn/"

    monkeypatch.setattr("pyisolate._internal.cuda_wheels.get_cuda_wheel_runtime", lambda **kw: runtime)
    monkeypatch.setattr(
        "pyisolate._internal.cuda_wheels._fetch_index_html",
        lambda url: _simple_index_html(wheel_313_only) if url == page_url else None,
    )

    from packaging.requirements import Requirement

    with pytest.raises(CUDAWheelResolutionError, match="No compatible CUDA wheel found"):
        resolve_cuda_wheel_url(
            Requirement("flash-attn"),
            {
                "index_url": "https://example.invalid/cuda-wheels/",
                "packages": ["flash-attn"],
                "package_map": {},
            },
            runtime,
            target_python=(3, 12),
        )


# ── index_urls (plural) support tests ─────────────────────────────────


def test_normalize_config_index_urls():
    """AC-1: _normalize_cuda_wheel_config accepts index_urls (plural list)."""
    config = {
        "index_urls": [
            "https://download.pytorch.org/whl/cu128",
            "https://pozzettiandrea.github.io/cuda-wheels/",
        ],
        "packages": ["cumesh", "torch"],
        "package_map": {},
    }
    result = _normalize_cuda_wheel_config(config)
    assert "index_urls" in result
    assert isinstance(result["index_urls"], list)
    assert len(result["index_urls"]) == 2
    assert result["index_urls"][0] == "https://download.pytorch.org/whl/cu128/"
    assert result["index_urls"][1] == "https://pozzettiandrea.github.io/cuda-wheels/"


def test_normalize_config_singular_returns_index_urls_key():
    """AC-2: _normalize_cuda_wheel_config with index_url (singular) returns index_urls key (plural)."""
    config = {
        "index_url": "https://example.invalid/cuda-wheels",
        "packages": ["flash-attn"],
        "package_map": {},
    }
    result = _normalize_cuda_wheel_config(config)
    assert "index_urls" in result, "Expected 'index_urls' key in normalized config"
    assert "index_url" not in result, "Expected NO 'index_url' key in normalized config"
    assert result["index_urls"] == ["https://example.invalid/cuda-wheels/"]


def test_resolve_iterates_multiple_indexes(monkeypatch):
    """AC-3: resolve_cuda_wheel_url iterates multiple index URLs to find a package."""
    runtime = _runtime()
    wheel = _wheel_filename("cumesh", "0.0.1+cu128torch28")
    # cumesh only exists on the second index URL
    second_index_page = "https://pozzettiandrea.github.io/cuda-wheels/cumesh/"

    monkeypatch.setattr("pyisolate._internal.cuda_wheels.get_cuda_wheel_runtime", lambda **kw: runtime)
    monkeypatch.setattr(
        "pyisolate._internal.cuda_wheels._fetch_index_html",
        lambda url: _simple_index_html(wheel) if url == second_index_page else None,
    )

    from packaging.requirements import Requirement

    url = resolve_cuda_wheel_url(
        Requirement("cumesh"),
        {
            "index_urls": [
                "https://download.pytorch.org/whl/cu128/",
                "https://pozzettiandrea.github.io/cuda-wheels/",
            ],
            "packages": ["cumesh"],
            "package_map": {},
        },
        runtime,
    )
    assert url.endswith(".whl")
    assert "cumesh" in url
    assert "pozzettiandrea" in url


# ── Live network tests ────────────────────────────────────────────────


@pytest.mark.network
def test_resolve_sageattention_for_target_python_312():
    """AC-1/AC-2: Live index resolves cp312 wheel with correct torch+CUDA pattern."""
    from packaging.requirements import Requirement

    from pyisolate._internal.cuda_wheels import get_cuda_wheel_runtime

    runtime = get_cuda_wheel_runtime(target_python=(3, 12))
    url = resolve_cuda_wheel_url(
        Requirement("sageattention"),
        {
            "index_url": "https://pozzettiandrea.github.io/cuda-wheels/",
            "packages": ["sageattention"],
            "package_map": {},
        },
        runtime,
        target_python=(3, 12),
    )
    assert "cp312" in url, f"Expected cp312 in URL: {url}"
    assert "cp313" not in url, f"Unexpected cp313 in URL: {url}"
    # AC-2: torch+CUDA pattern preserved (URL may use dotted "torch2.9" or nodot "torch29")
    assert runtime["cuda_nodot"] in url, f"Expected cuda {runtime['cuda_nodot']} in URL: {url}"
    assert f"torch{runtime['torch']}" in url or f"torch{runtime['torch_nodot']}" in url, (
        f"Expected torch {runtime['torch']} or {runtime['torch_nodot']} in URL: {url}"
    )


@pytest.mark.network
def test_resolve_sageattention_host_tags_selects_cp313():
    """AC-3: Without target_python selects host cpXXX; with target_python=(3, 11) selects cp311."""
    from packaging.requirements import Requirement

    from pyisolate._internal.cuda_wheels import get_cuda_wheel_runtime

    # Host tags (should select host interpreter's cp tag — cp312 on this venv)
    runtime_host = get_cuda_wheel_runtime()
    config = {
        "index_url": "https://pozzettiandrea.github.io/cuda-wheels/",
        "packages": ["sageattention"],
        "package_map": {},
    }
    url_host = resolve_cuda_wheel_url(Requirement("sageattention"), config, runtime_host)
    # The host cpXXX tag should be present
    import sys

    host_cp = f"cp{sys.version_info.major}{sys.version_info.minor}"
    assert host_cp in url_host, f"Expected {host_cp} in host URL: {url_host}"

    # Target tags (3, 11) — should select cp311, different from host
    runtime_target = get_cuda_wheel_runtime(target_python=(3, 11))
    url_target = resolve_cuda_wheel_url(
        Requirement("sageattention"), config, runtime_target, target_python=(3, 11)
    )
    assert "cp311" in url_target, f"Expected cp311 in target URL: {url_target}"
    assert host_cp not in url_target or host_cp == "cp311", (
        f"Unexpected {host_cp} in target URL: {url_target}"
    )


def test_extra_index_urls_plumbed_to_install_command(tmp_path, monkeypatch):
    """extra_index_urls in ExtensionConfig become --extra-index-url args in uv pip install."""
    from pyisolate._internal.environment import install_dependencies

    venv_path = tmp_path / "venvs" / "test-ext"

    config = {
        "name": "test-ext",
        "module_path": str(tmp_path),
        "isolated": True,
        "dependencies": ["numpy>=1.0"],
        "share_torch": True,
        "share_cuda_ipc": False,
        "sandbox_mode": "disabled",
        "sandbox": {},
        "extra_index_urls": ["https://example.invalid/simple"],
    }

    captured_cmd: list[str] = []

    class FakeProc:
        def __init__(self, cmd, **kwargs):
            captured_cmd.extend(cmd)
            self.stdout = iter([])
            self.returncode = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait(self):
            return 0

    monkeypatch.setattr("subprocess.Popen", FakeProc)
    monkeypatch.setattr("subprocess.check_call", lambda cmd, **kw: None)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(
        "pyisolate._internal.environment.exclude_satisfied_requirements",
        lambda config, reqs, python_exe: reqs,
    )

    # Set up fake venv structure (skip create_venv — we only test install_dependencies)
    bin_dir = venv_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake_python = bin_dir / "python"
    fake_python.write_text("#!/bin/sh\n")
    fake_python.chmod(0o755)

    install_dependencies(venv_path, config, "test-ext")

    assert "--extra-index-url" in captured_cmd
    idx = captured_cmd.index("--extra-index-url")
    assert captured_cmd[idx + 1] == "https://example.invalid/simple"


def test_share_torch_cuda_wheels_install_uses_no_deps_for_resolved_urls(tmp_path, monkeypatch):
    """Resolved CUDA wheel URLs must install in a separate --no-deps step under share_torch."""
    from pyisolate._internal.environment import install_dependencies

    venv_path = tmp_path / "venvs" / "test-ext"
    bin_dir = venv_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake_python = bin_dir / "python"
    fake_python.write_text("#!/bin/sh\n")
    fake_python.chmod(0o755)

    config = {
        "name": "test-ext",
        "module_path": str(tmp_path),
        "isolated": True,
        "dependencies": ["cc-torch", "torch-generic-nms", "timm"],
        "share_torch": True,
        "share_cuda_ipc": False,
        "sandbox_mode": "disabled",
        "sandbox": {},
        "cuda_wheels": {
            "index_url": "https://example.invalid/cuda-wheels/",
            "packages": ["cc-torch", "torch-generic-nms"],
            "package_map": {},
        },
    }

    popen_calls: list[list[str]] = []

    class FakeProc:
        def __init__(self, cmd, **kwargs):
            popen_calls.append(cmd)
            self.stdout = iter([])
            self.returncode = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self):
            return 0

    monkeypatch.setattr("subprocess.Popen", FakeProc)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(
        "pyisolate._internal.environment.exclude_satisfied_requirements",
        lambda config, reqs, python_exe: reqs,
    )
    monkeypatch.setattr(
        "pyisolate._internal.environment.get_cuda_wheel_runtime_descriptor",
        lambda: {"torch": "2.11", "cuda": "13.0", "python_tags": ["cp313"]},
    )
    monkeypatch.setattr(
        "pyisolate._internal.environment.resolve_cuda_wheel_requirements",
        lambda requirements, config: [
            "https://github.com/example/cuda-wheels/releases/download/cc_torch-latest/cc_torch-0.2-py3-none-any.whl",
            "https://github.com/example/cuda-wheels/releases/download/torch_generic_nms-latest/torch_generic_nms-0.1-py3-none-any.whl",
            "timm",
        ],
    )

    install_dependencies(venv_path, config, "test-ext")

    assert len(popen_calls) == 2
    assert "--no-deps" not in popen_calls[0]
    assert "timm" in popen_calls[0]
    assert "--no-deps" in popen_calls[1]
    assert any("cc_torch-0.2-py3-none-any.whl" in token for token in popen_calls[1])
    assert any("torch_generic_nms-0.1-py3-none-any.whl" in token for token in popen_calls[1])


def test_share_torch_named_packages_can_install_with_no_deps(tmp_path, monkeypatch):
    """Named share_torch packages should install via a separate --no-deps step."""
    from pyisolate._internal.environment import install_dependencies

    venv_path = tmp_path / "venvs" / "test-ext"
    bin_dir = venv_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake_python = bin_dir / "python"
    fake_python.write_text("#!/bin/sh\n")
    fake_python.chmod(0o755)

    config = {
        "name": "test-ext",
        "module_path": str(tmp_path),
        "isolated": True,
        "dependencies": ["cc-torch", "torch-generic-nms", "timm", "pyyaml"],
        "share_torch": True,
        "share_torch_no_deps": ["timm"],
        "share_cuda_ipc": False,
        "sandbox_mode": "disabled",
        "sandbox": {},
        "cuda_wheels": {
            "index_url": "https://example.invalid/cuda-wheels/",
            "packages": ["cc-torch", "torch-generic-nms"],
            "package_map": {},
        },
    }

    popen_calls: list[list[str]] = []

    class FakeProc:
        def __init__(self, cmd, **kwargs):
            popen_calls.append(cmd)
            self.stdout = iter([])
            self.returncode = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self):
            return 0

    monkeypatch.setattr("subprocess.Popen", FakeProc)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(
        "pyisolate._internal.environment.exclude_satisfied_requirements",
        lambda config, reqs, python_exe: reqs,
    )
    monkeypatch.setattr(
        "pyisolate._internal.environment.get_cuda_wheel_runtime_descriptor",
        lambda: {"torch": "2.11", "cuda": "13.0", "python_tags": ["cp313"]},
    )
    monkeypatch.setattr(
        "pyisolate._internal.environment.resolve_cuda_wheel_requirements",
        lambda requirements, config: [
            "https://github.com/example/cuda-wheels/releases/download/cc_torch-latest/cc_torch-0.2-py3-none-any.whl",
            "https://github.com/example/cuda-wheels/releases/download/torch_generic_nms-latest/torch_generic_nms-0.1-py3-none-any.whl",
            "timm",
            "pyyaml",
        ],
    )

    install_dependencies(venv_path, config, "test-ext")

    assert len(popen_calls) == 3
    assert "--no-deps" not in popen_calls[0]
    assert "pyyaml" in popen_calls[0]
    assert "timm" not in popen_calls[0]
    assert "--no-deps" in popen_calls[1]
    assert "timm" in popen_calls[1]
    assert "--no-deps" in popen_calls[2]
    assert any("cc_torch-0.2-py3-none-any.whl" in token for token in popen_calls[2])
