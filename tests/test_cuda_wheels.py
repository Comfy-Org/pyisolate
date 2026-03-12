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
    get_cuda_wheel_runtime,
    resolve_cuda_wheel_requirements,
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

    monkeypatch.setattr("pyisolate._internal.cuda_wheels.get_cuda_wheel_runtime", lambda: runtime)
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

    monkeypatch.setattr("pyisolate._internal.cuda_wheels.get_cuda_wheel_runtime", lambda: runtime)
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

    monkeypatch.setattr("pyisolate._internal.cuda_wheels.get_cuda_wheel_runtime", lambda: runtime)
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

    monkeypatch.setattr("pyisolate._internal.cuda_wheels.get_cuda_wheel_runtime", lambda: runtime)
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

    monkeypatch.setattr("pyisolate._internal.cuda_wheels.get_cuda_wheel_runtime", lambda: runtime)
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

    monkeypatch.setattr("pyisolate._internal.cuda_wheels.get_cuda_wheel_runtime", lambda: runtime)
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

    monkeypatch.setattr("pyisolate._internal.cuda_wheels.get_cuda_wheel_runtime", lambda: runtime)
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
