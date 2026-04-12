"""Process-level integration tests for the sealed conda runtime."""

from __future__ import annotations

import contextlib
import gc
import os
import shutil
import site
import uuid
from pathlib import Path
from typing import cast

import pytest
import torch  # noqa: E402

from pyisolate._internal.host import Extension  # noqa: E402
from pyisolate.config import ExtensionConfig  # noqa: E402
from pyisolate.sealed import SealedNodeExtension  # noqa: E402

PIXI_AVAILABLE = shutil.which("pixi") is not None

pytestmark = pytest.mark.skipif(not PIXI_AVAILABLE, reason="pixi not on PATH")


def _expected_pixi_python(env_path: Path) -> Path:
    if os.name == "nt":
        return env_path / ".pixi" / "envs" / "default" / "python.exe"
    return env_path / ".pixi" / "envs" / "default" / "bin" / "python"


def _shm_snapshot() -> set[str]:
    shm_root = Path("/dev/shm")
    if os.name == "nt" or not shm_root.exists():
        return set()
    return {path.name for path in shm_root.glob("torch_*")}


def _build_conda_config(fixture_path: Path, run_dir: Path) -> ExtensionConfig:
    # Inlined from fixtures/conda_sealed_node/pyproject.toml — no TOML parser needed.
    return cast(
        ExtensionConfig,
        {
        "name": "conda-sealed-node",
        "module_path": str(fixture_path),
        "isolated": True,
        "dependencies": ["packaging"],
        "apis": [],
        "env": {
            "PYISOLATE_ARTIFACT_DIR": str(run_dir / "artifacts"),
            "PYISOLATE_SIGNAL_CLEANUP": "1",
        },
        "share_torch": False,
        "share_cuda_ipc": False,
        "sandbox": {"writable_paths": [str(run_dir / "artifacts")]},
        "package_manager": "conda",
        "execution_model": "sealed_worker",
        "conda_channels": ["conda-forge"],
        "conda_dependencies": ["boltons", "numpy"],
        },
    )


@pytest.mark.asyncio
async def test_conda_sealed_runtime_avoids_host_path_leakage() -> None:
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "conda_sealed_node"
    run_root = Path(__file__).resolve().parent.parent / ".pytest_artifacts" / "conda_integration"
    run_dir = run_root / uuid.uuid4().hex
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    venv_root = run_dir / "venvs"
    venv_root.mkdir(parents=True, exist_ok=True)
    config = _build_conda_config(fixture_path, run_dir)

    ext = Extension(
        module_path=str(fixture_path),
        extension_type=SealedNodeExtension,
        config=config,
        venv_root_path=str(venv_root),
    )

    try:
        ext.ensure_process_started()
        proxy = ext.get_proxy()

        nodes = await proxy.list_nodes()
        assert nodes == {
            "InspectRuntime": "Inspect Runtime",
            "EchoTensor": "Echo Tensor",
            "OpenWeatherDataset": "Open Weather Dataset",
        }

        pixi_manifest = (ext.venv_path / "pixi.toml").read_text(encoding="utf-8")
        assert 'boltons = "*"' in pixi_manifest
        assert 'packaging = "*"' in pixi_manifest

        (
            path_dump,
            host_leak_report,
            python_exe,
        ) = await proxy.execute_node("InspectRuntime")
        path_entries = path_dump.splitlines()
        assert str(fixture_path) in path_entries
        assert site.getusersitepackages() not in path_dump
        assert python_exe == str(_expected_pixi_python(ext.venv_path))

        weather_sum, grib_path = await proxy.execute_node("OpenWeatherDataset")
        assert weather_sum == pytest.approx(10.0)
        assert Path(grib_path).exists()

        shm_before = _shm_snapshot()
        input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        echoed_tensor, saw_json_tensor = await proxy.execute_node("EchoTensor", tensor=input_tensor)
        shm_after = _shm_snapshot()

        assert torch.equal(echoed_tensor, input_tensor)
        assert saw_json_tensor is False
        launch_args = " ".join(str(part) for part in ext.proc.args)
        if os.name != "nt":
            assert shm_after == shm_before
            assert launch_args.startswith("bwrap ")
        assert str(_expected_pixi_python(ext.venv_path)) in launch_args
        assert "PYTHONPATH" not in launch_args
    finally:
        with contextlib.suppress(Exception):
            if "proxy" in locals():
                await proxy.flush_transport_state()
        with contextlib.suppress(UnboundLocalError):
            del echoed_tensor
        with contextlib.suppress(UnboundLocalError):
            del input_tensor
        gc.collect()
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.synchronize()
        ext.stop()
        shutil.rmtree(run_dir, ignore_errors=True)
