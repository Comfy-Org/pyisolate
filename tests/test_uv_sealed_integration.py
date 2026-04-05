"""Process-level integration tests for the toolkit-owned uv sealed worker."""

from __future__ import annotations

import contextlib
import gc
import os
import shutil
import site
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import torch  # noqa: E402

from pyisolate._internal.host import Extension  # noqa: E402
from pyisolate.sealed import SealedNodeExtension  # noqa: E402

UV_BIN = Path(sys.executable).with_name("uv.exe" if os.name == "nt" else "uv")
UV_AVAILABLE = shutil.which("uv") is not None or UV_BIN.exists()
BWRAP_AVAILABLE = os.name == "nt" or shutil.which("bwrap") is not None

pytestmark = [
    pytest.mark.skipif(not UV_AVAILABLE, reason="uv not on PATH"),
    pytest.mark.skipif(not BWRAP_AVAILABLE, reason="bwrap not on PATH"),
]


def _fixture_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "uv_sealed_worker"


def _shm_snapshot() -> set[str]:
    shm_root = Path("/dev/shm")
    if os.name == "nt" or not shm_root.exists():
        return set()
    return {path.name for path in shm_root.glob("torch_*")}


def _host_site_root() -> str:
    return str(Path(sys.executable).resolve().parents[1])


def _build_uv_config(fixture_path: Path, run_dir: Path) -> dict:
    # Inlined from fixtures/uv_sealed_worker/pyproject.toml — no TOML parser needed.
    return {
        "name": "uv-sealed-worker",
        "module_path": str(fixture_path),
        "isolated": True,
        "dependencies": ["boltons"],
        "apis": [],
        "env": {
            "PYISOLATE_ARTIFACT_DIR": str(run_dir / "artifacts"),
            "PYISOLATE_SIGNAL_CLEANUP": "1",
        },
        "share_torch": False,
        "share_cuda_ipc": False,
        "sandbox": {"writable_paths": [str(run_dir / "artifacts")]},
        "package_manager": "uv",
        "execution_model": "sealed_worker",
    }


@pytest.mark.asyncio
async def test_uv_sealed_runtime_uses_toolkit_fixture_without_host_leakage() -> None:
    fixture_path = _fixture_path()
    run_root = Path(__file__).resolve().parent.parent / ".pytest_artifacts" / "uv_sealed_integration"
    run_dir = run_root / uuid.uuid4().hex
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    venv_root = run_dir / "venvs"
    venv_root.mkdir(parents=True, exist_ok=True)
    config = _build_uv_config(fixture_path, run_dir)

    ext = Extension(
        module_path=str(fixture_path),
        extension_type=SealedNodeExtension,
        config=config,
        venv_root_path=str(venv_root),
    )

    try:
        path_env = os.environ.get("PATH", "")
        uv_path = f"{UV_BIN.parent}{os.pathsep}{path_env}" if UV_BIN.exists() else path_env
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv("PATH", uv_path)
            monkeypatch.setenv("PYISOLATE_ARTIFACT_DIR", str(run_dir / "artifacts"))
            wheel_index = "https://pollockjj.github.io/wheels/"
            monkeypatch.setenv("UV_EXTRA_INDEX_URL", wheel_index)
            print(f"UV_EXTRA_INDEX_URL={wheel_index}")
            try:
                ext.ensure_process_started()
            except RuntimeError as exc:
                if "bubblewrap" in str(exc).lower():
                    pytest.skip(f"bwrap unavailable on this platform: {exc}")
                raise
            # Verify pyisolate was installed in the child venv from the published index
            if os.name == "nt":
                child_python = Path(ext.venv_path) / "Scripts" / "python.exe"
            else:
                child_python = Path(ext.venv_path) / "bin" / "python3"
            pip_show = subprocess.run(
                [str(child_python), "-m", "pip", "show", "pyisolate"],
                capture_output=True,
                text=True,
                check=False,
            )
            print(f"child venv pyisolate install:\n{pip_show.stdout}")
            assert "pyisolate" in pip_show.stdout, "pyisolate not installed in child venv"
            assert "0.10.1" in pip_show.stdout, "pyisolate version mismatch in child venv"
        proxy = ext.get_proxy()

        nodes = await proxy.list_nodes()
        assert nodes == {
            "UVSealedRuntimeProbe": "UV Sealed Runtime Probe",
            "UVSealedBoltonsSlugify": "UV Sealed Boltons Slugify",
            "UVSealedFilesystemBarrier": "UV Sealed Filesystem Barrier",
            "UVSealedTensorEcho": "UV Sealed Tensor Echo",
            "UVSealedLatentEcho": "UV Sealed Latent Echo",
        }

        (
            path_dump,
            boltons_origin,
            report,
            saw_user_site,
        ) = await proxy.execute_node("UVSealedRuntimeProbe")
        print(report)
        print(f"child boltons origin: {boltons_origin}")

        assert str(ext.venv_path) in boltons_origin
        assert _host_site_root() not in boltons_origin
        assert site.getusersitepackages() not in path_dump
        assert saw_user_site is False

        slug, slug_origin = await proxy.execute_node(
            "UVSealedBoltonsSlugify", text="Sealed Worker Still Works"
        )
        assert slug == "sealed_worker_still_works"
        assert slug_origin == boltons_origin

        (
            barrier_report,
            outside_blocked,
            module_mutation_blocked,
            artifact_write_ok,
        ) = await proxy.execute_node("UVSealedFilesystemBarrier")
        print(barrier_report)
        assert artifact_write_ok is True
        if os.name != "nt":
            assert outside_blocked is True
            assert module_mutation_blocked is True

        shm_before = _shm_snapshot()
        input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        echoed_tensor, saw_json_tensor = await proxy.execute_node("UVSealedTensorEcho", tensor=input_tensor)
        shm_after = _shm_snapshot()

        max_abs = float((echoed_tensor - input_tensor).abs().max().item())
        print(f"tensor roundtrip max_abs={max_abs:.8f}")
        print(f"launch args={ext.proc.args}")

        assert torch.equal(echoed_tensor, input_tensor)
        assert max_abs <= 1e-5
        assert saw_json_tensor is True
        launch_args = " ".join(str(part) for part in ext.proc.args)
        if os.name != "nt":
            assert shm_after == shm_before
            assert launch_args.startswith("bwrap ")
        assert "pyisolate._internal.uds_client" in launch_args

        artifact_dir = run_dir / "artifacts"
        assert (artifact_dir / "child_bootstrap_paths.txt").exists()
        # child_import_trace.txt is only written by setup_child_environment,
        # which sealed workers skip (no host sys.path application).
        assert (artifact_dir / "filesystem_barrier_probe.txt").exists()
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
        if getattr(ext, "proc", None) is not None:
            assert ext.proc.poll() is not None
        shutil.rmtree(run_dir, ignore_errors=True)
