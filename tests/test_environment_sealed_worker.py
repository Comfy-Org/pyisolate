from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

from pyisolate._internal import environment
from pyisolate.config import ExtensionConfig


def _mock_venv_python(venv_path: Path) -> None:
    python_exe = venv_path / "Scripts" / "python.exe" if os.name == "nt" else venv_path / "bin" / "python"
    python_exe.parent.mkdir(parents=True, exist_ok=True)
    python_exe.write_text("#!/usr/bin/env python\n", encoding="utf-8")


def _capture_install_commands(monkeypatch: Any) -> list[list[str]]:
    popen_calls: list[list[str]] = []

    class MockPopen:
        def __init__(self, cmd: Any, **kwargs: Any) -> None:
            popen_calls.append(cmd)
            self.stdout = io.StringIO("installed\n")

        def wait(self) -> Any:
            return 0

        def __enter__(self) -> Any:
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
            return False

    monkeypatch.setattr(environment.shutil, "which", lambda binary: "/usr/bin/uv")
    monkeypatch.setattr(environment.subprocess, "Popen", MockPopen)
    return popen_calls


def test_sealed_worker_uv_does_not_auto_inject_torch(monkeypatch: Any, tmp_path: Path) -> None:
    venv_path = tmp_path / "venv"
    _mock_venv_python(venv_path)
    popen_calls = _capture_install_commands(monkeypatch)

    config: ExtensionConfig = {
        "name": "demo",
        "isolated": True,
        "dependencies": ["boltons"],
        "apis": [],
        "share_torch": False,
        "share_cuda_ipc": False,
        "execution_model": "sealed_worker",
    }

    environment.install_dependencies(venv_path, config, "demo")

    assert len(popen_calls) == 1
    cmd = popen_calls[0]
    assert "boltons" in cmd
    assert not any(str(part).startswith("torch==") for part in cmd)


def test_host_coupled_uv_still_auto_injects_torch(monkeypatch: Any, tmp_path: Path) -> None:
    venv_path = tmp_path / "venv"
    _mock_venv_python(venv_path)
    popen_calls = _capture_install_commands(monkeypatch)

    config: ExtensionConfig = {
        "name": "demo",
        "isolated": True,
        "dependencies": ["boltons"],
        "apis": [],
        "share_torch": False,
        "share_cuda_ipc": False,
    }

    environment.install_dependencies(venv_path, config, "demo")

    assert len(popen_calls) == 1
    cmd = popen_calls[0]
    assert "boltons" in cmd
    assert any(str(part).startswith("torch==") for part in cmd)
