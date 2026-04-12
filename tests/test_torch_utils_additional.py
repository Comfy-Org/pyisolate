from typing import Any
from types import SimpleNamespace

from pyisolate._internal import torch_utils


def test_get_torch_ecosystem_packages_includes_distributions(monkeypatch: Any) -> Any:
    def fake_distributions() -> Any:
        meta = SimpleNamespace(metadata={"Name": "nvidia-cublas"})
        meta2 = SimpleNamespace(metadata={"Name": "torch-hub"})
        return [meta, meta2]

    torch_utils.get_torch_ecosystem_packages.cache_clear()
    monkeypatch.setattr(torch_utils.importlib_metadata, "distributions", fake_distributions)
    pkgs = torch_utils.get_torch_ecosystem_packages()
    assert "nvidia-cublas" in pkgs
    assert "torch-hub" in pkgs


def test_get_torch_ecosystem_packages_handles_exception(monkeypatch: Any) -> None:
    def bad_distributions() -> None:
        raise RuntimeError("boom")

    torch_utils.get_torch_ecosystem_packages.cache_clear()
    monkeypatch.setattr(torch_utils.importlib_metadata, "distributions", bad_distributions)
    pkgs = torch_utils.get_torch_ecosystem_packages()
    assert "torch" in pkgs  # base set still returned
