"""Repo-owned uv sealed worker fixture for pyisolate integration tests."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path
from typing import Any


def _artifact_dir() -> Path:
    artifact_dir = Path(os.environ["PYISOLATE_ARTIFACT_DIR"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


class UVSealedRuntimeProbeNode:
    RETURN_TYPES = ("STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = (
        "path_dump",
        "boltons_origin",
        "report",
        "saw_user_site",
    )
    FUNCTION = "probe"
    CATEGORY = "tests"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802
        return {"required": {}}

    def probe(self) -> tuple[str, str, str, bool]:
        from boltons import strutils  # type: ignore[import-untyped]

        artifact_dir = _artifact_dir()
        path_dump = "\n".join(sys.path)
        user_site = site.getusersitepackages()
        saw_user_site = user_site in sys.path
        report = f"python={sys.executable}\nuser_site={user_site}\npaths={len(sys.path)}"

        (artifact_dir / "child_bootstrap_paths.txt").write_text(path_dump, encoding="utf-8")
        (artifact_dir / "child_dependency_dump.txt").write_text(strutils.__file__, encoding="utf-8")
        return (
            path_dump,
            strutils.__file__,
            report,
            saw_user_site,
        )


class UVSealedBoltonsSlugifyNode:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("slug", "slug_origin")
    FUNCTION = "slug"
    CATEGORY = "tests"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802
        return {"required": {"text": ("STRING",)}}

    def slug(self, text: str) -> tuple[str, str]:
        from boltons import strutils  # type: ignore[import-untyped]

        return (strutils.slugify(text, delim="_"), strutils.__file__)


class UVSealedFilesystemBarrierNode:
    RETURN_TYPES = ("STRING", "BOOLEAN", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("report", "outside_blocked", "module_mutation_blocked", "artifact_write_ok")
    FUNCTION = "probe"
    CATEGORY = "tests"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802
        return {"required": {}}

    def probe(self) -> tuple[str, bool, bool, bool]:
        artifact_dir = _artifact_dir()
        fixture_dir = Path(__file__).resolve().parent
        outside_probe = (
            Path("/usr/.__outside_probe.txt")
            if os.name != "nt"
            else fixture_dir.parent / ".__outside_probe.txt"
        )
        module_probe = fixture_dir / ".__module_probe.txt"
        artifact_probe = artifact_dir / "filesystem_barrier_probe.txt"

        outside_blocked = False
        module_mutation_blocked = False
        artifact_write_ok = False

        try:
            outside_probe.write_text("probe", encoding="utf-8")
        except Exception:
            outside_blocked = True
        else:
            outside_probe.unlink(missing_ok=True)

        try:
            module_probe.write_text("probe", encoding="utf-8")
        except Exception:
            module_mutation_blocked = True
        else:
            module_probe.unlink(missing_ok=True)

        artifact_probe.write_text("ok", encoding="utf-8")
        artifact_write_ok = artifact_probe.exists()

        report = (
            f"outside_blocked={outside_blocked}\n"
            f"module_mutation_blocked={module_mutation_blocked}\n"
            f"artifact_write_ok={artifact_write_ok}"
        )
        return (report, outside_blocked, module_mutation_blocked, artifact_write_ok)


class UVSealedTensorEchoNode:
    RETURN_TYPES = ("TENSOR", "BOOLEAN")
    RETURN_NAMES = ("tensor", "saw_json_tensor")
    FUNCTION = "echo"
    CATEGORY = "tests"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802
        return {"required": {"tensor": ("TENSOR",)}}

    def echo(self, tensor: Any) -> tuple[Any, bool]:
        saw_json_tensor = isinstance(tensor, dict) and tensor.get("__type__") == "TensorValue"
        return (tensor, saw_json_tensor)


class UVSealedLatentEchoNode:
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "echo"
    CATEGORY = "tests"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802
        return {"required": {"latent": ("LATENT",)}}

    def echo(self, latent: Any) -> tuple[Any]:
        return (latent,)


NODE_CLASS_MAPPINGS = {
    "UVSealedRuntimeProbe": UVSealedRuntimeProbeNode,
    "UVSealedBoltonsSlugify": UVSealedBoltonsSlugifyNode,
    "UVSealedFilesystemBarrier": UVSealedFilesystemBarrierNode,
    "UVSealedTensorEcho": UVSealedTensorEchoNode,
    "UVSealedLatentEcho": UVSealedLatentEchoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UVSealedRuntimeProbe": "UV Sealed Runtime Probe",
    "UVSealedBoltonsSlugify": "UV Sealed Boltons Slugify",
    "UVSealedFilesystemBarrier": "UV Sealed Filesystem Barrier",
    "UVSealedTensorEcho": "UV Sealed Tensor Echo",
    "UVSealedLatentEcho": "UV Sealed Latent Echo",
}
