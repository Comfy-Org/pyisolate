"""Simple V1-style node fixture for sealed conda integration tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


class InspectRuntimeNode:
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "path_dump",
        "host_leak_report",
        "python_exe",
    )
    FUNCTION = "inspect"
    CATEGORY = "tests"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802
        return {"required": {}}

    def inspect(self) -> tuple[str, str, str]:
        path_dump = "\n".join(sys.path)
        host_leak_report = f"sys_path_count={len(sys.path)}"
        return (path_dump, host_leak_report, sys.executable)


class EchoTensorNode:
    RETURN_TYPES = ("TENSOR", "BOOLEAN")
    RETURN_NAMES = ("tensor", "saw_json_tensor")
    FUNCTION = "echo"
    CATEGORY = "tests"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - Comfy node API requires this name
        return {"required": {"tensor": ("TENSOR",)}}

    def echo(self, tensor: Any) -> tuple[Any, bool]:
        saw_json_tensor = isinstance(tensor, dict) and tensor.get("__type__") == "TensorValue"
        return (tensor, saw_json_tensor)


class OpenWeatherDatasetNode:
    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("sum_value", "grib_path")
    FUNCTION = "open_dataset"
    CATEGORY = "tests"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802 - Comfy node API requires this name
        return {"required": {}}

    def open_dataset(self) -> tuple[float, str]:
        from boltons import strutils  # type: ignore[import-untyped]
        from packaging.version import Version

        artifact_dir = Path(os.environ["PYISOLATE_ARTIFACT_DIR"])
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "weather_fixture.txt"
        label = strutils.slugify("Open Weather Dataset", delim="_")
        total = 10.0
        artifact_path.write_text(
            f"label={label}\npackaging={Version('1.0')}\nsum={total}\n",
            encoding="utf-8",
        )
        return (total, str(artifact_path))


NODE_CLASS_MAPPINGS = {
    "InspectRuntime": InspectRuntimeNode,
    "EchoTensor": EchoTensorNode,
    "OpenWeatherDataset": OpenWeatherDatasetNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "InspectRuntime": "Inspect Runtime",
    "EchoTensor": "Echo Tensor",
    "OpenWeatherDataset": "Open Weather Dataset",
}
