from typing import Any
from pyisolate._internal import host


class FakeAdapter:
    identifier = "fake"

    def __init__(self, preferred_root: Any = "/tmp/ComfyUI") -> None:
        self.preferred_root = preferred_root

    def get_path_config(self, module_path: Any) -> Any:
        return {
            "preferred_root": self.preferred_root,
            "additional_paths": [f"{self.preferred_root}/custom_nodes"],
        }

    def setup_child_environment(self, snapshot: Any) -> Any:
        return None

    def register_serializers(self, registry: Any) -> Any:
        return None

    def provide_rpc_services(self) -> Any:
        return []

    def handle_api_registration(self, api: Any, rpc: Any) -> Any:
        return None


def test_build_extension_snapshot_includes_adapter(monkeypatch: Any) -> None:
    from pyisolate._internal.adapter_registry import AdapterRegistry

    monkeypatch.setattr(AdapterRegistry, "get", lambda: FakeAdapter())

    snapshot = host.build_extension_snapshot("/tmp/ComfyUI/custom_nodes/demo")

    assert "sys_path" in snapshot
    assert snapshot["adapter_name"] == "fake"
    preferred_root = snapshot["preferred_root"]
    assert isinstance(preferred_root, str)
    assert preferred_root.endswith("ComfyUI")
    assert snapshot.get("additional_paths")
    context_data = snapshot.get("context_data", {})
    assert isinstance(context_data, dict)
    assert context_data.get("module_path") == "/tmp/ComfyUI/custom_nodes/demo"


def test_build_extension_snapshot_no_adapter(monkeypatch: Any) -> None:
    from pyisolate._internal.adapter_registry import AdapterRegistry

    monkeypatch.setattr(AdapterRegistry, "get", lambda: None)

    snapshot = host.build_extension_snapshot("/tmp/nowhere")
    assert "sys_path" in snapshot
    assert snapshot["adapter_name"] is None
    assert snapshot.get("preferred_root") is None
    assert snapshot.get("additional_paths") == []
