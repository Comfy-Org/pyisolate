#!/usr/bin/env python3
"""torch_share example host — demonstrates host-coupled isolation with shared torch.

Loads an extension in a child process with share_torch=True, calls methods
via RPC, and verifies the round-trip. Exit 0 + "PASS" on success.
"""

import asyncio
import contextlib
import logging
import os
import sys
import tempfile
from pathlib import Path

# Ensure pyisolate is importable from the repo root
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from pyisolate._internal.adapter_registry import AdapterRegistry
from pyisolate._internal.rpc_protocol import AsyncRPC, ProxiedSingleton
from pyisolate._internal.sandbox_detect import detect_sandbox_capability
from pyisolate.config import ExtensionConfig, SandboxMode
from pyisolate.host import Extension
from pyisolate.interfaces import SerializerRegistryProtocol

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class MinimalAdapter:
    @property
    def identifier(self) -> str:
        return "torch_share_example"

    def get_path_config(self, module_path: str) -> dict | None:
        return {"preferred_root": os.getcwd(), "additional_paths": []}

    def setup_child_environment(self, snapshot: dict) -> None:
        pass

    def register_serializers(self, registry: SerializerRegistryProtocol) -> None:
        try:
            from pyisolate._internal.tensor_serializer import deserialize_tensor, serialize_tensor
            registry.register("torch.Tensor", serialize_tensor, deserialize_tensor)
        except Exception:
            pass

    def provide_rpc_services(self) -> list[type[ProxiedSingleton]]:
        return []

    def handle_api_registration(self, api: ProxiedSingleton, rpc: AsyncRPC) -> None:
        pass


async def main() -> int:
    pyisolate_root = str(repo_root)
    example_dir = Path(__file__).resolve().parent
    extension_path = str(example_dir / "extension")

    tmp = tempfile.mkdtemp(prefix="torch_share_example_")
    venv_root = os.path.join(tmp, "venvs")
    os.makedirs(venv_root, exist_ok=True)

    # Shared temp for torch file_system IPC
    shared_tmp = os.path.join(tmp, "ipc_shared")
    os.makedirs(shared_tmp, exist_ok=True)
    os.environ["TMPDIR"] = shared_tmp

    # Ensure uv is findable
    venv_bin = os.path.dirname(sys.executable)
    path = os.environ.get("PATH", "")
    if venv_bin not in path.split(os.pathsep):
        os.environ["PATH"] = f"{venv_bin}{os.pathsep}{path}"

    AdapterRegistry.unregister()
    AdapterRegistry.register(MinimalAdapter())

    try:
        # Use the test harness extension which has ping/compute methods
        # already defined on the type so the host-side proxy can see them.
        from tests.harness.test_package import ReferenceTestExtension

        config = ExtensionConfig(
            name="torch_share_example",
            module_path=str(Path(repo_root) / "tests" / "harness" / "test_package"),
            isolated=True,
            dependencies=[f"-e {pyisolate_root}"],
            apis=[],
            share_torch=True,
            share_cuda_ipc=False,
            sandbox_mode=SandboxMode.DISABLED,
            sandbox={"writable_paths": [shared_tmp]},
            env={"PYISOLATE_SIGNAL_CLEANUP": "1"},
        )

        logger.info("Loading torch_share extension...")
        ext = Extension(
            module_path=str(Path(repo_root) / "tests" / "harness" / "test_package"),
            extension_type=ReferenceTestExtension,
            config=config,
            venv_root_path=venv_root,
        )
        ext.ensure_process_started()

        proxy = ext.get_proxy()
        result = await proxy.ping()
        logger.info(f"ping result: {result}")

        if result == "pong":
            logger.info("PASS — torch_share example completed successfully")
            return 0
        else:
            logger.error(f"FAIL — unexpected ping result: {result}")
            return 1
    finally:
        with contextlib.suppress(Exception):
            ext.stop()
        AdapterRegistry.unregister()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
