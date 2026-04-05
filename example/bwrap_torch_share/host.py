#!/usr/bin/env python3
"""bwrap + torch_share example — sandbox-enforced host-coupled isolation.

Same as torch_share but with sandbox_mode=REQUIRED. The child runs
inside a bubblewrap sandbox with deny-by-default filesystem.
Linux only. Exit 0 + "PASS" on success.
"""

import asyncio
import contextlib
import logging
import os
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from pyisolate._internal.adapter_registry import AdapterRegistry
from pyisolate._internal.rpc_protocol import AsyncRPC, ProxiedSingleton
from pyisolate._internal.sandbox_detect import detect_sandbox_capability
from pyisolate.config import ExtensionConfig, SandboxMode
from pyisolate.host import Extension
from pyisolate.interfaces import SerializerRegistryProtocol
from tests.harness.test_package import ReferenceTestExtension

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class MinimalAdapter:
    @property
    def identifier(self) -> str:
        return "bwrap_torch_share_example"

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
    if sys.platform != "linux":
        logger.error("bwrap examples are Linux-only")
        return 1

    cap = detect_sandbox_capability()
    if not cap.available:
        logger.error(f"bwrap not available: {cap.restriction_model}")
        return 1

    pyisolate_root = str(repo_root)
    test_pkg_path = str(Path(repo_root) / "tests" / "harness" / "test_package")

    tmp = tempfile.mkdtemp(prefix="bwrap_torch_share_")
    venv_root = os.path.join(tmp, "venvs")
    os.makedirs(venv_root, exist_ok=True)
    shared_tmp = os.path.join(tmp, "ipc_shared")
    os.makedirs(shared_tmp, exist_ok=True)
    os.environ["TMPDIR"] = shared_tmp

    venv_bin = os.path.dirname(sys.executable)
    path = os.environ.get("PATH", "")
    if venv_bin not in path.split(os.pathsep):
        os.environ["PATH"] = f"{venv_bin}{os.pathsep}{path}"

    AdapterRegistry.unregister()
    AdapterRegistry.register(MinimalAdapter())

    ext = None
    try:
        config = ExtensionConfig(
            name="bwrap_torch_share_example",
            module_path=test_pkg_path,
            isolated=True,
            dependencies=[f"-e {pyisolate_root}"],
            apis=[],
            share_torch=True,
            share_cuda_ipc=False,
            sandbox_mode=SandboxMode.REQUIRED,
            sandbox={"writable_paths": [shared_tmp, "/dev/shm"]},
            env={"PYISOLATE_SIGNAL_CLEANUP": "1"},
        )

        logger.info("Loading bwrap + torch_share extension...")
        ext = Extension(
            module_path=test_pkg_path,
            extension_type=ReferenceTestExtension,
            config=config,
            venv_root_path=venv_root,
        )
        ext.ensure_process_started()

        proxy = ext.get_proxy()
        result = await proxy.ping()
        logger.info(f"ping result: {result}")

        if result == "pong":
            logger.info("PASS — bwrap + torch_share example completed successfully")
            return 0
        else:
            logger.error(f"FAIL — unexpected ping result: {result}")
            return 1
    finally:
        if ext is not None:
            with contextlib.suppress(Exception):
                ext.stop()
        AdapterRegistry.unregister()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
