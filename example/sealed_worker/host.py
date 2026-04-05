#!/usr/bin/env python3
"""sealed_worker example host — demonstrates sealed worker isolation.

Loads an extension with share_torch=False and execution_model=sealed_worker.
The child does NOT inherit the host's torch installation.
Exit 0 + "PASS" on success.
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
from pyisolate.config import ExtensionConfig, SandboxMode
from pyisolate.host import Extension
from pyisolate.interfaces import SerializerRegistryProtocol
from example.sealed_worker.extension import SealedWorkerExtension

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class MinimalAdapter:
    @property
    def identifier(self) -> str:
        return "sealed_worker_example"

    def get_path_config(self, module_path: str) -> dict | None:
        return {"preferred_root": os.getcwd(), "additional_paths": []}

    def setup_child_environment(self, snapshot: dict) -> None:
        pass

    def register_serializers(self, registry: SerializerRegistryProtocol) -> None:
        pass

    def provide_rpc_services(self) -> list[type[ProxiedSingleton]]:
        return []

    def handle_api_registration(self, api: ProxiedSingleton, rpc: AsyncRPC) -> None:
        pass


async def main() -> int:
    pyisolate_root = str(repo_root)
    extension_module_path = str(Path(__file__).resolve().parent / "extension")

    tmp = tempfile.mkdtemp(prefix="sealed_worker_example_")
    venv_root = os.path.join(tmp, "venvs")
    os.makedirs(venv_root, exist_ok=True)

    venv_bin = os.path.dirname(sys.executable)
    path = os.environ.get("PATH", "")
    if venv_bin not in path.split(os.pathsep):
        os.environ["PATH"] = f"{venv_bin}{os.pathsep}{path}"

    AdapterRegistry.unregister()
    AdapterRegistry.register(MinimalAdapter())

    ext = None
    try:
        config = ExtensionConfig(
            name="sealed_worker_example",
            module_path=extension_module_path,
            isolated=True,
            dependencies=[f"-e {pyisolate_root}"],
            apis=[],
            share_torch=False,
            share_cuda_ipc=False,
            execution_model="sealed_worker",
            sandbox_mode=SandboxMode.DISABLED,
            env={"PYISOLATE_SIGNAL_CLEANUP": "1"},
        )

        logger.info("Loading sealed_worker extension...")
        ext = Extension(
            module_path=extension_module_path,
            extension_type=SealedWorkerExtension,
            config=config,
            venv_root_path=venv_root,
        )
        ext.ensure_process_started()

        proxy = ext.get_proxy()
        result = await proxy.ping()
        logger.info(f"ping result: {result}")

        if result == "pong_sealed":
            logger.info("PASS — sealed_worker example completed successfully")
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
