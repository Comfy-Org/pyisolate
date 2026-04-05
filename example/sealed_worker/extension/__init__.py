"""sealed_worker example extension — runs in an isolated child with no shared torch.

Demonstrates sealed_worker mode: the child has its own Python environment
and communicates with the host purely via JSON-RPC.
"""

import logging
import sys
from typing import Any

from pyisolate.shared import ExtensionBase

logger = logging.getLogger(__name__)


class SealedWorkerExtension(ExtensionBase):
    async def on_module_loaded(self, module: Any) -> None:
        logger.info("[sealed_worker] Extension loaded in child")

    async def ping(self) -> str:
        return "pong_sealed"

    async def get_python_version(self) -> str:
        return sys.version

    async def check_torch_absent(self) -> bool:
        """Return True if torch is NOT importable (expected for sealed_worker)."""
        try:
            import torch  # noqa: F401
            return False
        except ImportError:
            return True


def extension_entrypoint() -> ExtensionBase:
    return SealedWorkerExtension()
