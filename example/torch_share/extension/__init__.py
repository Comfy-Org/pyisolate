"""torch_share example extension — runs in the isolated child process.

Demonstrates a host-coupled extension with share_torch=True.
The child has access to the host's torch installation via shared memory.
"""

import logging
from typing import Any

from pyisolate.shared import ExtensionBase

logger = logging.getLogger(__name__)


class TorchShareExtension(ExtensionBase):
    async def on_module_loaded(self, module: Any) -> None:
        logger.info("[torch_share] Extension loaded in child")

    async def ping(self) -> str:
        return "pong_torch_share"

    async def compute(self, x: int) -> dict[str, Any]:
        """Demonstrate torch is available in child and return a result."""
        try:
            import torch
            t = torch.tensor([float(x)])
            return {"value": x * 2, "torch_available": True, "tensor_sum": float(t.sum())}
        except ImportError:
            return {"value": x * 2, "torch_available": False, "tensor_sum": 0.0}


def extension_entrypoint() -> ExtensionBase:
    return TorchShareExtension()
