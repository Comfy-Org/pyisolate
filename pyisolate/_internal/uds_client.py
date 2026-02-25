"""Entry point for isolated child processes (JSON-RPC).

This module is invoked by the host process as `python -m pyisolate._internal.uds_client`.
It connects to the host via Unix Domain Socket (UDS) using JSON-RPC,
receives bootstrap configuration, and delegates to the standard async_entrypoint.

This replaces the old pickle-based client.py entrypoint.

Environment variables expected:
- PYISOLATE_UDS_ADDRESS: Path to the Unix socket to connect to
- PYISOLATE_CHILD: Set to "1" by the host
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal  # Required for graceful shutdown handling
import socket
import sys
from contextlib import AbstractContextManager as ContextManager
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ..config import ExtensionConfig

from .tensor_serializer import register_tensor_serializer
from .torch_gate import get_torch_optional

logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for isolated child processes."""

    def handle_signal(signum: int, frame: Any) -> None:
        logger.info("Received signal %s. Initiating graceful shutdown...", signum)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    # -------------------------------------------------------------------------

    logging.basicConfig(format="%(message)s", level=logging.INFO, force=True)

    # Get UDS address from environment
    uds_address = os.environ.get("PYISOLATE_UDS_ADDRESS")
    if not uds_address:
        raise RuntimeError(
            "PYISOLATE_UDS_ADDRESS not set. This module should only be invoked via host launcher."
        )

    # Connect to host - supports both UDS paths and TCP addresses
    logger.debug("Connecting to host at %s", uds_address)
    if uds_address.startswith("tcp://"):
        # TCP fallback for Windows without AF_UNIX
        import re

        match = re.match(r"tcp://([^:]+):(\d+)", uds_address)
        if not match:
            raise RuntimeError(f"Invalid TCP address format: {uds_address}")
        host, port = match.group(1), int(match.group(2))
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.connect((host, port))
    else:
        # Unix Domain Socket
        client_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)  # type: ignore[attr-defined]
        client_sock.connect(uds_address)

    # Create JSON transport (no pickle)
    from .rpc_transports import JSONSocketTransport

    transport = JSONSocketTransport(client_sock)

    # Receive bootstrap data from host via JSON
    bootstrap_data = transport.recv()
    logger.debug("Received bootstrap data")

    # Apply host snapshot to environment
    snapshot = bootstrap_data.get("snapshot", {})
    os.environ["PYISOLATE_HOST_SNAPSHOT"] = json.dumps(snapshot)
    os.environ["PYISOLATE_CHILD"] = "1"

    # Bootstrap the child environment (apply sys.path, etc.)
    from .bootstrap import bootstrap_child

    bootstrap_child()

    # Import remaining dependencies after bootstrap
    from ..shared import ExtensionBase

    # Extract configuration from bootstrap data
    config: ExtensionConfig = bootstrap_data["config"]
    module_path: str = config["module_path"]

    # Extension type is serialized as "module.classname" string reference
    ext_type_ref = bootstrap_data.get("extension_type_ref", "pyisolate.shared.ExtensionBase")

    # Resolve extension type from string reference
    try:
        parts = ext_type_ref.rsplit(".", 1)
        if len(parts) == 2:
            import importlib

            module = importlib.import_module(parts[0])
            extension_type = getattr(module, parts[1])
        else:
            extension_type = ExtensionBase
    except Exception as e:
        logger.warning("Could not resolve extension type %s: %s", ext_type_ref, e)
        extension_type = ExtensionBase

    # Run the async entrypoint
    asyncio.run(
        _async_uds_entrypoint(
            transport=transport,
            module_path=module_path,
            extension_type=extension_type,
            config=config,
        )
    )


async def _async_uds_entrypoint(
    transport: Any,
    module_path: str,
    extension_type: type[Any],
    config: ExtensionConfig,
) -> None:
    """Async entrypoint for isolated processes using JSON-RPC transport."""
    from ..interfaces import IsolationAdapter
    from .rpc_protocol import (
        AsyncRPC,
        ProxiedSingleton,
        set_child_rpc_instance,
    )

    # RPC uses the existing JSONSocketTransport
    rpc = AsyncRPC(transport=transport)
    set_child_rpc_instance(rpc)

    torch, _ = get_torch_optional()
    if torch is not None:
        # Register tensor serializer only when torch is available.
        from .serialization_registry import SerializerRegistry

        register_tensor_serializer(SerializerRegistry.get_instance())
        # Ensure file_system strategy for CPU tensors.
        torch.multiprocessing.set_sharing_strategy("file_system")
    elif config.get("share_torch", False):
        raise RuntimeError(
            "share_torch=True requires PyTorch. Install 'torch' to use tensor-sharing features."
        )

    # Instantiate extension
    extension = extension_type()
    extension._initialize_rpc(rpc)

    try:
        await extension.before_module_loaded()
    except Exception as exc:
        logger.error("Extension before_module_loaded failed: %s", exc, exc_info=True)
        raise

    # Set up torch inference mode if share_torch enabled
    context: ContextManager[Any] = nullcontext()
    if config.get("share_torch", False):
        assert torch is not None
        context = cast(ContextManager[Any], torch.inference_mode())

    if not os.path.isdir(module_path):
        raise ValueError(f"Module path {module_path} is not a directory.")

    # v1.0: Child adapter was registered during bootstrap_child() -> _rehydrate_adapter()
    # So we can just fetch it from the registry
    from .adapter_registry import AdapterRegistry

    adapter: IsolationAdapter | None = AdapterRegistry.get()

    # Register serializers in child process
    if adapter:
        from .serialization_registry import SerializerRegistry

        adapter.register_serializers(SerializerRegistry.get_instance())

    with context:
        rpc.register_callee(extension, "extension")

        # Register APIs from config
        apis = config.get("apis", [])
        resolved_apis = []

        # Resolve string references back to classes if needed
        for api_item in apis:
            if isinstance(api_item, str):
                try:
                    import importlib

                    parts = api_item.rsplit(".", 1)
                    if len(parts) == 2:
                        mod = importlib.import_module(parts[0])
                        resolved_apis.append(getattr(mod, parts[1]))
                    else:
                        logger.warning("Invalid API reference format: %s", api_item)
                except Exception as e:
                    logger.warning("Failed to resolve API %s: %s", api_item, e)
            else:
                resolved_apis.append(api_item)

        for api in resolved_apis:
            api.use_remote(rpc)
            if adapter:
                api_instance = cast(ProxiedSingleton, getattr(api, "instance", api))
                logger.debug("Calling handle_api_registration for %s", api_instance.__class__.__name__)
                adapter.handle_api_registration(api_instance, rpc)

        # Import and load the extension module
        import importlib.util

        sys_module_name = os.path.basename(module_path).replace("-", "_").replace(".", "_")
        module_spec = importlib.util.spec_from_file_location(
            sys_module_name, os.path.join(module_path, "__init__.py")
        )

        assert module_spec is not None
        assert module_spec.loader is not None

        rpc.run()

        try:
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[sys_module_name] = module
            module_spec.loader.exec_module(module)
            await extension.on_module_loaded(module)
            await rpc.run_until_stopped()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            # Keep RPC alive so the host can gracefully skip broken extensions
            # instead of seeing a connection-reset hard failure.
            logger.warning("Extension module loading/execution failed for %s: %s", module_path, exc)
            await rpc.run_until_stopped()


if __name__ == "__main__":
    main()
