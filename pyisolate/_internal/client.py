"""PyIsolate child process entrypoint and path unification.

Imported by isolated child processes during ``multiprocessing.spawn``. The
module-level path setup must execute before any heavy imports so the child sees
the preferred host root ahead of the isolated venv site-packages. Environment
variables used here:

- ``PYISOLATE_CHILD``: Indicates this interpreter is an isolated child process.
- ``PYISOLATE_HOST_SNAPSHOT``: JSON snapshot containing host ``sys.path`` and env vars.
- ``PYISOLATE_MODULE_PATH``: Path to the extension being loaded (used to detect a preferred root).
- ``PYISOLATE_PATH_DEBUG``: Enables verbose sys.path logging when set.
"""

import asyncio
import importlib.util
import logging
import os
import sys
from contextlib import AbstractContextManager as ContextManager
from contextlib import nullcontext
from logging.handlers import QueueHandler
from typing import Any, cast

from ..config import ExtensionConfig
from ..interfaces import IsolationAdapter
from ..shared import ExtensionBase
from .bootstrap import bootstrap_child
from .rpc_protocol import AsyncRPC, ProxiedSingleton, set_child_rpc_instance

logger = logging.getLogger(__name__)

_adapter: IsolationAdapter | None = None
_bootstrap_done = False


def _ensure_bootstrap() -> None:
    """Bootstrap the child environment on first call.

    Deferred to avoid circular imports during module initialization.
    The adapter loads host application modules which try to import pyisolate,
    but pyisolate's __init__ might not be fully initialized yet.
    """
    global _adapter, _bootstrap_done
    if _bootstrap_done:
        return
    _bootstrap_done = True

    if os.environ.get("PYISOLATE_CHILD"):
        _adapter = bootstrap_child()


async def async_entrypoint(
    module_path: str,
    extension_type: type[ExtensionBase],
    config: ExtensionConfig,
    to_extension: Any,
    from_extension: Any,
    log_queue: Any,
) -> None:
    """Asynchronous entrypoint for isolated extension processes.

    Sets up the RPC channel, registers proxies for shared singletons, imports the
    extension module, and runs lifecycle hooks inside the isolated process.

    Args:
        module_path: Absolute path to the extension module directory.
        extension_type: ``ExtensionBase`` subclass to instantiate.
        config: Extension configuration (dependencies, APIs, share_torch, etc.).
        to_extension: Queue carrying host → extension RPC messages.
        from_extension: Queue carrying extension → host RPC messages.
        log_queue: Optional queue for forwarding child logs to the host.
    """
    # Deferred bootstrap to avoid circular imports
    _ensure_bootstrap()

    if os.environ.get("PYISOLATE_CHILD") and log_queue is not None:
        root = logging.getLogger()
        root.addHandler(QueueHandler(log_queue))
        root.setLevel(logging.INFO)

    rpc = AsyncRPC(recv_queue=to_extension, send_queue=from_extension)
    set_child_rpc_instance(rpc)

    extension = extension_type()
    extension._initialize_rpc(rpc)

    try:
        await extension.before_module_loaded()
    except Exception as exc:  # pragma: no cover - fail loud path
        logger.error("Extension before_module_loaded failed: %s", exc, exc_info=True)
        raise

    context: ContextManager[Any] = nullcontext()
    if config["share_torch"]:
        import torch

        context = cast(ContextManager[Any], torch.inference_mode())

    if not os.path.isdir(module_path):
        raise ValueError(f"Module path {module_path} is not a directory.")

    with context:
        rpc.register_callee(extension, "extension")
        for api in config["apis"]:
            api.use_remote(rpc)
            if _adapter:
                api_instance = cast(ProxiedSingleton, getattr(api, "instance", api))
                _adapter.handle_api_registration(api_instance, rpc)

        # Let the adapter wire child-side event hooks (e.g., progress bar)
        if _adapter and hasattr(_adapter, "setup_child_event_hooks"):
            _adapter.setup_child_event_hooks(extension)

        # Sanitize module name for use as Python identifier.
        # Replace '-' and '.' with '_' to prevent import errors when module names contain
        # non-identifier characters (e.g., "my-node" → "my_node", "my.node" → "my_node").
        # Required because we dynamically import modules by name and Python identifiers
        # cannot contain hyphens or dots outside of attribute access.
        sys_module_name = os.path.basename(module_path).replace("-", "_").replace(".", "_")
        module_spec = importlib.util.spec_from_file_location(
            sys_module_name, os.path.join(module_path, "__init__.py")
        )

        assert module_spec is not None
        assert module_spec.loader is not None

        try:
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[sys_module_name] = module
            module_spec.loader.exec_module(module)

            rpc.run()
            await extension.on_module_loaded(module)
            await rpc.run_until_stopped()
        except Exception as exc:  # pragma: no cover - fail loud path
            logger.error(
                "Extension module loading/execution failed for %s: %s", module_path, exc, exc_info=True
            )
            raise


def entrypoint(
    module_path: str,
    extension_type: type[ExtensionBase],
    config: ExtensionConfig,
    to_extension: Any,
    from_extension: Any,
    log_queue: Any,
) -> None:
    """Synchronous wrapper around :func:`async_entrypoint`.

    This is invoked by ``multiprocessing.Process`` and simply drives the async
    entrypoint inside a fresh asyncio event loop.
    """
    asyncio.run(
        async_entrypoint(
            module_path,
            extension_type,
            config,
            to_extension,
            from_extension,
            log_queue,
        )
    )
