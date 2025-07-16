import asyncio
import importlib.util
import logging
import sys
from contextlib import nullcontext

from ..config import ExtensionConfig
from ..shared import ExtensionBase
from .shared import AsyncRPC

logger = logging.getLogger(__name__)


async def async_entrypoint(
    module_path: str,
    extension_type: type[ExtensionBase],
    config: ExtensionConfig,
    to_extension,
    from_extension,
) -> None:
    """
    Asynchronous entrypoint for the module.
    """
    logger.debug("Loading extension with Python executable: %s", sys.executable)
    logger.debug("Loading extension from: %s", module_path)

    # Robustly ensure only the venv's site-packages are present in sys.path
    import os
    import site

    venv_prefix = sys.prefix
    venv_site_packages = [p for p in site.getsitepackages() if p.startswith(venv_prefix)]
    # Remove all site-packages not in the current venv
    sys.path = [p for p in sys.path if not (("site-packages" in p) and (not p.startswith(venv_prefix)))]
    # Prepend all venv site-packages to sys.path (in order)
    for p in reversed(venv_site_packages):
        if p not in sys.path:
            sys.path.insert(0, p)

    rpc = AsyncRPC(recv_queue=to_extension, send_queue=from_extension)
    extension = extension_type()
    extension._initialize_rpc(rpc)
    await extension.before_module_loaded()

    context = nullcontext()
    if config["share_torch"]:
        import torch

        context = torch.inference_mode()

    if not os.path.isdir(module_path):
        raise ValueError(f"Module path {module_path} is not a directory.")

    with context:
        try:
            rpc.register_callee(extension, "extension")
            for api in config["apis"]:
                api.use_remote(rpc)

            # If it's a directory, load the __init__.py file
            sys_module_name = module_path.replace(".", "_x_")  # Replace dots to avoid conflicts
            module_spec = importlib.util.spec_from_file_location(
                sys_module_name, os.path.join(module_path, "__init__.py")
            )

            assert module_spec is not None, f"Module spec for {module_path} is None"
            assert module_spec.loader is not None, f"Module loader for {module_path} is None"

            # Create the module and execute it
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[sys_module_name] = module

            module_spec.loader.exec_module(module)

            # Start processing RPC in case the module uses it during loading
            rpc.run()
            try:
                await extension.on_module_loaded(module)
            except Exception as e:
                import traceback

                logger.error("Error in on_module_loaded for %s: %s", module_path, e)
                logger.error("Exception details:\n%s", traceback.format_exc())
                await rpc.stop()
                return

            await rpc.run_until_stopped()

        except Exception as e:
            import traceback

            logger.error("Error loading extension from %s: %s", module_path, e)
            logger.error("Exception details:\n%s", traceback.format_exc())


def entrypoint(
    module_path: str,
    extension_type: type[ExtensionBase],
    config: ExtensionConfig,
    to_extension,
    from_extension,
) -> None:
    asyncio.run(async_entrypoint(module_path, extension_type, config, to_extension, from_extension))
