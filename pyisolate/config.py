from __future__ import annotations

import sys
from enum import Enum
from typing import TYPE_CHECKING, Any, TypedDict

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired

if TYPE_CHECKING:
    from ._internal.rpc_protocol import ProxiedSingleton


class SandboxMode(Enum):
    """Sandbox enforcement mode for Linux process isolation.

    REQUIRED: (Default) Fail loudly if bubblewrap is unavailable. This is the
              only safe option for running untrusted code.
    DISABLED: Skip sandbox entirely. USE AT YOUR OWN RISK. This exposes your
              filesystem, network, and GPU memory to untrusted extensions.
    """

    REQUIRED = "required"
    DISABLED = "disabled"


class ExtensionManagerConfig(TypedDict):
    """Configuration for the :class:`ExtensionManager`.

    Controls where isolated virtual environments are created for extensions.
    """

    venv_root_path: str
    """Root directory where isolated venvs will be created (one subdir per extension)."""


class SandboxConfig(TypedDict, total=False):
    writable_paths: list[str]
    readonly_paths: list[str] | dict[str, str]  # Supports src:dst mapping
    network: bool


class CUDAWheelConfig(TypedDict):
    """Configuration for custom CUDA wheel resolution."""

    index_url: str
    """Base URL containing per-package simple index directories."""

    packages: list[str]
    """Canonicalized dependency names that must resolve via the custom index."""

    package_map: NotRequired[dict[str, str]]
    """Optional canonical dependency-name to index-package-name overrides."""


class ExtensionConfig(TypedDict):
    """Configuration for a single extension managed by PyIsolate."""

    name: str
    """Unique name for the extension (used for venv directory naming)."""

    module_path: str
    """Filesystem path to the extension package containing ``__init__.py``."""

    isolated: bool
    """Whether to run the extension in an isolated venv versus the host process."""

    dependencies: list[str]
    """List of pip requirement specifiers to install into the extension venv."""

    apis: list[type[ProxiedSingleton]]
    """ProxiedSingleton classes exposed to this extension for shared services."""

    share_torch: bool
    """If True, reuse host torch via torch.multiprocessing and zero-copy tensors."""

    share_cuda_ipc: bool
    """If True, attempt CUDA IPC-based tensor transport (Linux only, requires ``share_torch``)."""

    sandbox: dict[str, Any]
    """Configuration for the sandbox (e.g. writable_paths, network access)."""

    sandbox_mode: SandboxMode
    """Sandbox enforcement mode. Default is REQUIRED (fail if bwrap unavailable).
    Set to DISABLED only if you fully trust all code and accept the security risk."""

    env: dict[str, str]
    """Environment variable overrides for the child process."""

    cuda_wheels: NotRequired[CUDAWheelConfig]
    """Optional custom CUDA wheel resolution configuration for selected dependencies."""
