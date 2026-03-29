import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .sandbox_detect import RestrictionModel

if TYPE_CHECKING:
    from ..interfaces import IsolationAdapter

logger = logging.getLogger(__name__)

# Paths that would significantly weaken sandbox security if exposed
FORBIDDEN_ADAPTER_PATHS: frozenset[str] = frozenset(
    {"/", "/etc", "/root", "/home", "/var", "/run", "/proc", "/sys"}
)

# ---------------------------------------------------------------------------
# Sandbox System Path Allow-List (DENY-BY-DEFAULT)
# ---------------------------------------------------------------------------
# These are the ONLY system paths exposed to sandboxed processes.
# Everything else is denied. This is a security-critical list.

SANDBOX_SYSTEM_PATHS: list[str] = [
    "/usr",  # System binaries and libraries
    "/lib",  # Core libraries
    "/lib64",  # 64-bit libraries (if exists)
    "/lib32",  # 32-bit libraries (if exists)
    "/bin",  # Essential binaries
    "/sbin",  # System binaries
    "/opt",  # Hosted toolcache interpreters (e.g., GitHub Actions setup-python)
    "/etc/alternatives",  # Symlink management
    "/etc/ld.so.cache",  # Dynamic linker cache
    "/etc/ld.so.conf",  # Dynamic linker config
    "/etc/ld.so.conf.d",  # Dynamic linker config dir
    "/etc/ssl",  # SSL certificates
    "/etc/ca-certificates",  # CA certificates
    "/etc/pki",  # PKI certificates (RHEL/CentOS)
    "/etc/resolv.conf",  # DNS (if network enabled)
    "/etc/hosts",  # Host resolution
    "/etc/nsswitch.conf",  # Name service switch config
    "/etc/passwd",  # User info (read-only, needed for getpwuid)
    "/etc/group",  # Group info
    "/etc/localtime",  # Timezone
    "/etc/timezone",  # Timezone name
]

# GPU device paths for CUDA passthrough
GPU_PASSTHROUGH_PATTERNS: list[str] = [
    "nvidia*",  # GPU devices
    "nvidiactl",  # Control device
    "nvidia-uvm",  # Unified memory
    "nvidia-uvm-tools",  # UVM tools
    "dri",  # Direct Rendering Infrastructure
]

FORBIDDEN_WRITABLE_BIND_PATHS: frozenset[str] = frozenset({"/tmp"})


def _linuxbrew_root(path: Path) -> str | None:
    """Return the Linuxbrew root for a path under ``.../.linuxbrew/...``."""
    parts = path.parts
    if ".linuxbrew" not in parts:
        return None
    idx = parts.index(".linuxbrew")
    return str(Path(*parts[: idx + 1]))


def _validate_adapter_path(path: str) -> bool:
    """Validate that an adapter-provided path doesn't weaken sandbox security.

    Args:
        path: Path to validate

    Returns:
        True if path is safe to add, False if it would weaken sandbox
    """
    # Normalize the path
    normalized = os.path.normpath(path)

    # Check against forbidden paths
    if normalized in FORBIDDEN_ADAPTER_PATHS:
        return False

    # Check if path is a parent of forbidden paths (e.g., "/" is parent of all)
    for forbidden in FORBIDDEN_ADAPTER_PATHS:
        if forbidden.startswith(normalized + "/") or normalized == forbidden:
            return False

    return True


def build_bwrap_command(
    python_exe: str,
    module_path: str,
    venv_path: str,
    uds_address: str,
    allow_gpu: bool = False,
    sandbox_config: dict[str, Any] | None = None,
    restriction_model: RestrictionModel = RestrictionModel.NONE,
    env_overrides: dict[str, str] | None = None,
    adapter: "IsolationAdapter | None" = None,
    execution_model: str = "host-coupled",
    sealed_host_ro_paths: list[str] | None = None,
) -> list[str]:
    """Build the bubblewrap command for launching a sandboxed process.

    Security properties:
    - DENY-BY-DEFAULT filesystem (explicit allow-list only)
    - Venv is READ-ONLY (prevents persistent infection)
    - User namespace isolation (unprivileged execution)
    - PID namespace isolation (process isolation)
    - Network isolated by default
    - /dev/shm shared (required for CUDA IPC, documented risk)

    Args:
        python_exe: Path to the Python interpreter in the venv
        module_path: Path to the extension module directory
        venv_path: Path to the isolated venv
        uds_address: Path to the Unix socket for IPC
        sandbox_config: SandboxConfig dict with network, writable_paths, readonly_paths
        allow_gpu: Whether to enable GPU passthrough
        restriction_model: Detected system restriction model
        env_overrides: Additional environment variables to set
        adapter: Optional IsolationAdapter for extending sandbox paths

    Returns:
        Command list suitable for subprocess.Popen
    """
    if sandbox_config is None:
        sandbox_config = {}

    cmd = ["bwrap"]
    is_sealed_worker = execution_model == "sealed_worker"

    # Namespace Isolation Logic
    # -------------------------
    # We attempt full user namespace isolation (--unshare-user) if possible.
    # This allows us to map UIDs and isolate the process fully.
    # However, modern distros often restrict this (AppArmor, sysctl).

    if restriction_model == RestrictionModel.NONE:
        # Full isolation available
        cmd.extend(["--unshare-user", "--unshare-pid"])
        # We do NOT unshare-ipc because CUDA shared memory (legacy) and
        # Python SharedMemory lease require /dev/shm access in the host namespace
        # (or a shared namespace). Since we bind /dev/shm, we keep IPC shared.
    else:
        # Run in degraded mode (no user/pid namespace)
        # We still get filesystem isolation via mount namespace (bwrap default).
        pass

    # New session (detach from terminal)
    cmd.append("--new-session")

    # Ensure child dies when parent dies (prevent zombie processes)
    cmd.append("--die-with-parent")

    # Essential virtual filesystems
    cmd.extend(["--proc", "/proc"])
    cmd.extend(["--dev", "/dev"])
    cmd.extend(["--tmpfs", "/tmp"])

    # DENY-BY-DEFAULT: Only bind required system paths (read-only)
    # Start with default paths
    system_paths = list(SANDBOX_SYSTEM_PATHS)

    # Always explicitly bind the python base prefix to ensure the interpreter
    # and its lib dependencies are accessible if installed in non-standard locations (e.g. linuxbrew)
    if sys.base_prefix not in system_paths:
        system_paths.append(sys.base_prefix)

    # uv-created venv interpreters can be symlinks into a non-standard
    # installation prefix (for example Homebrew/Linuxbrew cellar paths).
    # Bind that resolved prefix too, or bwrap will fail to exec the venv's
    # python binary even though the venv itself is mounted.
    python_path = Path(python_exe)
    try:
        if python_path.is_symlink():
            raw_link_target = os.readlink(python_exe)
            raw_link_path = Path(raw_link_target)
            if not raw_link_path.is_absolute():
                raw_link_path = (python_path.parent / raw_link_path).resolve()
            raw_python_prefix = str(raw_link_path.parent.parent)
            if raw_python_prefix not in system_paths:
                system_paths.append(raw_python_prefix)
            raw_linuxbrew_root = _linuxbrew_root(raw_link_path)
            if raw_linuxbrew_root and raw_linuxbrew_root not in system_paths:
                system_paths.append(raw_linuxbrew_root)
    except OSError:
        pass

    try:
        resolved_python_path = python_path.resolve()
        resolved_python_prefix = str(resolved_python_path.parent.parent)
        if resolved_python_prefix not in system_paths:
            system_paths.append(resolved_python_prefix)
        resolved_linuxbrew_root = _linuxbrew_root(resolved_python_path)
        if resolved_linuxbrew_root and resolved_linuxbrew_root not in system_paths:
            system_paths.append(resolved_linuxbrew_root)
    except OSError:
        pass

    if not is_sealed_worker:
        # Query adapter for additional paths only for host-coupled execution.
        get_adapter_paths = getattr(adapter, "get_sandbox_system_paths", lambda: None)
        adapter_paths = get_adapter_paths()
        if adapter_paths:
            for path in adapter_paths:
                if _validate_adapter_path(path):
                    system_paths.append(path)
                else:
                    logger.warning("Adapter path '%s' rejected: would weaken sandbox security", path)

    for sys_path in system_paths:
        if os.path.exists(sys_path):
            cmd.extend(["--ro-bind", sys_path, sys_path])

    # Venv: READ-ONLY (prevent malicious modification)
    cmd.extend(["--ro-bind", str(venv_path), str(venv_path)])

    # Module path: READ-ONLY
    cmd.extend(["--ro-bind", str(module_path), str(module_path)])

    if is_sealed_worker and sealed_host_ro_paths:
        for ro_path in sealed_host_ro_paths:
            normalized = os.path.normpath(ro_path)
            if os.path.exists(normalized):
                cmd.extend(["--ro-bind", normalized, normalized])

    # GPU passthrough (if enabled)
    if allow_gpu:
        cmd.extend(["--ro-bind", "/sys", "/sys"])
        dev_path = Path("/dev")

        # Start with default GPU patterns
        gpu_patterns = list(GPU_PASSTHROUGH_PATTERNS)

        if not is_sealed_worker:
            # Query adapter for additional GPU patterns only for host-coupled execution.
            get_gpu_patterns = getattr(adapter, "get_sandbox_gpu_patterns", lambda: None)
            adapter_gpu_patterns = get_gpu_patterns()
            if adapter_gpu_patterns:
                gpu_patterns.extend(adapter_gpu_patterns)

        for pattern in gpu_patterns:
            for dev in dev_path.glob(pattern):
                if dev.exists():
                    cmd.extend(["--dev-bind", str(dev), str(dev)])
        # CUDA IPC requires shared memory
        # SECURITY: /dev/shm is shared. This is a known side-channel risk
        # but unavoidable for zero-copy tensor transfer. Document this trade-off.
        # MOVED: /dev/shm binding is now global (see below) because CPU tensors need it too.

        # CUDA library and runtime paths (read-only)
        # /usr/local/cuda is covered by /usr bind, so we skip it to avoid symlink/mount issues
        cuda_paths = {
            "/opt/cuda",  # Alternative CUDA location
            "/run/nvidia-persistenced",  # Persistence daemon
        }

        # Add CUDA_HOME if set and not in /usr (redundant)
        cuda_home = os.environ.get("CUDA_HOME")
        if cuda_home:
            cuda_paths.add(cuda_home)

        for cuda_path in cuda_paths:
            if os.path.exists(cuda_path):
                # Skip if already covered by /usr bind
                if cuda_path.startswith("/usr/") and not cuda_path.startswith("/usr/local/"):
                    # Actually /usr/local is in /usr.
                    # Safe heuristic: if it starts with /usr, we assume covered.
                    continue
                cmd.extend(["--ro-bind", cuda_path, cuda_path])

    # Network Isolation
    # Default: ISOLATED (--unshare-net) unless explicitly allowed in config
    allow_network = sandbox_config.get("network", False)
    if not allow_network:
        cmd.append("--unshare-net")
    # If allow_network is True, we simply don't unshare, inheriting host network.

    # MOVED: path bindings moved to end to prevent masking by RO binds

    pyisolate_path: Path | None = None
    if not is_sealed_worker:
        # 1. Host venv site-packages: READ-ONLY (for share_torch inheritance via .pth file)
        # The child venv has a .pth file pointing to host site-packages for torch sharing
        # We find where 'torch' is likely installed (host site-packages)
        host_site_packages = Path(sys.executable).parent.parent / "lib"
        for sp in host_site_packages.glob("python*/site-packages"):
            if sp.exists():
                cmd.extend(["--ro-bind", str(sp), str(sp)])
                break

        # 2. PyIsolate package path: READ-ONLY (needed for sandbox_client/uds_client)
        import pyisolate as pyisolate_pkg

        pyisolate_path = Path(pyisolate_pkg.__file__).parent.parent.resolve()
        cmd.extend(["--ro-bind", str(pyisolate_path), str(pyisolate_path)])

        # Application paths (e.g., ComfyUI) are provided by the adapter via
        # get_sandbox_system_paths() at lines 161-166. No direct framework imports.

    # Shared memory is only needed for host-coupled shared-memory tensor transport.
    if not is_sealed_worker and Path("/dev/shm").exists():
        cmd.extend(["--bind", "/dev/shm", "/dev/shm"])

    # UDS socket directory must be accessible
    uds_dir = os.path.dirname(uds_address)
    if uds_dir:
        # Create parent directories for UDS mount point to ensure they exist in tmpfs structure
        parts = Path(uds_dir).parts
        current = Path("/")
        for part in parts[1:]:
            current = current / part
            cmd.extend(["--dir", str(current)])

    if uds_dir and os.path.exists(uds_dir):
        cmd.extend(["--bind", uds_dir, uds_dir])

    # ---------------------------------------------------------------------------
    # CONFIG OVERRIDES (Must happen LAST to override default RO binds)
    # ---------------------------------------------------------------------------

    # 1. Writable paths from config (user-specified)
    # Placed here so they can punch holes in RO binds (e.g. ComfyUI/temp inside RO ComfyUI)
    for path in sandbox_config.get("writable_paths", []):
        normalized_path = os.path.normpath(path)
        if normalized_path in FORBIDDEN_WRITABLE_BIND_PATHS:
            logger.warning("Skipping forbidden writable sandbox path: %s", normalized_path)
            continue
        if os.path.exists(path):
            cmd.extend(["--bind", path, path])

    # 2. Read-only paths from config
    ro_paths = sandbox_config.get("readonly_paths", [])
    if isinstance(ro_paths, list):
        for path in ro_paths:
            if os.path.exists(path):
                cmd.extend(["--ro-bind", path, path])
    elif isinstance(ro_paths, dict):
        for src, dst in ro_paths.items():
            if os.path.exists(src):
                cmd.extend(["--ro-bind", src, dst])

    if is_sealed_worker:
        cmd.append("--clearenv")

    # Environment variables
    cmd.extend(["--setenv", "PYISOLATE_UDS_ADDRESS", uds_address])
    cmd.extend(["--setenv", "PYISOLATE_CHILD", "1"])
    # Propagate PYISOLATE_IMPORT_TORCH from env_overrides if set
    if env_overrides and env_overrides.get("PYISOLATE_IMPORT_TORCH") == "0":
        cmd.extend(["--setenv", "PYISOLATE_IMPORT_TORCH", "0"])

    if is_sealed_worker:
        # Hermetic sealed workers get an explicit env allowlist.
        for env_var in ["PATH", "LANG", "LC_ALL", "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES"]:
            if env_var in os.environ:
                cmd.extend(["--setenv", env_var, os.environ[env_var]])
        if "NVIDIA_VISIBLE_DEVICES" in os.environ:
            cmd.extend(["--setenv", "NVIDIA_VISIBLE_DEVICES", os.environ["NVIDIA_VISIBLE_DEVICES"]])
        if "LD_LIBRARY_PATH" in os.environ:
            cmd.extend(["--setenv", "LD_LIBRARY_PATH", os.environ["LD_LIBRARY_PATH"]])
        if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
            cmd.extend(["--setenv", "PYTORCH_CUDA_ALLOC_CONF", os.environ["PYTORCH_CUDA_ALLOC_CONF"]])
        if "TORCH_CUDA_ARCH_LIST" in os.environ:
            cmd.extend(["--setenv", "TORCH_CUDA_ARCH_LIST", os.environ["TORCH_CUDA_ARCH_LIST"]])
        cmd.extend(["--setenv", "HOME", "/tmp"])
        cmd.extend(["--setenv", "TMPDIR", "/tmp"])
        cmd.extend(["--setenv", "PYTHONNOUSERSITE", "1"])
    else:
        # Set PYTHONPATH to include pyisolate package
        # This ensures the child can find 'pyisolate' even if not installed in its venv
        pyisolate_parent = str(pyisolate_path)
        # Start with our explicitly bound package
        new_pythonpath_parts = [pyisolate_parent]

        # Check existing PYTHONPATH
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        if existing_pythonpath:
            new_pythonpath_parts.append(existing_pythonpath)

        cmd.extend(["--setenv", "PYTHONPATH", ":".join(new_pythonpath_parts)])

        # Inherit select environment variables
        # Standard environment
        for env_var in ["PATH", "HOME", "LANG", "LC_ALL"]:
            if env_var in os.environ:
                cmd.extend(["--setenv", env_var, os.environ[env_var]])

        # CUDA/GPU environment variables (critical for GPU access)
        cuda_env_vars = [
            "CUDA_HOME",
            "CUDA_PATH",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "LD_LIBRARY_PATH",
            "PYTORCH_CUDA_ALLOC_CONF",
            "TORCH_CUDA_ARCH_LIST",
            "PYISOLATE_ENABLE_CUDA_IPC",
            "PYISOLATE_ENABLE_CUDA_IPC",
        ]
        for env_var in cuda_env_vars:
            if env_var in os.environ:
                cmd.extend(["--setenv", env_var, os.environ[env_var]])

        # Coverage / Profiling forwarding
        for key, val in os.environ.items():
            if key.startswith(("COV_", "COVERAGE_")):
                cmd.extend(["--setenv", key, val])

    # Env overrides from config
    if env_overrides:
        for key, val in env_overrides.items():
            cmd.extend(["--setenv", key, val])

    # Command to run (Corrected to uds_client for main branch architecture)
    cmd.extend([python_exe, "-m", "pyisolate._internal.uds_client"])

    return cmd
