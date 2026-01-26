import contextlib
import logging
import os
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Generic, TypeVar

from ..config import ExtensionConfig
from ..shared import ExtensionBase
from .client import entrypoint
from .shared import AsyncRPC

logger = logging.getLogger(__name__)


def normalize_extension_name(name: str) -> str:
    """
    Normalize an extension name to be safe for use in filesystem paths and shell commands.

    This function:
    - Replaces spaces and unsafe characters with underscores
    - Removes directory traversal attempts
    - Ensures the name is not empty
    - Preserves Unicode characters (for non-English names)

    Args:
        name: The original extension name

    Returns:
        A normalized, filesystem-safe version of the name

    Raises:
        ValueError: If the name is empty or only contains invalid characters
    """
    if not name:
        raise ValueError("Extension name cannot be empty")

    # Remove any directory traversal attempts or absolute path indicators
    # Replace path separators with underscores
    name = name.replace("/", "_").replace("\\", "_")

    # Remove leading dots to prevent hidden files
    while name.startswith("."):
        name = name[1:]

    # Replace consecutive dots that are part of directory traversal
    name = name.replace("..", "_")

    # Replace problematic characters with underscores
    # This includes spaces, shell metacharacters, and control characters
    # But preserves Unicode letters, numbers, and some safe punctuation
    unsafe_chars = [
        " ",  # Spaces
        "\t",  # Tabs
        "\n",  # Newlines
        "\r",  # Carriage returns
        ";",  # Command separator
        "|",  # Pipe
        "&",  # Background/and
        "$",  # Variable expansion
        "`",  # Command substitution
        "(",  # Subshell
        ")",  # Subshell
        "<",  # Redirect
        ">",  # Redirect
        '"',  # Quote
        "'",  # Quote
        "\\",  # Escape (already handled above)
        "!",  # History expansion
        "{",  # Brace expansion
        "}",  # Brace expansion
        "[",  # Glob
        "]",  # Glob
        "*",  # Glob
        "?",  # Glob
        "~",  # Home directory
        "#",  # Comment
        "%",  # Job control
        "=",  # Assignment
        ":",  # Path separator
        ",",  # Various uses
        "\0",  # Null byte
    ]

    for char in unsafe_chars:
        name = name.replace(char, "_")

    # Replace multiple consecutive underscores with a single underscore
    name = re.sub(r"_+", "_", name)

    # Remove leading and trailing underscores
    name = name.strip("_")

    # If the name is now empty (was all invalid chars), raise an error
    if not name:
        raise ValueError("Extension name contains only invalid characters")

    return name


def validate_dependency(dep: str) -> None:
    """Validate a single dependency specification."""
    if not dep:
        return

    # Special case: allow "-e" for editable installs followed by a path
    if dep == "-e":
        # This is OK, it should be followed by a path in the next argument
        return

    # Check if it looks like a command-line option (but allow -e)
    if dep.startswith("-") and not dep.startswith("-e "):
        raise ValueError(
            f"Invalid dependency '{dep}'. "
            "Dependencies cannot start with '-' as this could be a command option."
        )

    # Basic validation for common injection patterns
    # Note: We allow < and > as they're used in version specifiers
    dangerous_patterns = ["&&", "||", ";", "|", "`", "$", "\n", "\r", "\0"]
    for pattern in dangerous_patterns:
        if pattern in dep:
            raise ValueError(
                f"Invalid dependency '{dep}'. Contains potentially dangerous character: '{pattern}'"
            )


def validate_path_within_root(path: Path, root: Path) -> None:
    """Ensure a path is within the expected root directory."""
    try:
        # Resolve both paths to absolute paths
        resolved_path = path.resolve()
        resolved_root = root.resolve()

        # Check if the path is within the root
        resolved_path.relative_to(resolved_root)
    except ValueError as err:
        raise ValueError(f"Path '{path}' is not within the expected root directory '{root}'") from err


@contextmanager
def environment(**env_vars):
    """Context manager for temporarily setting environment variables"""
    original = {}

    # Save original values and set new ones
    for key, value in env_vars.items():
        original[key] = os.environ.get(key)
        os.environ[key] = str(value)

    try:
        yield
    finally:
        # Restore original values
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


T = TypeVar("T", bound=ExtensionBase)


class Extension(Generic[T]):
    def __init__(
        self,
        module_path: str,
        extension_type: type[T],
        config: ExtensionConfig,
        venv_root_path: str,
    ) -> None:
        # Store original name for display purposes
        self.name = config["name"]

        # Normalize the name for filesystem operations
        self.normalized_name = normalize_extension_name(self.name)

        # Log if normalization changed the name
        if self.normalized_name != self.name:
            logger.debug(
                f"Extension name '{self.name}' normalized to '{self.normalized_name}' "
                "for filesystem compatibility"
            )

        # Validate all dependencies
        for dep in config["dependencies"]:
            validate_dependency(dep)

        # Use Path for safer path operations with normalized name
        venv_root = Path(venv_root_path).resolve()
        self.venv_path = venv_root / self.normalized_name

        # Ensure the venv path is within the root directory
        validate_path_within_root(self.venv_path, venv_root)

        self.module_path = module_path
        self.config = config
        self.extension_type = extension_type

        if self.config["share_torch"]:
            import torch.multiprocessing

            self.mp = torch.multiprocessing
        else:
            import multiprocessing

            self.mp = multiprocessing

        start_method = self.mp.get_start_method(allow_none=True)
        if start_method is None:
            self.mp.set_start_method("spawn")
        elif start_method != "spawn":
            raise RuntimeError(
                f"Invalid start method {start_method} for pyisolate. "
                "Pyisolate requires the 'spawn' start method to work correctly."
            )
        self.to_extension = self.mp.Queue()
        self.from_extension = self.mp.Queue()
        self.extension_proxy = None
        self.proc = self.__launch()
        self.rpc = AsyncRPC(recv_queue=self.from_extension, send_queue=self.to_extension)
        for api in config["apis"]:
            api()._register(self.rpc)
        self.rpc.run()

    def get_proxy(self) -> T:
        if self.extension_proxy is None:
            self.extension_proxy = self.rpc.create_caller(self.extension_type, "extension")

        return self.extension_proxy

    def stop(self) -> None:
        """Stop the extension process and clean up resources."""
        try:
            # Terminate the process first to prevent further issues
            if hasattr(self, "proc") and self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=5.0)

                # Force kill if still alive
                if self.proc.is_alive():
                    logger.warning(f"Extension {self.name} did not terminate gracefully, force killing")
                    self.proc.kill()
                    self.proc.join()

            # Clean up queues
            if hasattr(self, "to_extension"):
                with contextlib.suppress(Exception):
                    self.to_extension.close()
            if hasattr(self, "from_extension"):
                with contextlib.suppress(Exception):
                    self.from_extension.close()

        except Exception as e:
            logger.error(f"Error stopping extension {self.name}: {e}")

    def __launch(self):
        """
        Launch the extension in a separate process.
        """
        # Create the virtual environment for the extension
        self._create_extension_venv()

        # Install dependencies in the virtual environment
        self._install_dependencies()

        # Set the Python executable from the virtual environment
        executable = sys._base_executable if os.name == "nt" else str(self.venv_path / "bin" / "python")  # type: ignore
        logger.debug(f"Launching extension {self.name} with Python executable: {executable}")
        self.mp.set_executable(executable)
        context = nullcontext()
        if os.name == "nt":
            # On Windows, we need to set the environment variables for the subprocess
            context = environment(
                VIRTUAL_ENV=str(self.venv_path),
            )
        with context:
            proc = self.mp.Process(
                target=entrypoint,
                args=(
                    self.module_path,
                    self.extension_type,
                    self.config,
                    self.to_extension,
                    self.from_extension,
                ),
            )
            proc.start()
        return proc

    def _create_extension_venv(self):
        """
        Create a virtual environment for the extension if it doesn't exist.
        """
        # Ensure parent directory exists
        self.venv_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.venv_path.exists():
            logger.debug(f"Creating virtual environment for extension {self.name} at {self.venv_path}")

            # Find uv executable path for better security
            uv_path = shutil.which("uv")
            if not uv_path:
                raise RuntimeError("uv command not found in PATH")

            # Use the resolved, validated path
            import sys
            py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
            subprocess.check_call([uv_path, "venv", str(self.venv_path), "--python", py_version])  # noqa: S603

    # TODO(Optimization): Only do this when we update a extension to reduce startup time?
    def _install_dependencies(self):
        """
        Install dependencies in the extension's virtual environment.
        """
        if os.name == "nt":
            python_executable = self.venv_path / "Scripts" / "python.exe"
        else:
            python_executable = self.venv_path / "bin" / "python"

        # Ensure the Python executable exists
        if not python_executable.exists():
            raise RuntimeError(f"Python executable not found at {python_executable}")

        # Find uv executable path for better security
        uv_path = shutil.which("uv")
        if not uv_path:
            raise RuntimeError("uv command not found in PATH")

        uv_args = [uv_path, "pip", "install", "--python", str(python_executable)]

        uv_common_args = []

        # Set up a local cache directory next to venvs to ensure same filesystem
        # This enables hardlinking and saves disk space
        cache_dir = self.venv_path.parent / ".uv_cache"
        cache_dir.mkdir(exist_ok=True)
        uv_common_args.extend(["--cache-dir", str(cache_dir)])

        # Detect Intel/XPU backend for special index URL
        use_xpu_backend = False
        backend_env = os.environ.get("PYISOLATE_BACKEND", "auto").lower()
        if backend_env == "xpu" or self.config.get("backend") == "xpu":
            use_xpu_backend = True
        # Also check for Intel GPU in device name if available
        if not use_xpu_backend and "intel" in str(self.config.get("device_name", "")).lower():
            use_xpu_backend = True

        # Install the same version of torch as the current process, if needed
        torch_requirement = None
        torch_index_args = []
        if self.config["share_torch"]:
            import torch

            torch_version = torch.__version__
            # Remove any '+cpu', '+xpu', or other local version tags
            if "+" in torch_version:
                torch_version = torch_version.split("+")[0]
            cuda_version = getattr(torch.version, "cuda", None)  # type: ignore
            if use_xpu_backend:
                torch_requirement = "torch>=2.7.0"
                torch_index_args = ["--index-url", "https://download.pytorch.org/whl/xpu"]
            elif cuda_version:
                torch_requirement = f"torch=={torch_version}"
                torch_index_args = [
                    "--extra-index-url",
                    f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}",
                ]
            else:
                torch_requirement = f"torch=={torch_version}"

        # Install extension dependencies from config
        if self.config["dependencies"] or self.config["share_torch"]:
            logger.debug(f"Installing extension dependencies for {self.name}...")

            # Re-validate dependencies before passing to subprocess (defense in depth)
            safe_dependencies = []
            torch_in_deps = False
            for dep in self.config["dependencies"]:
                validate_dependency(dep)
                # Remove any '+xpu' or '+cpu' from torch dependencies for Intel/XPU
                if use_xpu_backend and dep.startswith("torch"):
                    dep = "torch>=2.7.0"
                if dep.startswith("torch"):
                    torch_in_deps = True
                safe_dependencies.append(dep)

            # Only add torch requirement if not already present
            if torch_requirement and not torch_in_deps:
                safe_dependencies.insert(0, torch_requirement)
                uv_args += torch_index_args

            # In normal mode, suppress output unless there are actual changes
            always_output = logger.isEnabledFor(logging.DEBUG)
            try:
                result = subprocess.run(  # noqa: S603
                    uv_args + safe_dependencies + uv_common_args,
                    capture_output=not always_output,
                    text=True,
                    check=True,
                )
                # Only show output if there were actual changes (installations/updates)
                if (
                    not always_output
                    and result.stderr
                    and ("Installed" in result.stderr or "Uninstalled" in result.stderr)
                ):
                    logger.info(f"Dependencies updated for {self.name}:\n{result.stderr.strip()}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install dependencies for {self.name}: {e}")
                if e.stderr:
                    logger.error(f"Error details: {e.stderr}")
                raise
        else:
            logger.debug(f"No dependencies to install for {self.name}")

    def join(self):
        """
        Wait for the extension process to finish.
        """
        self.proc.join()
