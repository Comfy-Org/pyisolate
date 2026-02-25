import hashlib
import json
import logging
import os
import re
import shutil
import site
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from ..config import ExtensionConfig
from ..path_helpers import serialize_host_snapshot
from .torch_utils import get_torch_ecosystem_packages

logger = logging.getLogger(__name__)

_DANGEROUS_PATTERNS = ("&&", "||", "|", "`", "$", "\n", "\r", "\0")
_UNSAFE_CHARS = frozenset(" \t\n\r|&$`()<>\"'\\!{}[]*?~#%=,")


def normalize_extension_name(name: str) -> str:
    """
    Normalize an extension name for filesystem and shell safety.

    Replaces unsafe characters, strips traversal attempts, and ensures a non-empty
    result while preserving Unicode characters.

    Raises:
        ValueError: If the normalized name would be empty.
    """
    if not name:
        raise ValueError("Extension name cannot be empty")

    name = name.replace("/", "_").replace("\\", "_")
    while name.startswith("."):
        name = name[1:]
    name = name.replace("..", "_")

    for char in _UNSAFE_CHARS:
        name = name.replace(char, "_")

    name = re.sub(r"_+", "_", name)
    name = name.strip("_")

    if not name:
        raise ValueError("Extension name contains only invalid characters")
    return name


def validate_dependency(dep: str) -> None:
    """Validate a single dependency specification."""
    if not dep:
        return
    # Allow `-e` flag for editable installs (e.g., `-e /path/to/package` or `-e .`)
    # This enables development workflows where the extension is pip-installed in editable mode
    if dep == "-e":
        return
    if dep.startswith("-") and not dep.startswith("-e "):
        raise ValueError(
            f"Invalid dependency '{dep}'. "
            "Dependencies cannot start with '-' as this could be a command option."
        )
    for pattern in _DANGEROUS_PATTERNS:
        if pattern in dep:
            raise ValueError(
                f"Invalid dependency '{dep}'. Contains potentially dangerous character: '{pattern}'"
            )


def validate_path_within_root(path: Path, root: Path) -> None:
    """Ensure ``path`` is contained within ``root`` to avoid path escape."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as err:
        raise ValueError(f"Path '{path}' is not within root '{root}'") from err


@contextmanager
def environment(**env_vars: Any) -> Iterator[None]:
    """Temporarily set environment variables inside a context."""
    original: dict[str, str | None] = {}
    for key, value in env_vars.items():
        original[key] = os.environ.get(key)
        os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_extension_snapshot(module_path: str) -> dict[str, object]:
    """Construct snapshot payload with adapter metadata for child bootstrap."""
    snapshot: dict[str, object] = serialize_host_snapshot()

    adapter = None
    path_config: dict[str, object] = {}
    try:
        # v1.0: Check registry first
        from .adapter_registry import AdapterRegistry

        adapter = AdapterRegistry.get()
    except Exception as exc:
        logger.warning("Adapter load failed: %s", exc)

    if adapter:
        try:
            path_config = adapter.get_path_config(module_path) or {}
        except Exception as exc:
            logger.warning("Adapter path config failed: %s", exc)

        # Register serializers in host process (needed for RPC serialization)
        try:
            from .serialization_registry import SerializerRegistry

            registry = SerializerRegistry.get_instance()
            adapter.register_serializers(registry)
        except Exception as exc:
            logger.warning("Adapter serializer registration failed: %s", exc)

    # v1.0: Serialize adapter reference for rehydration
    adapter_ref: str | None = None  # noqa: UP045
    if adapter:
        cls = adapter.__class__
        # Constraint: Adapter class must be importable (not defined in __main__ or closure)
        if cls.__module__ == "__main__":
            logger.warning(
                "Adapter class %s is defined in __main__ and cannot be rehydrated in child", cls.__name__
            )
        else:
            adapter_ref = f"{cls.__module__}:{cls.__name__}"

    snapshot.update(
        {
            "adapter_ref": adapter_ref,
            "adapter_name": adapter.identifier if adapter else None,
            "preferred_root": path_config.get("preferred_root"),
            "additional_paths": path_config.get("additional_paths", []),
            "context_data": {"module_path": module_path},
        }
    )
    return snapshot


def _detect_pyisolate_version() -> str:
    try:
        return importlib_metadata.version("pyisolate")
    except Exception:
        return "0.0.0"


pyisolate_version = _detect_pyisolate_version()


def exclude_satisfied_requirements(
    config: ExtensionConfig, requirements: list[str], python_exe: Path
) -> list[str]:
    """Filter requirements to skip packages already satisfied in the venv.

    When ``share_torch`` is enabled, the child venv inherits host site-packages
    via a .pth file. Torch ecosystem packages MUST be byte-identical between
    parent and child for shared memory tensor passing to work correctly.
    Reinstalling could resolve to different versions, breaking the share_torch
    contract. This is a correctness requirement, not a performance optimization.
    """
    from packaging.requirements import Requirement

    try:
        result = subprocess.run(  # noqa: S603  # Trusted: system pip executable
            [str(python_exe), "-m", "pip", "list", "--format", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        # Newer uv versions can create venvs without pip unless seeded.
        # If pip is unavailable, skip filtering and install requested deps.
        if "No module named pip" in (exc.stderr or ""):
            logger.debug("pip unavailable in %s; skipping satisfied-requirement filter", python_exe)
            return requirements
        raise
    installed = {pkg["name"].lower(): pkg["version"] for pkg in json.loads(result.stdout)}
    torch_ecosystem = get_torch_ecosystem_packages()

    filtered = []
    for req_str in requirements:
        req_str_stripped = req_str.strip()
        if req_str_stripped.startswith("-e ") or req_str_stripped == "-e":
            filtered.append(req_str)
            continue
        if req_str_stripped.startswith(("/", "./")):
            filtered.append(req_str)
            continue

        try:
            req = Requirement(req_str)
            pkg_name_lower = req.name.lower()

            # Torch ecosystem packages are inherited when share_torch=True; skip
            # reinstalling them to avoid conflicts and unnecessary downloads.
            if config["share_torch"] and pkg_name_lower in torch_ecosystem:
                continue

            if pkg_name_lower in installed:
                installed_version = installed[pkg_name_lower]
                if not req.specifier or installed_version in req.specifier:
                    continue

            filtered.append(req_str)
        except Exception:
            filtered.append(req_str)

    return filtered


def create_venv(venv_path: Path, config: ExtensionConfig) -> None:
    """Create the virtual environment for this extension using uv."""
    venv_path.parent.mkdir(parents=True, exist_ok=True)

    uv_path = shutil.which("uv")
    if not uv_path:
        raise RuntimeError(
            "uv is required but not found. Install it with: pip install uv\n"
            "See https://github.com/astral-sh/uv for installation options."
        )

    if not venv_path.exists():
        subprocess.check_call(
            [  # noqa: S603  # Trusted: uv venv command
                uv_path,
                "venv",
                str(venv_path),
                "--seed",
                "--python",
                sys.executable,
            ]
        )

        if config["share_torch"]:
            if os.name == "nt":
                child_site = venv_path / "Lib" / "site-packages"
            else:
                vi = sys.version_info
                child_site = venv_path / "lib" / f"python{vi.major}.{vi.minor}" / "site-packages"

            if not child_site.exists():
                raise RuntimeError(
                    f"site-packages not found at expected path: {child_site}. venv may be malformed."
                )

            parent_sites = site.getsitepackages()
            host_prefix = sys.prefix
            valid_parents = [p for p in parent_sites if p.startswith(host_prefix)]
            if not valid_parents:
                valid_parents = [p for p in sys.path if "site-packages" in p and p.startswith(host_prefix)]
            if not valid_parents:
                raise RuntimeError(
                    "Could not determine parent site-packages path to inherit. "
                    f"host_prefix={host_prefix}, site_packages={parent_sites}, "
                    f"valid_parents={valid_parents}, "
                    f"candidates={[p for p in sys.path if 'site-packages' in p]}"
                )

            # On Windows, getsitepackages() may return venv root before site-packages.
            # Prefer the actual site-packages path for correct package inheritance.
            site_packages_paths = [p for p in valid_parents if "site-packages" in p]
            parent_site = site_packages_paths[0] if site_packages_paths else valid_parents[0]
            pth_content = f"import site; site.addsitedir(r'{parent_site}')\n"
            pth_file = child_site / "_pyisolate_parent.pth"
            pth_file.write_text(pth_content)


def install_dependencies(venv_path: Path, config: ExtensionConfig, name: str) -> None:
    """Install extension dependencies into the venv, skipping already-satisfied ones."""
    # Windows multiprocessing/Manager uses the interpreter path for spawned
    # processes. The explicit Scripts/python.exe path is required to avoid
    # handle issues when multiprocessing.set_executable is involved.
    python_exe = venv_path / "Scripts" / "python.exe" if os.name == "nt" else venv_path / "bin" / "python"

    if not python_exe.exists():
        raise RuntimeError(f"Python executable not found at {python_exe}")

    uv_path = shutil.which("uv")
    if not uv_path:
        raise RuntimeError(
            "uv is required but not found. Install it with: pip install uv\n"
            "See https://github.com/astral-sh/uv for installation options."
        )

    safe_deps: list[str] = []
    for dep in config["dependencies"]:
        validate_dependency(dep)
        safe_deps.append(dep)

    if config["share_torch"] and safe_deps:
        safe_deps = exclude_satisfied_requirements(config, safe_deps, python_exe)

    if not safe_deps:
        return

    # uv handles hardlink vs copy automatically based on filesystem support
    cmd_prefix: list[str] = [uv_path, "pip", "install", "--python", str(python_exe)]
    cache_dir_override = os.environ.get("PYISOLATE_UV_CACHE_DIR")
    cache_dir = Path(cache_dir_override) if cache_dir_override else (venv_path.parent / ".uv_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    common_args: list[str] = ["--cache-dir", str(cache_dir)]

    torch_spec: str | None = None
    if not config["share_torch"]:
        import torch

        torch_version: str = str(torch.__version__)
        if torch_version.endswith("+cpu"):
            torch_version = torch_version[:-4]
        cuda_version = torch.version.cuda  # type: ignore[attr-defined]
        if cuda_version:
            common_args += [
                "--extra-index-url",
                f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}",
            ]
        if "dev" in torch_version or "+" in torch_version:
            common_args += ["--index-strategy", "unsafe-best-match"]
        torch_spec = f"torch=={torch_version}"
        safe_deps.insert(0, torch_spec)

    descriptor = {
        "dependencies": safe_deps,
        "share_torch": config["share_torch"],
        "torch_spec": torch_spec,
        "pyisolate": pyisolate_version,
        "python": sys.version,
    }
    fingerprint = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
    lock_path = venv_path / ".pyisolate_deps.json"

    if lock_path.exists():
        try:
            cached = json.loads(lock_path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint and cached.get("descriptor") == descriptor:
                return
        except Exception as exc:
            logger.debug("Dependency cache read failed: %s", exc)

    install_targets: list[str] = []
    i = 0
    while i < len(safe_deps):
        dep = safe_deps[i]
        dep_stripped = dep.strip()

        # Support split editable args from existing callers:
        # ["-e", "/path/to/pkg"].
        if dep_stripped == "-e":
            if i + 1 >= len(safe_deps):
                raise ValueError("Editable dependency '-e' must include a path or URL")
            editable_target = safe_deps[i + 1].strip()
            if not editable_target:
                raise ValueError("Editable dependency '-e' must include a path or URL")
            install_targets.extend(["-e", editable_target])
            i += 2
            continue

        if dep_stripped.startswith("-e "):
            editable_target = dep_stripped[3:].strip()
            if not editable_target:
                raise ValueError("Editable dependency must include a path or URL after '-e'")
            install_targets.extend(["-e", editable_target])
        else:
            install_targets.append(dep)
        i += 1

    cmd = cmd_prefix + install_targets + common_args

    with subprocess.Popen(  # noqa: S603  # Trusted: validated pip/uv install cmd
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        output_lines: list[str] = []
        for line in proc.stdout:
            clean = line.rstrip()
            # Filter out pyisolate install messages to avoid polluting logs
            # with internal dependency resolution noise that isn't actionable
            # for users debugging their own extension dependencies.
            if "pyisolate==" not in clean and "pyisolate @" not in clean:
                output_lines.append(clean)
        return_code = proc.wait()

    if return_code != 0:
        detail = "\n".join(output_lines) or "(no output)"
        raise RuntimeError(f"Install failed for {name}: {detail}")

    lock_path.write_text(
        json.dumps({"fingerprint": fingerprint, "descriptor": descriptor}, indent=2),
        encoding="utf-8",
    )
