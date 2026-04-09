"""Conda/pixi environment creation for pyisolate extensions."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from packaging.utils import canonicalize_name

from ..config import CUDAWheelConfig, ExtensionConfig

logger = logging.getLogger(__name__)


def _pyisolate_source_path() -> Path:
    """Return the local pyisolate source tree for sealed-worker child installs."""
    return Path(__file__).resolve().parents[2]


def _toml_path_string(path: Path) -> str:
    """Serialize a local path safely for pixi TOML."""
    return path.as_posix()


def _detect_glibc_version() -> str | None:
    """Detect the host glibc version for pixi system-requirements."""
    try:
        import platform as plat

        ver = plat.libc_ver()[1]
        if ver:
            return ver
    except Exception:
        pass
    return None


def _generate_pixi_toml(config: ExtensionConfig) -> str:
    """Generate a pixi.toml manifest from an ExtensionConfig.

    Maps conda_dependencies → [dependencies], pip dependencies → [pypi-dependencies].
    """
    lines: list[str] = []

    # [workspace] section
    name = config.get("module", "extension")
    lines.append("[workspace]")
    lines.append(f'name = "{name}"')
    lines.append('version = "0.1.0"')

    # channels
    channels = config.get("conda_channels", [])
    if channels:
        channels_str = ", ".join(f'"{c}"' for c in channels)
        lines.append(f"channels = [{channels_str}]")

    # platforms
    platforms = config.get("conda_platforms", [])
    if not platforms:
        # Auto-detect current platform
        if sys.platform == "linux":
            platforms = ["linux-64"]
        elif sys.platform == "darwin":
            import platform as plat

            arch = plat.machine()
            platforms = ["osx-arm64"] if arch == "arm64" else ["osx-64"]
        elif sys.platform == "win32":
            platforms = ["win-64"]
        else:
            platforms = ["linux-64"]
    platforms_str = ", ".join(f'"{p}"' for p in platforms)
    lines.append(f"platforms = [{platforms_str}]")
    lines.append("")

    # [system-requirements] — detect host glibc for correct wheel matching
    if sys.platform == "linux":
        glibc_version = _detect_glibc_version()
        if glibc_version:
            lines.append("[system-requirements]")
            lines.append(f'libc = {{ family = "glibc", version = "{glibc_version}" }}')
            lines.append("")

    # [dependencies] — conda packages
    conda_deps = config.get("conda_dependencies", [])
    lines.append("[dependencies]")
    python_version = config.get("conda_python", "*")
    lines.append(f'python = "{python_version}"')
    if conda_deps:
        for dep in conda_deps:
            # Parse "numpy>=1.20" → name="numpy", spec=">=1.20"
            # Parse "numpy" → name="numpy", spec="*"
            name_part, sep, version_part, _extras, _marker = _parse_dep(dep)
            if version_part:
                lines.append(f'{name_part} = "{version_part}"')
            else:
                lines.append(f'{name_part} = "*"')
    lines.append("")

    # [pypi-dependencies] — pip packages
    pip_deps = list(config.get("dependencies", []))

    cuda_wheels_config = config.get("cuda_wheels")
    cuda_wheel_packages: set[str] = set()
    if cuda_wheels_config:
        cuda_wheel_packages = {
            canonicalize_name(package_name) for package_name in cuda_wheels_config.get("packages", [])
        }

    if config.get("package_manager") == "conda":
        # [pypi-options] — extra index URLs and find-links for local wheels
        pypi_options_lines: list[str] = []
        if cuda_wheels_config:
            # Support both single index_url and multiple index_urls
            index_urls = cuda_wheels_config.get("index_urls", [])
            if not index_urls:
                single = cuda_wheels_config.get("index_url", "")
                if single:
                    index_urls = [single]
            if index_urls:
                urls_str = ", ".join(f'"{u}"' for u in index_urls)
                pypi_options_lines.append(f"extra-index-urls = [{urls_str}]")
        find_links_raw = config.get("find_links", [])
        if isinstance(find_links_raw, str):
            find_links: list[str] = [find_links_raw]
        elif isinstance(find_links_raw, list):
            find_links = find_links_raw
        else:
            find_links = []
        if find_links:
            # Resolve relative paths against the extension's module_path
            resolved = []
            module_path = config.get("module_path")
            for link in find_links:
                link_path = Path(link)
                if not link_path.is_absolute() and module_path:
                    link_path = Path(module_path) / link_path
                resolved.append(f'{{ path = "{_toml_path_string(link_path)}" }}')
            links_str = ", ".join(resolved)
            pypi_options_lines.append(f"find-links = [{links_str}]")
            # Pixi treats find-links as exclusive; add PyPI so regular deps resolve
            has_extra_index = any("extra-index-urls" in line for line in pypi_options_lines)
            if not has_extra_index:
                pypi_options_lines.insert(0, 'extra-index-urls = ["https://pypi.org/simple/"]')
        if pypi_options_lines:
            lines.append("[pypi-options]")
            lines.extend(pypi_options_lines)
            lines.append("")

        lines.append("[pypi-dependencies]")
        source_path = _pyisolate_source_path()
        if (source_path / "pyproject.toml").exists():
            lines.append(f'pyisolate = {{ path = "{_toml_path_string(source_path)}" }}')
        else:
            try:
                version = importlib.metadata.version("pyisolate")
            except importlib.metadata.PackageNotFoundError:
                version = "0.0.0"
            lines.append(f'pyisolate = "=={version}"')
        for dep in pip_deps:
            name_part, sep, version_part, extras, marker = _parse_dep(dep)
            if cuda_wheel_packages and canonicalize_name(name_part) in cuda_wheel_packages:
                continue
            # Build pixi inline table fields
            fields: list[str] = []
            if sep == "@":
                fields.append(f'url = "{version_part}"')
            else:
                ver = version_part if version_part else "*"
                fields.append(f'version = "{ver}"')
            if extras:
                extras_str = ", ".join(f'"{e}"' for e in extras)
                fields.append(f"extras = [{extras_str}]")
            if marker:
                fields.append(f'markers = "{marker}"')
            # Emit: simple string form when only version, inline table otherwise
            if len(fields) == 1 and fields[0].startswith("version"):
                ver = version_part if version_part else "*"
                lines.append(f'{name_part} = "{ver}"')
            else:
                lines.append(f"{name_part} = {{ {', '.join(fields)} }}")
        lines.append("")

    return "\n".join(lines) + "\n"


def _build_pixi_install_env(env_path: Path) -> dict[str, str]:
    """Build a stable subprocess env for pixi installs.

    Harness-backed tests may leave ambient TMPDIR pointing at a deleted
    directory. Give pixi its own guaranteed-writable temp root instead of
    inheriting whatever process-global temp state happens to exist.
    """
    env = os.environ.copy()
    pixi_tmp = env_path / ".tmp"
    pixi_tmp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(pixi_tmp)
    env["TMP"] = str(pixi_tmp)
    env["TEMP"] = str(pixi_tmp)
    return env


def _parse_dep(dep: str) -> tuple[str, str, str, list[str], str]:
    """Parse a PEP 508 dependency string into (name, separator, version_spec, extras, marker).

    Handles extras (trimesh[easy]>=4.0.0), URL deps (pkg @ https://...),
    and PEP 508 environment markers (jax>=0.4.30; sys_platform == 'linux').
    Extras are extracted and returned separately for pixi ``extras = [...]`` syntax.
    Markers are split off before version parsing so they don't contaminate the
    version spec.

    Examples:
        "numpy>=1.20" → ("numpy", ">=", ">=1.20", [], "")
        "numpy" → ("numpy", "", "", [], "")
        "scipy==1.10.0" → ("scipy", "==", "==1.10.0", [], "")
        "trimesh[easy]>=4.0.0" → ("trimesh", ">=", ">=4.0.0", ["easy"], "")
        "jax[cuda12]>=0.4.30" → ("jax", ">=", ">=0.4.30", ["cuda12"], "")
        "jax[cuda12]>=0.4.30; sys_platform == 'linux'"
            → ("jax", ">=", ">=0.4.30", ["cuda12"], "sys_platform == 'linux'")
        "pkg @ https://example.com/pkg.whl"
            → ("pkg", "@", "https://example.com/pkg.whl", [], "")
        "pkg @ https://example.com/pkg.whl ; python_version >= '3.12'"
            → ("pkg", "@", "https://example.com/pkg.whl", [], "python_version >= '3.12'")
    """
    # Split off PEP 508 marker before any other parsing.
    # Markers follow a semicolon: "dep_spec ; marker_expr"
    marker = ""
    if ";" in dep:
        dep, _, marker = dep.partition(";")
        dep = dep.strip()
        marker = marker.strip()

    # Extract extras if present
    extras: list[str] = []
    if "[" in dep:
        bracket_start = dep.index("[")
        bracket_end = dep.index("]", bracket_start)
        extras_str = dep[bracket_start + 1 : bracket_end]
        extras = [e.strip() for e in extras_str.split(",") if e.strip()]

    # Handle URL deps: "name @ url"
    if " @ " in dep:
        name_part, _, url = dep.partition(" @ ")
        name_part = name_part.strip()
        if "[" in name_part:
            name_part = name_part[: name_part.index("[")]
        return name_part.strip(), "@", url.strip(), extras, marker

    # Strip extras from dep before parsing version
    clean_dep = dep
    if "[" in dep:
        bracket_start = dep.index("[")
        bracket_end = dep.index("]", bracket_start)
        clean_dep = dep[:bracket_start] + dep[bracket_end + 1 :]

    for sep in (">=", "<=", "==", "!=", "~=", ">", "<"):
        idx = clean_dep.find(sep)
        if idx > 0:
            return clean_dep[:idx].strip(), sep, clean_dep[idx:].strip(), extras, marker
    return clean_dep.strip(), "", "", extras, marker


def create_conda_env(env_path: Path, config: ExtensionConfig, name: str) -> None:
    """Create a conda/pixi environment for an extension.

    Writes pixi.toml, runs pixi install, and writes a fingerprint lock file.
    Skips install if the fingerprint matches a previous run.
    """
    env_path.mkdir(parents=True, exist_ok=True)

    from pyisolate._internal.pixi_provisioner import ensure_pixi
    pixi_path = ensure_pixi()

    cuda_wheels_config = config.get("cuda_wheels")

    # Generate pixi.toml content
    toml_content = _generate_pixi_toml(config)

    # Build fingerprint descriptor
    descriptor = {
        "conda_dependencies": config.get("conda_dependencies", []),
        "pip_dependencies": config.get("dependencies", []),
        "channels": config.get("conda_channels", []),
        "platforms": config.get("conda_platforms", []),
        "cuda_wheels": config.get("cuda_wheels"),
        "find_links": config.get("find_links", []),
        "pixi_toml": toml_content,
    }
    fingerprint = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()

    # Check fingerprint — skip install if unchanged
    lock_path = env_path / ".pyisolate_deps.json"
    if lock_path.exists():
        try:
            cached = json.loads(lock_path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint and cached.get("descriptor") == descriptor:
                # Verify python exe still exists before skipping
                _resolve_pixi_python(env_path)
                logger.debug(
                    "Conda env fingerprint match for %s, skipping pixi install",
                    name,
                )
                return
        except Exception as exc:
            logger.debug("Conda fingerprint cache read failed: %s", exc)

    # Write pixi.toml
    toml_path = env_path / "pixi.toml"
    toml_path.write_text(toml_content, encoding="utf-8")

    # Run pixi install
    pixi_env = _build_pixi_install_env(env_path)
    subprocess.check_call(
        [pixi_path, "install", "--manifest-path", str(toml_path)],
        env=pixi_env,  # noqa: S603
    )

    # Verify python exists after install
    python_exe = _resolve_pixi_python(env_path)

    if cuda_wheels_config:
        _install_cuda_wheels_into_pixi(
            python_exe,
            config,
            cuda_wheels_config,
            name,
        )

    # Install local wheels from find_links directories (post-pixi, --no-deps)
    fl_raw = config.get("find_links", [])
    if isinstance(fl_raw, str):
        fl_list: list[str] = [fl_raw]
    elif isinstance(fl_raw, list):
        fl_list = fl_raw
    else:
        fl_list = []
    if fl_list:
        _install_local_wheels(python_exe, config, fl_list, name)

    # Write fingerprint
    lock_path.write_text(
        json.dumps({"fingerprint": fingerprint, "descriptor": descriptor}),
        encoding="utf-8",
    )


def _parse_conda_python_target(conda_python: str) -> tuple[int, int] | None:
    """Parse a conda_python spec like ``"3.12.*"`` into ``(3, 12)``.

    Returns ``None`` for wildcard (``"*"``) or unparseable values,
    which causes the CUDA wheel resolver to fall back to host tags.
    """
    if not conda_python or conda_python == "*":
        return None
    import re

    match = re.match(r"(\d+)\.(\d+)", conda_python)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)))


def _resolve_uv_exe(python_exe: Path) -> str:
    """Resolve the uv executable path, preferring the pixi env's bin dir.

    On Linux, pixi places uv alongside python in `bin/`. On Windows, python
    is at `.pixi/envs/default/python.exe` (no bin/ subdirectory), so the
    sibling path does not exist. Falls back to system PATH via shutil.which.
    """
    local_uv = python_exe.parent / "uv"
    if local_uv.exists():
        return str(local_uv)
    found = shutil.which("uv")
    if found:
        return found
    raise RuntimeError(f"uv is required but not found. Checked pixi env ({local_uv}) and system PATH.")


def _install_cuda_wheels_into_pixi(
    python_exe: Path,
    config: ExtensionConfig,
    cuda_wheels_config: CUDAWheelConfig,
    name: str,
) -> None:
    """Install CUDA wheels into a pixi environment via pip --no-deps.

    Uses the same resolver as the uv path (cuda_wheels.py) but installs
    into the pixi env's Python instead of a uv venv.
    """
    from .cuda_wheels import resolve_cuda_wheel_requirements

    target_python = _parse_conda_python_target(str(config.get("conda_python", "*")))
    deps = list(config.get("dependencies", []))
    resolved = resolve_cuda_wheel_requirements(deps, cuda_wheels_config, target_python=target_python)

    wheel_urls = []
    for orig, res in zip(deps, resolved, strict=True):
        if orig != res:
            wheel_urls.append(res)
            logger.info("][ CUDA_WHEEL_CONDA ext=%s dep=%s -> %s", name, orig, res)

    if not wheel_urls:
        return

    uv_exe = _resolve_uv_exe(python_exe)
    pip_cmd = [uv_exe, "pip", "install", "--no-deps", "--python", str(python_exe)]
    pip_cmd.extend(wheel_urls)

    logger.info("][ CUDA_WHEEL_CONDA_INSTALL ext=%s count=%d", name, len(wheel_urls))
    subprocess.check_call(pip_cmd)  # noqa: S603


def _install_local_wheels(
    python_exe: Path,
    config: ExtensionConfig,
    find_links: list[str],
    name: str,
) -> None:
    """Install all .whl files from find_links directories with --no-deps.

    Used for pre-built CUDA/extension wheels that are shipped in-repo
    and have internal cross-dependencies that pixi's resolver can't handle.
    """
    module_path = config.get("module_path")
    wheel_files: list[str] = []
    for link_dir in find_links:
        link_path = Path(link_dir)
        if not link_path.is_absolute() and module_path:
            link_path = Path(module_path) / link_path
        if link_path.is_dir():
            for whl in sorted(link_path.glob("*.whl")):
                wheel_files.append(str(whl))

    if not wheel_files:
        return

    uv_exe = _resolve_uv_exe(python_exe)
    pip_cmd = [uv_exe, "pip", "install", "--no-deps", "--python", str(python_exe)]
    pip_cmd.extend(wheel_files)

    logger.info("][ LOCAL_WHEELS ext=%s count=%d files=%s", name, len(wheel_files), wheel_files)
    subprocess.check_call(pip_cmd)  # noqa: S603


def _resolve_pixi_python(env_path: Path) -> Path:
    """Resolve the Python interpreter inside a pixi environment.

    Returns the path to the pixi-managed Python, NEVER the host interpreter.
    Raises RuntimeError if the Python executable does not exist.
    """
    if os.name == "nt":
        python_exe = env_path / ".pixi" / "envs" / "default" / "python.exe"
    else:
        python_exe = env_path / ".pixi" / "envs" / "default" / "bin" / "python"

    if not python_exe.exists():
        raise RuntimeError(
            f"Python executable not found at {python_exe}. "
            "pixi install may have failed or the environment is corrupted."
        )

    return python_exe
