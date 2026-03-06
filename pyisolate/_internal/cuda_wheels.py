from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import TypedDict, cast
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import urlopen

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

from ..config import CUDAWheelConfig

_TORCH_VERSION_RE = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)")
_CUDA_LOCAL_PATTERNS = (
    re.compile(r"(^|[.-])cu(?P<cuda>\d+)torch(?P<torch>\d+)([.-]|$)"),
    re.compile(r"(^|[.-])pt(?P<torch>\d+)cu(?P<cuda>\d+)([.-]|$)"),
)


class CUDAWheelRuntime(TypedDict):
    torch: str
    torch_nodot: str
    cuda: str
    cuda_nodot: str
    python_tags: list[str]


class _SimpleIndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attributes = dict(attrs)
        href = attributes.get("href")
        if href:
            self.hrefs.append(href)


class CUDAWheelResolutionError(RuntimeError):
    pass


def _parse_major_minor(version_text: str, label: str) -> str:
    match = _TORCH_VERSION_RE.match(version_text)
    if not match:
        raise CUDAWheelResolutionError(f"Could not parse {label} major.minor from '{version_text}'")
    return f"{match.group('major')}.{match.group('minor')}"


def get_cuda_wheel_runtime() -> CUDAWheelRuntime:
    try:
        import torch
    except ImportError as exc:
        raise CUDAWheelResolutionError(
            "Custom CUDA wheel resolution requires host torch to be installed"
        ) from exc

    torch_version = _parse_major_minor(str(torch.__version__), "torch version")
    cuda_version = torch.version.cuda  # type: ignore[attr-defined]
    if not cuda_version:
        raise CUDAWheelResolutionError(
            "Custom CUDA wheel resolution requires a CUDA-enabled host torch build"
        )
    cuda_major_minor = _parse_major_minor(str(cuda_version), "CUDA version")
    return {
        "torch": torch_version,
        "torch_nodot": torch_version.replace(".", ""),
        "cuda": cuda_major_minor,
        "cuda_nodot": cuda_major_minor.replace(".", ""),
        "python_tags": [str(tag) for tag in sys_tags()],
    }


def get_cuda_wheel_runtime_descriptor() -> dict[str, object]:
    runtime = get_cuda_wheel_runtime()
    return {
        "torch": runtime["torch"],
        "torch_nodot": runtime["torch_nodot"],
        "cuda": runtime["cuda"],
        "cuda_nodot": runtime["cuda_nodot"],
        "python_tags": runtime["python_tags"],
    }


def _normalize_cuda_wheel_config(config: CUDAWheelConfig) -> CUDAWheelConfig:
    index_url = config.get("index_url")
    packages = config.get("packages")
    package_map = config.get("package_map", {})

    if not isinstance(index_url, str) or not index_url.strip():
        raise CUDAWheelResolutionError("cuda_wheels.index_url must be a non-empty string")
    if not isinstance(packages, list) or not all(
        isinstance(package_name, str) and package_name.strip() for package_name in packages
    ):
        raise CUDAWheelResolutionError("cuda_wheels.packages must be a list of non-empty strings")
    if not isinstance(package_map, dict):
        raise CUDAWheelResolutionError("cuda_wheels.package_map must be a mapping")

    normalized_map: dict[str, str] = {}
    for dependency_name, index_package_name in package_map.items():
        if not isinstance(dependency_name, str) or not dependency_name.strip():
            raise CUDAWheelResolutionError("cuda_wheels.package_map keys must be non-empty strings")
        if not isinstance(index_package_name, str) or not index_package_name.strip():
            raise CUDAWheelResolutionError("cuda_wheels.package_map values must be non-empty strings")
        normalized_map[canonicalize_name(dependency_name)] = index_package_name.strip()

    return {
        "index_url": index_url.rstrip("/") + "/",
        "packages": [canonicalize_name(package_name) for package_name in packages],
        "package_map": normalized_map,
    }


def _candidate_package_names(dependency_name: str, package_map: dict[str, str]) -> list[str]:
    candidates: list[str] = []
    mapped_name = package_map.get(dependency_name)
    if mapped_name:
        candidates.append(mapped_name.strip())
        candidates.append(mapped_name.replace("-", "_"))
        candidates.append(mapped_name.replace("_", "-"))

    candidates.append(dependency_name)
    candidates.append(dependency_name.replace("-", "_"))
    candidates.append(dependency_name.replace("_", "-"))

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def _fetch_index_html(url: str) -> str | None:
    try:
        with urlopen(url, timeout=30) as response:  # noqa: S310 - URL is explicit extension config
            content: bytes = response.read()
            return content.decode("utf-8")
    except (HTTPError, URLError, FileNotFoundError):
        return None


def _parse_index_links(page_url: str, html: str) -> list[str]:
    parser = _SimpleIndexParser()
    parser.feed(html)
    return [urljoin(page_url, href) for href in parser.hrefs]


def _normalize_wheel_url(raw_url: str) -> str:
    parsed_url = urlparse(raw_url)
    return parsed_url._replace(path=unquote(parsed_url.path)).geturl()


def _matches_runtime(local_version: str | None, runtime: CUDAWheelRuntime) -> bool:
    if not local_version:
        return False
    normalized_local = local_version.lower()
    for pattern in _CUDA_LOCAL_PATTERNS:
        match = pattern.search(normalized_local)
        if not match:
            continue
        if match.group("torch") == runtime["torch_nodot"] and match.group("cuda") == runtime["cuda_nodot"]:
            return True
    return False


def resolve_cuda_wheel_url(
    requirement: Requirement, config: CUDAWheelConfig, runtime: CUDAWheelRuntime | None = None
) -> str:
    normalized_config = _normalize_cuda_wheel_config(config)
    dependency_name = canonicalize_name(requirement.name)
    runtime_info = runtime or get_cuda_wheel_runtime()
    supported_tag_list = list(sys_tags())
    supported_tags = set(supported_tag_list)
    tag_rank = {tag: idx for idx, tag in enumerate(supported_tag_list)}
    fetch_attempted = False
    candidates: list[tuple[Version, int, str]] = []

    for package_name in _candidate_package_names(dependency_name, normalized_config.get("package_map", {})):
        page_url = urljoin(normalized_config["index_url"], package_name.rstrip("/") + "/")
        html = _fetch_index_html(page_url)
        if html is None:
            continue
        fetch_attempted = True
        for wheel_url in _parse_index_links(page_url, html):
            parsed_url = urlparse(wheel_url)
            wheel_filename = unquote(parsed_url.path.rsplit("/", 1)[-1])
            if not wheel_filename.endswith(".whl"):
                continue
            try:
                wheel_name, wheel_version, _, wheel_tags = parse_wheel_filename(wheel_filename)
            except ValueError:
                continue
            if canonicalize_name(wheel_name) != dependency_name:
                continue
            matching_tags = wheel_tags.intersection(supported_tags)
            if not matching_tags:
                continue
            if not _matches_runtime(getattr(wheel_version, "local", None), runtime_info):
                continue
            if requirement.specifier and wheel_version not in requirement.specifier:
                continue
            candidates.append(
                (
                    wheel_version,
                    min(tag_rank[tag] for tag in matching_tags),
                    _normalize_wheel_url(wheel_url),
                )
            )

    if not fetch_attempted:
        raise CUDAWheelResolutionError(
            f"No CUDA wheel index page found for '{requirement.name}' under {normalized_config['index_url']}"
        )
    if not candidates:
        raise CUDAWheelResolutionError(
            "No compatible CUDA wheel found for "
            f"'{requirement}' (torch {runtime_info['torch']}, CUDA {runtime_info['cuda']})"
        )

    candidates.sort(key=lambda item: (item[0], -item[1]))
    return candidates[-1][2]


def resolve_cuda_wheel_requirements(requirements: list[str], config: CUDAWheelConfig) -> list[str]:
    normalized_config = _normalize_cuda_wheel_config(config)
    configured_packages = set(normalized_config["packages"])
    environment = cast(dict[str, str], default_environment())
    runtime = get_cuda_wheel_runtime()
    resolved_requirements: list[str] = []

    for dependency in requirements:
        stripped = dependency.strip()
        if not stripped or stripped == "-e" or stripped.startswith("-e "):
            resolved_requirements.append(dependency)
            continue
        if stripped.startswith(("/", "./", "../", "file://")):
            resolved_requirements.append(dependency)
            continue

        try:
            requirement = Requirement(stripped)
        except InvalidRequirement:
            resolved_requirements.append(dependency)
            continue

        dependency_name = canonicalize_name(requirement.name)
        if dependency_name not in configured_packages:
            resolved_requirements.append(dependency)
            continue
        if requirement.url:
            raise CUDAWheelResolutionError(
                f"cuda_wheels dependency '{requirement.name}' must not already use a direct URL"
            )
        if requirement.extras:
            raise CUDAWheelResolutionError(f"cuda_wheels dependency '{requirement.name}' must not use extras")
        if requirement.marker and not requirement.marker.evaluate(environment):
            resolved_requirements.append(dependency)
            continue

        resolved_requirements.append(resolve_cuda_wheel_url(requirement, normalized_config, runtime))

    return resolved_requirements
