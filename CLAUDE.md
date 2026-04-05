# CLAUDE.md

Guidance for AI agents maintaining the pyisolate codebase.

## Identity

**pyisolate** is a Python library (PyPI: `pyisolate`, v0.10.1) for running extensions in isolated virtual environments with seamless inter-process communication. It provides dependency isolation, zero-copy tensor transfer, and bubblewrap sandboxing for GPU-heavy workloads.

License: MIT | Python: >=3.10

## Architecture

### Public API (`pyisolate/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports: `ExtensionBase`, `ExtensionManager`, `ExtensionConfig`, `SandboxMode`, `ProxiedSingleton`, `SealedNodeExtension`, adapter registration |
| `host.py` | `ExtensionManager` — creates/manages isolated extensions and their venvs |
| `shared.py` | `ExtensionBase` / `ExtensionLocal` — base classes with lifecycle hooks (`before_module_loaded`, `on_module_loaded`) |
| `sealed.py` | `SealedNodeExtension` — minimal extension for sealed workers (no host framework imports) |
| `config.py` | TypedDicts: `ExtensionManagerConfig`, `ExtensionConfig`, `SandboxConfig`, `CUDAWheelConfig`. Enum: `SandboxMode` |
| `interfaces.py` | `IsolationAdapter` and `SerializerRegistryProtocol` — structural typing protocols for application adapters |
| `path_helpers.py` | Host `sys.path` serialization and child-side reconstruction |

### Internal (`pyisolate/_internal/`)

| File | Purpose |
|------|---------|
| `rpc_protocol.py` | `AsyncRPC` engine, `ProxiedSingleton` metaclass, `LocalMethodRegistry` |
| `rpc_transports.py` | `JSONSocketTransport` (primary) — length-prefixed JSON over UDS. No pickle. `QueueTransport` (legacy) |
| `rpc_serialization.py` | Message structures: `RPCRequest`, `RPCResponse`, `RPCCallback` |
| `serialization_registry.py` | `SerializerRegistry` — O(1) type lookup, MRO chain, `data_type` flag |
| `tensor_serializer.py` | Zero-copy tensors via `/dev/shm` (CPU) and CUDA IPC (GPU). `TensorKeeper` (5.0s default retention) |
| `model_serialization.py` | Generic `serialize_for_isolation()` / `deserialize_from_isolation()` |
| `host.py` | `Extension` class — process lifecycle (venv → deps → launch → RPC → shutdown) |
| `bootstrap.py` | Child-side init: sys.path reconstruction, adapter rehydration |
| `uds_client.py` | Child entrypoint (`python -m pyisolate._internal.uds_client`) |
| `environment.py` | uv venv creation, dependency installation, torch package exclusion |
| `environment_conda.py` | pixi/conda environment creation, fingerprint caching |
| `cuda_wheels.py` | CUDA wheel resolver — probes host torch/CUDA, fetches matching wheels |
| `sandbox.py` | `build_bwrap_command()` — deny-by-default filesystem, GPU passthrough, sealed-worker `--clearenv` |
| `sandbox_detect.py` | Multi-distro detection: RHEL, Ubuntu AppArmor, SELinux, Arch |
| `event_bridge.py` | Child-to-host event dispatch |
| `perf_trace.py` | Structured event logging (`PYISOLATE_TRACE_FILE`) |

### Other Directories

| Directory | Purpose |
|-----------|---------|
| `tests/` | Unit and integration tests (pytest, ~50 test functions) |
| `example/` | Working 3-extension demo showing dependency isolation (numpy 1.x vs 2.x) |
| `benchmarks/` | RPC overhead and memory benchmarks with cross-platform runner scripts |
| `docs/` | Reference docs: RPC protocol, debugging, edge cases, platform compatibility |

## Development Commands

```bash
# Environment setup
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
pre-commit install

# Testing
pytest                                    # all tests with coverage
pytest tests/test_rpc_contract.py -v      # specific test file
pytest -k "test_sandbox" -v               # pattern match

# Code quality
ruff check pyisolate tests
ruff format pyisolate tests

# Build
python -m build
```

## Public API Surface

Exported from `pyisolate/__init__.py`:

```
ExtensionBase           ExtensionManager        ExtensionManagerConfig
ExtensionConfig         SandboxMode             SealedNodeExtension
ProxiedSingleton        local_execution         singleton_scope
flush_tensor_keeper     purge_orphan_sender_shm_files
register_adapter        get_adapter
```

Everything in `_internal/` is private implementation. Do not expose internal types in public API changes.

## Isolation Modes

| Mode | Provisioner | share_torch | Tensor Transport |
|:-----|:------------|:------------|:-----------------|
| cuda_share | uv | yes | CUDA IPC + `/dev/shm` |
| torch_share | uv | yes | `/dev/shm` only |
| json_share | uv | no | JSON serialization |
| sealed_worker | uv or pixi | no | JSON serialization |

Invalid: pixi + `share_torch=True`. Invalid: `share_cuda_ipc=True` without `share_torch=True`.

## Testing Conventions

- Tests live in `tests/` with `test_` prefix
- Integration tests that create real venvs are slow (~30s each)
- Tests marked `@pytest.mark.network` require external network access
- `pytest --cov=pyisolate` for coverage (configured in pyproject.toml)
- No tests should import from `_internal/` unless testing internal behavior

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `PYISOLATE_CHILD` | `"1"` in child processes |
| `PYISOLATE_DEBUG_RPC` | `"1"` for verbose RPC logging |
| `PYISOLATE_TRACE_FILE` | Path for structured perf trace JSONL output |
| `PYISOLATE_ENABLE_CUDA_IPC` | `"1"` to enable CUDA IPC tensor transport |
| `PYISOLATE_PATH_DEBUG` | `"1"` for sys.path logging during child init |
| `PYISOLATE_ENFORCE_SANDBOX` | Force bwrap sandboxing |

## Key Invariants

- No pickle anywhere in the transport layer — JSON-RPC only
- Library is application-agnostic — no references to specific integrations in library code
- Fail loud — surface failures immediately, no silent degradation
- `_internal/` is private — public API goes through `__init__.py` exports
