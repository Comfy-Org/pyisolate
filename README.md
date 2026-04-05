# pyisolate

**Run Python extensions in isolated virtual environments with seamless inter-process communication.**

pyisolate enables you to run Python extensions with conflicting dependencies in the same application by automatically creating isolated environments for each extension. Extensions communicate with the host process through a transparent JSON-RPC system, making the isolation invisible to your code while keeping the host environment dependency-free.

## Requirements

- Python 3.10+
- The [`uv`](https://github.com/astral-sh/uv) CLI available on your `PATH`
- PyTorch is optional and only required for tensor-sharing features (`share_torch=True`)

## Key Benefits

- **Dependency Isolation**: Run extensions with incompatible dependencies (e.g., numpy 1.x and 2.x) in the same application
- **Zero-Copy PyTorch Tensor Sharing**: Share PyTorch tensors between processes without serialization overhead
- **Multiple Environment Backends**: Use `uv` by default or a conda/pixi environment for conda-native dependencies
- **Bubblewrap Sandboxing**: Deny-by-default filesystem isolation on Linux with GPU passthrough
- **Transparent Communication**: Call async methods across process boundaries as if they were local
- **Fast**: Uses `uv` for blazing-fast virtual environment creation

## Installation

```bash
pip install pyisolate
```

For development:
```bash
pip install pyisolate[dev]
```

## Quick Start

### Basic Usage

Create an extension that runs in an isolated environment:

```python
# extensions/my_extension/__init__.py
from pyisolate import ExtensionBase

class MyExtension(ExtensionBase):
    def on_module_loaded(self, module):
        self.module = module

    async def process_data(self, data):
        # This runs in an isolated process with its own dependencies
        import numpy as np  # This could be numpy 2.x
        return np.array(data).mean()
```

Load and use the extension from your main application:

```python
# main.py
import pyisolate
import asyncio

async def main():
    config = pyisolate.ExtensionManagerConfig(
        venv_root_path="./venvs"
    )
    manager = pyisolate.ExtensionManager(pyisolate.ExtensionBase, config)

    extension = manager.load_extension(
        pyisolate.ExtensionConfig(
            name="data_processor",
            module_path="./extensions/my_extension",
            isolated=True,
            dependencies=["numpy>=2.0.0"]
        )
    )

    result = await extension.process_data([1, 2, 3, 4, 5])
    print(f"Mean: {result}")  # Mean: 3.0

    await extension.stop()

asyncio.run(main())
```

### PyTorch Tensor Sharing

Share PyTorch tensors between processes without serialization:

```python
extension = manager.load_extension(
    pyisolate.ExtensionConfig(
        name="ml_processor",
        module_path="./extensions/ml_extension",
        share_torch=True,          # Enable zero-copy tensor sharing
        share_cuda_ipc=True,       # CUDA IPC for GPU tensors (Linux)
    )
)

# Large tensor is shared via /dev/shm, not serialized
large_tensor = torch.randn(1000, 1000)
mean = await extension.process_tensor(large_tensor)
```

### Shared State with Singletons

Share state across all extensions using ProxiedSingleton:

```python
from pyisolate import ProxiedSingleton

class DatabaseAPI(ProxiedSingleton):
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value
```

```python
# In any extension — returns proxy to host's real instance
db = DatabaseAPI()
await db.set("result", result)
```

### Execution Models

pyisolate provides two execution models:

- **`host-coupled`** (default): Child process shares the host's torch runtime and can use zero-copy tensor transfer via `/dev/shm` and CUDA IPC.
- **`sealed_worker`**: Fully isolated child with its own interpreter. No host `sys.path` reconstruction, JSON-RPC tensor transport only.

And two environment backends:

- **`uv`** (default): Fast pip-compatible virtual environments.
- **`conda`**: pixi-backed conda environments for packages that need conda-forge.

```python
# Sealed worker with conda environment
config = pyisolate.ExtensionConfig(
    name="weather_processor",
    module_path="./extensions/weather",
    isolated=True,
    execution_model="sealed_worker",
    package_manager="conda",
    share_torch=False,
    conda_channels=["conda-forge"],
    conda_dependencies=["eccodes", "cfgrib"],
    dependencies=["xarray", "cfgrib"],
)
```

### Implementing an Adapter

Applications integrate via the `IsolationAdapter` protocol:

```python
from pyisolate.interfaces import IsolationAdapter

class MyAppAdapter(IsolationAdapter):
    @property
    def identifier(self) -> str:
        return "myapp"

    def get_path_config(self, module_path: str) -> dict:
        return {
            "preferred_root": "/path/to/myapp",
            "additional_paths": ["/path/to/myapp/extensions"],
        }

    def register_serializers(self, registry) -> None:
        registry.register(
            "MyCustomType",
            serializer=lambda obj: {"data": obj.data},
            deserializer=lambda d: MyCustomType(d["data"]),
        )

    def provide_rpc_services(self) -> list:
        return [MyRegistry, MyProgressReporter]
```

## Architecture

```
┌─────────────────────┐     RPC      ┌─────────────┐
│    Host Process     │◄────────────►│ Extension A │
│                     │              │  (venv A)   │
│  ┌──────────────┐   │              └─────────────┘
│  │   Shared     │   │     RPC      ┌─────────────┐
│  │ Singletons   │   │◄────────────►│ Extension B │
│  └──────────────┘   │              │  (venv B)   │
└─────────────────────┘              └─────────────┘
```

## Features

### Core
- Automatic virtual environment management
- Bidirectional JSON-RPC over Unix Domain Sockets (no pickle)
- Full async/await support
- Lifecycle hooks: `before_module_loaded()`, `on_module_loaded()`, `stop()`
- Error propagation across process boundaries

### Advanced
- Bubblewrap sandbox with deny-by-default filesystem (Linux)
- CUDA wheel resolution for custom GPU package builds
- Zero-copy tensor transfer via CUDA IPC and `/dev/shm`
- Performance tracing (`PYISOLATE_TRACE_FILE`)
- Multi-distro sandbox detection (RHEL, Ubuntu, Arch, SELinux)

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PYISOLATE_CHILD` | Set to `"1"` in isolated child processes |
| `PYISOLATE_DEBUG_RPC` | `"1"` for verbose RPC message logging |
| `PYISOLATE_TRACE_FILE` | Path for structured performance trace output |
| `PYISOLATE_ENABLE_CUDA_IPC` | `"1"` to enable CUDA IPC tensor transport |
| `PYISOLATE_PATH_DEBUG` | `"1"` for detailed sys.path logging during child init |

## Development

```bash
# Setup development environment
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,test]"
pre-commit install

# Run tests
pytest

# Run linting
ruff check pyisolate tests

# Run benchmarks
python benchmarks/simple_benchmark.py
```

## Use Cases

pyisolate is designed for:

- **Plugin Systems**: When plugins may require conflicting dependencies
- **ML Pipelines**: Different models requiring different library versions
- **Microservices in a Box**: Multiple services with different dependencies in one app
- **Legacy Code Integration**: Wrapping legacy code with specific dependency requirements

## License

pyisolate is licensed under the MIT License. See [LICENSE](LICENSE) for details.
