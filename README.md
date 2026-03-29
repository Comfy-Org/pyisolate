# pyisolate

**Run Python extensions in isolated virtual environments with seamless inter-process communication.**

> 🚨 **Fail Loud Policy**: pyisolate assumes the rest of ComfyUI core is correct. Missing prerequisites or runtime failures immediately raise descriptive exceptions instead of being silently ignored.

pyisolate enables you to run Python extensions with conflicting dependencies in the same application by automatically creating isolated environments for each extension. The default provisioner uses `uv`, and ComfyUI integrations can also provision a conda environment through `pixi` when an extension needs conda-first packages. Extensions communicate with the host process through a transparent RPC system, making the isolation invisible to your code while keeping the host environment dependency-free.

## Requirements

- Python 3.9+
- The [`uv`](https://github.com/astral-sh/uv) CLI available on your `PATH`
- `pip`/`venv` for bootstrapping the development environment
- PyTorch is optional and only required for tensor-sharing features (for example, `share_torch=True`)

If you want tensor-sharing features, install PyTorch separately (for example: `pip install torch`).

## Environment Variables

PyIsolate uses several environment variables for configuration and debugging:

### Core Variables (Set by PyIsolate automatically)
- **`PYISOLATE_CHILD`**: Set to `"1"` in isolated child processes. Used to detect if code is running in host or child.
- **`PYISOLATE_HOST_SNAPSHOT`**: Path to JSON file containing the host's `sys.path` and environment variables. Used during child process initialization.
- **`PYISOLATE_MODULE_PATH`**: Path to the extension module being loaded. Used to detect ComfyUI root directory.

### Debug Variables (Set by user)
- **`PYISOLATE_PATH_DEBUG`**: Set to `"1"` to enable detailed sys.path logging during child process initialization. Useful for debugging import issues.

Example usage:
```bash
# Enable detailed path logging
export PYISOLATE_PATH_DEBUG=1
python main.py

# Disable path logging (default)
unset PYISOLATE_PATH_DEBUG
python main.py
```

## Quick Start

### Option A – run everything for me

```bash
cd /path/to/pyisolate
./quickstart.sh
```

The script installs `uv`, creates the dev venv, installs pyisolate in editable mode, runs the multi-extension example, and executes the Comfy Hello World demo.

### Option B – manual setup (5 minutes)

1. **Create the dev environment**
    ```bash
    cd /path/to/pyisolate
    uv venv
    source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
    uv pip install -e ".[dev]"
    ```
2. **Run the example extensions**
    ```bash
    cd example
    python main.py
    cd ..
    ```
    Expected output:
    ```
    Extension1      | ✓ PASSED      | Data processing with pandas/numpy 1.x
    Extension2      | ✓ PASSED      | Array processing with numpy 2.x
    Extension3      | ✓ PASSED      | HTML parsing with BeautifulSoup/scipy
    ```
3. **Run the Comfy Hello World**
    ```bash
    cd comfy_hello_world
    python main.py
    ```
    You should see the isolated custom node load, execute, and fetch data from the shared singleton service.

## Documentation

- Project site: https://comfy-org.github.io/pyisolate/
- Walkthroughs & architecture notes: see `mysolate/HELLO_WORLD.md` and `mysolate/GETTING_STARTED.md`

## Key Benefits

- 🔒 **Dependency Isolation**: Run extensions with incompatible dependencies (e.g., numpy 1.x and 2.x) in the same application
- 🚀 **Zero-Copy PyTorch Tensor Sharing**: Share PyTorch tensors between processes without serialization overhead
- 📦 **Multiple Environment Backends**: Use `uv` by default or a conda/pixi environment when the extension needs conda-native dependencies
- 🔄 **Transparent Communication**: Call async methods across process boundaries as if they were local
- 🎯 **Simple API**: Clean, intuitive interface with minimal boilerplate
- ⚡ **Fast**: Uses `uv` for blazing-fast virtual environment creation

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
    # Configure the extension manager
    config = pyisolate.ExtensionManagerConfig(
        venv_root_path="./venvs"
    )
    manager = pyisolate.ExtensionManager(pyisolate.ExtensionBase, config)

    # Load an extension with specific dependencies
    extension = manager.load_extension(
        pyisolate.ExtensionConfig(
            name="data_processor",
            module_path="./extensions/my_extension",
            isolated=True,
            dependencies=["numpy>=2.0.0"]
        )
    )

    # Use the extension
    result = await extension.process_data([1, 2, 3, 4, 5])
    print(f"Mean: {result}")  # Mean: 3.0

    # Cleanup
    await extension.stop()

asyncio.run(main())
```

### PyTorch Tensor Sharing

Share PyTorch tensors between processes without serialization:

```python
# extensions/ml_extension/__init__.py
from pyisolate import ExtensionBase
import torch

class MLExtension(ExtensionBase):
    async def process_tensor(self, tensor: torch.Tensor):
        # Tensor is shared, not copied!
        return tensor.mean()
```

```python
# main.py
extension = manager.load_extension(
    pyisolate.ExtensionConfig(
        name="ml_processor",
        module_path="./extensions/ml_extension",
        share_torch=True  # Enable zero-copy tensor sharing
    )
)

# Large tensor is shared, not serialized
large_tensor = torch.randn(1000, 1000)
mean = await extension.process_tensor(large_tensor)
```

### Execution Model Axis

ComfyUI integrations now treat environment provisioning and runtime boundary as separate choices:

- `package_manager = "uv"` or `package_manager = "conda"` chooses how the child environment is built
- `execution_model = "host-coupled"` or `execution_model = "sealed_worker"` chooses how much host runtime state the child may inherit

`host-coupled` remains the default for the classic `uv` path. `sealed_worker` is the foreign-interpreter path: no host `sys.path` reconstruction, no host framework runtime imports as a crutch, JSON-RPC tensor transport, and no sandbox in this phase.

### UV Backend for Sealed Workers

ComfyUI extensions can also request a sealed `uv` worker explicitly:

```toml
[project]
name = "uv-sealed-node"
version = "0.1.0"
dependencies = ["boltons"]

[tool.comfy.isolation]
can_isolate = true
package_manager = "uv"
execution_model = "sealed_worker"
share_torch = false
```

Trade-offs for `package_manager = "uv"` with `execution_model = "sealed_worker"`:

- `share_torch` must be `False`
- tensors cross the boundary through JSON-compatible RPC values instead of shared-memory tensor handles
- host `sys.path` reconstruction is disabled
- host framework runtime imports such as `comfy.isolation.extension_wrapper` must not be required in the child
- `bwrap` sandboxing is intentionally disabled in this phase

### Conda Backend for Sealed Workers

ComfyUI extensions can declare a conda-backed isolated environment in `pyproject.toml`:

```toml
[project]
name = "weather-node"
version = "0.1.0"
dependencies = ["xarray", "cfgrib"]

[tool.comfy.isolation]
can_isolate = true
package_manager = "conda"
share_torch = false
conda_channels = ["conda-forge"]
conda_dependencies = ["eccodes", "cfgrib"]
```

Trade-offs for `package_manager = "conda"`:

- `share_torch` is forced `False`
- `bwrap` sandboxing is skipped
- the child uses its own interpreter instead of the host Python
- the child is treated as a sealed foreign runtime and must not import host framework runtime code through leaked `sys.path`
- tensor transfer crosses the RPC boundary as JSON-compatible values instead of shared-memory tensor handles

### Shared State with Singletons

Share state across all extensions using ProxiedSingleton:

```python
# shared.py
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
# extensions/extension_a/__init__.py
class ExtensionA(ExtensionBase):
    async def save_result(self, result):
        db = DatabaseAPI()  # Returns proxy to host's instance
        await db.set("result", result)

# extensions/extension_b/__init__.py
class ExtensionB(ExtensionBase):
    async def get_result(self):
        db = DatabaseAPI()  # Returns proxy to host's instance
        return await db.get("result")
```

### Complete Application Structure

A complete pyisolate application requires a special `main.py` entry point to handle virtual environment activation:

```python
# main.py
if __name__ == "__main__":
    # When running as the main script, import and run your host application
    from host import main
    main()
else:
    # When imported by extension processes, ensure venv is properly activated
    import os
    import site
    import sys

    if os.name == "nt":  # Windows-specific venv activation
        venv = os.environ.get("VIRTUAL_ENV", "")
        if venv != "":
            sys.path.insert(0, os.path.join(venv, "Lib", "site-packages"))
            site.addsitedir(os.path.join(venv, "Lib", "site-packages"))
```

```python
# host.py - Your main application logic
import pyisolate
import asyncio

async def async_main():
    # Create extension manager
    config = pyisolate.ExtensionManagerConfig(
        venv_root_path="./extension-venvs"
    )
    manager = pyisolate.ExtensionManager(ExtensionBase, config)

    # Load extensions (e.g., from a directory or configuration file)
    extensions = []
    for extension_path in discover_extensions():
        extension_config = pyisolate.ExtensionConfig(
            name=extension_name,
            module_path=extension_path,
            isolated=True,
            dependencies=load_dependencies(extension_path),
            apis=[SharedAPI]  # Optional shared singletons
        )
        extension = manager.load_extension(extension_config)
        extensions.append(extension)

    # Use extensions
    for extension in extensions:
        result = await extension.process()
        print(f"Result: {result}")

    # Clean shutdown
    for extension in extensions:
        await extension.stop()

def main():
    asyncio.run(async_main())
```

This structure ensures that:
- The host application runs normally when executed directly
- Extension processes properly activate their virtual environments when spawned
- Windows-specific path handling is properly managed

## Features

### Core Features
- **Automatic Virtual Environment Management**: Creates and manages isolated environments automatically
- **Bidirectional RPC**: Extensions can call host methods and vice versa
- **Async/Await Support**: Full support for asynchronous programming
- **Lifecycle Hooks**: `before_module_loaded()`, `on_module_loaded()`, and `stop()` for setup/teardown
- **Error Propagation**: Exceptions are properly propagated across process boundaries

### Advanced Features
- **Dependency Resolution**: Automatically installs extension-specific dependencies
- **Platform Support**: Works on Windows, Linux, and soon to be tested on macOS
- **Context Tracking**: Ensures callbacks happen on the same asyncio loop as the original call
- **Fast Installation**: Uses `uv` for 10-100x faster package installation without every extension having its own copy of libraries

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

## Implementing a Host Adapter (IsolationAdapter)

When integrating pyisolate with your application (like ComfyUI), you implement the `IsolationAdapter` protocol. This tells pyisolate how to configure isolated processes for your environment.

### Reference Implementation

The canonical example is in `tests/fixtures/test_adapter.py`:

```python
from pyisolate.interfaces import IsolationAdapter
from pyisolate._internal.shared import ProxiedSingleton

class MockHostAdapter(IsolationAdapter):
    """Reference adapter showing all protocol methods."""

    @property
    def identifier(self) -> str:
        """Return unique adapter identifier (e.g., 'comfyui')."""
        return "myapp"

    def get_path_config(self, module_path: str) -> dict:
        """Configure sys.path for isolated extensions.

        Returns:
            - preferred_root: Your app's root directory
            - additional_paths: Extra paths for imports
        """
        return {
            "preferred_root": "/path/to/myapp",
            "additional_paths": ["/path/to/myapp/extensions"],
        }

    def setup_child_environment(self, snapshot: dict) -> None:
        """Configure child process after sys.path reconstruction."""
        pass  # Set up logging, environment, etc.

    def register_serializers(self, registry) -> None:
        """Register custom type serializers for RPC transport."""
        registry.register(
            "MyCustomType",
            serializer=lambda obj: {"data": obj.data},
            deserializer=lambda d: MyCustomType(d["data"]),
        )

    def provide_rpc_services(self) -> list:
        """Return ProxiedSingleton classes to expose via RPC."""
        return [MyRegistry, MyProgressReporter]

    def handle_api_registration(self, api, rpc) -> None:
        """Post-registration hook for API-specific setup."""
        pass
```

### Testing Your Adapter

Run the contract tests to verify your adapter implements the protocol correctly:

```bash
# The test suite verifies all protocol methods
pytest tests/test_adapter_contract.py -v
```

## Roadmap

### ✅ Completed
- [x] Core isolation and RPC system
- [x] Automatic virtual environment creation
- [x] Bidirectional communication
- [x] PyTorch tensor sharing
- [x] Shared singleton pattern
- [x] Comprehensive test suite
- [x] Windows, Linux support
- [x] Security features (path normalization)
- [x] Fast installation with `uv`
- [x] Context tracking for RPC calls
- [x] Async/await support
- [x] Performance benchmarking suite
- [x] Memory usage tracking and benchmarking
- [x] Network access restrictions
- [x] Filesystem access sandboxing

### 🚧 In Progress
- [ ] Documentation site
- [ ] macOS testing
- [ ] Wrapper for non-async calls between processes

### 🔮 Future Plans
- [ ] CPU/Memory usage limits
- [ ] Hot-reloading of extensions
- [ ] Distributed RPC (across machines)
- [ ] Profiling and debugging tools

## Use Cases

pyisolate is perfect for:

- **Plugin Systems**: When plugins may require conflicting dependencies
- **ML Pipelines**: Different models requiring different library versions
- **Microservices in a Box**: Multiple services with different dependencies in one app
- **Testing**: Running tests with different dependency versions in parallel
- **Legacy Code Integration**: Wrapping legacy code with specific dependency requirements

## Development

We welcome contributions!

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

### Benchmarking

pyisolate includes a comprehensive benchmarking suite to measure RPC call overhead:

```bash
# Install benchmark dependencies
uv pip install -e ".[bench]"

# Quick benchmark using existing example extensions
python benchmarks/simple_benchmark.py

# Full benchmark suite with statistical analysis
python benchmarks/benchmark.py

# Quick mode with fewer iterations for faster results
python benchmarks/benchmark.py --quick

# Skip torch benchmarks (if torch not available)
python benchmarks/benchmark.py --no-torch

# Skip GPU benchmarks
python benchmarks/benchmark.py --no-gpu
```

#### Example Benchmark Output

```
==================================================
BENCHMARK RESULTS
==================================================
Test            Mean (ms)    Std Dev (ms)   Runs  
--------------------------------------------------
small_int       0.63         0.05           1000  
small_string    0.64         0.06           1000  
medium_string   0.65         0.07           1000  
tiny_tensor     0.79         0.08           1000  
small_tensor    0.80         0.11           1000  
medium_tensor   0.81         0.06           1000  
large_tensor    0.78         0.08           1000  
model_tensor    0.88         0.29           1000  

Fastest result: 0.63ms
```

The benchmarks measure:

1. **Small Data RPC Overhead**: ~0.6ms for basic data types (integers, strings)
2. **Tensor Overhead**: Minimal overhead (~0.2ms) for sharing tensors up to 6GB via zero-copy shared memory
3. **Scaling**: Performance remains O(1) regardless of tensor size

> ⚠️ **Note for CPU Tensors**: When checking out or running benchmarks with `share_torch=True`, ensuring `TMPDIR=/dev/shm` is recommended to guarantee that shared memory files are visible to sandboxed child processes.


## License

pyisolate is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Built on Python's `multiprocessing` and `asyncio`
- Uses [`uv`](https://github.com/astral-sh/uv) for fast package management
- Inspired by plugin systems like Chrome Extensions and VS Code Extensions

---

**Star this repo** if you find it useful! ⭐
