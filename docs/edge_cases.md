# Edge Cases and Known Limitations

This document describes edge cases, known limitations, and their workarounds in pyisolate.

## Tensor Handling Edge Cases

### 1. Re-sharing IPC Tensors

**Scenario**: A tensor received via CUDA IPC cannot be re-shared to another process.

**Behavior**: PyTorch raises `RuntimeError: received from another process`

**Handling**: PyIsolate automatically clones the tensor:
```python
# In tensor_serializer.py
if "received from another process" in str(e):
    tensor_size_mb = t.numel() * t.element_size() / (1024 * 1024)
    if tensor_size_mb > 100:
        logger.warning("PERFORMANCE: Cloning large CUDA tensor...")
    t = t.clone()  # Clone to make shareable
```

**Impact**: Performance penalty for large tensors. Design nodes to avoid returning unmodified input tensors.

### 2. Shared Memory File Deletion Race

**Scenario**: Tensor's shared memory file deleted before receiver opens it.

**Behavior**: `FileNotFoundError` on deserialization.

**Handling**: `TensorKeeper` holds tensor references for 5.0 seconds:
```python
class TensorKeeper:
    def __init__(self, retention_seconds: float = 5.0):
        # Keeps strong references to prevent GC
```

**Mitigation**: Increase retention for slow environments:
```python
from pyisolate._internal.tensor_serializer import _tensor_keeper
_tensor_keeper.retention_seconds = 60.0  # 60 seconds
```

### 3. Large Tensor Memory Pressure

**Scenario**: Multiple large tensors in TensorKeeper exhaust memory.

**Behavior**: Out of memory errors.

**Mitigation**:
- Process tensors in smaller batches
- Reduce TensorKeeper retention time for fast networks
- Monitor `/dev/shm` usage: `df -h /dev/shm`

## Singleton Edge Cases

### 1. Instantiation Before use_remote()

**Scenario**: Singleton instantiated before `use_remote()` called.

**Behavior**: Local instance created instead of RPC proxy.

**Impact**: Calls go to local instance, not remote service.

**Prevention**:
```python
# WRONG - creates local instance
instance = MyService()
MyService.use_remote(rpc)

# CORRECT - injects proxy first
MyService.use_remote(rpc)
instance = MyService()
```

### 2. inject_instance() After Instantiation

**Scenario**: Attempting to inject after singleton exists.

**Behavior**: `AssertionError` raised.

**Design**: This is intentional to prevent silent behavior changes:
```python
assert cls not in SingletonMetaclass._instances, (
    f"Cannot inject instance for {cls.__name__}: singleton already exists."
)
```

### 3. Nested Singleton Registration

**Scenario**: ProxiedSingleton with type-hinted singleton attributes.

**Behavior**: Both parent and nested singletons are registered:
```python
class Parent(ProxiedSingleton):
    child: Child  # Type hint triggers registration

# use_remote registers both Parent and Child
Parent.use_remote(rpc)
```

**Note**: Only type-hinted attributes (not instance attributes) trigger automatic registration.

## Sandbox Edge Cases

### 1. AppArmor Restrictions (Ubuntu)

**Scenario**: Ubuntu's AppArmor restricts bwrap.

**Behavior**: Sandbox detection returns `RestrictionModel.APPARMOR`.

**Handling**: PyIsolate runs in degraded mode without user namespace isolation.

**Detection**:
```python
from pyisolate._internal.sandbox_detect import detect_restriction_model, RestrictionModel
if detect_restriction_model() == RestrictionModel.APPARMOR:
    print("Running in degraded sandbox mode")
```

### 2. Missing /dev/shm

**Scenario**: System without /dev/shm or with limited size.

**Behavior**: Tensor serialization fails.

**Workaround**: Mount tmpfs at /dev/shm or increase its size:
```bash
sudo mount -t tmpfs -o size=4G tmpfs /dev/shm
```

### 3. Forbidden Adapter Paths

**Scenario**: Adapter provides dangerous paths like "/" or "/etc".

**Behavior**: Paths are silently rejected with warning:
```python
FORBIDDEN_ADAPTER_PATHS = frozenset({"/", "/etc", "/root", "/home", ...})
if normalized in FORBIDDEN_ADAPTER_PATHS:
    logger.warning("Adapter path '%s' rejected: would weaken sandbox security", path)
    return False
```

## RPC Edge Cases

### 1. Recursive Callbacks

**Scenario**: Callback triggers another RPC call back to extension.

**Behavior**: Supported via `parent_call_id` tracking.

**Limitation**: Deep recursion can exhaust call ID space or cause deadlocks.

**Best Practice**: Limit callback depth; use async patterns for deep nesting.

### 2. RPC During Shutdown

**Scenario**: RPC call initiated while connection is closing.

**Behavior**: Call may fail or timeout.

**Handling**: Check connection state before calls; handle gracefully.

### 3. Non-Serializable Return Values

**Scenario**: Method returns object that can't be JSON serialized.

**Behavior**: Serialization error raised.

**Handling**: Register custom serializers:
```python
from pyisolate._internal.serialization_registry import SerializerRegistry

registry = SerializerRegistry.get_instance()
registry.register(
    "MyType",
    lambda obj: {"__type__": "MyType", "data": obj.data},
    lambda d: MyType(d["data"])
)
```

## Event Loop Edge Cases

### 1. Loop Closed Between Calls

**Scenario**: Event loop closed and recreated between RPC calls.

**Behavior**: Singletons survive; RPC continues to work.

**Design**: `ProxiedSingleton` instances are resilient to loop recreation:
```python
# Test from test_rpc_contract.py
def test_singleton_survives_loop_recreation(self):
    loop1 = asyncio.new_event_loop()
    asyncio.set_event_loop(loop1)
    registry = MockRegistry()
    obj_id = registry.register("loop1_object")
    loop1.close()

    loop2 = asyncio.new_event_loop()
    asyncio.set_event_loop(loop2)
    result = registry.get(obj_id)  # Still works
    assert result == "loop1_object"
```

### 2. Multiple Event Loops

**Scenario**: Multiple threads with their own event loops.

**Behavior**: Each AsyncRPC instance tracks its loop via context variables.

**Note**: `calling_loop` in `RPCPendingRequest` ensures responses route correctly.

## Platform-Specific Edge Cases

### 1. macOS Limitations

**Scenario**: macOS doesn't support Linux namespaces.

**Behavior**: Sandbox mode unavailable; falls back to non-isolated execution.

**Detection**: `SandboxMode.DISABLED` on macOS.

### 2. Docker Constraints

**Scenario**: Running inside Docker container.

**Behavior**: May need `--privileged` or specific capabilities for user namespaces.

**Check**:
```bash
# Inside container
capsh --print | grep cap_sys_admin
```

### 3. WSL2 Limitations

**Scenario**: Windows Subsystem for Linux.

**Behavior**: Some namespace features may be restricted depending on WSL version.

**Workaround**: Use latest WSL2 with updated kernel.

## Best Practices for Edge Cases

1. **Always check restriction model** before assuming full sandbox capability
2. **Handle RPC errors gracefully** - network issues can cause timeouts
3. **Avoid returning large unmodified tensors** - triggers expensive cloning
4. **Call use_remote() early** - before any singleton instantiation
5. **Monitor /dev/shm usage** - especially with many large tensors
6. **Test with debug logging** - `PYISOLATE_DEBUG_RPC=1` reveals communication issues
