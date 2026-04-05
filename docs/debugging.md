# Debugging Guide

This guide covers common debugging scenarios and troubleshooting techniques for pyisolate.

## Environment Variables

### PYISOLATE_DEBUG_RPC

Enable verbose RPC message logging:

```bash
export PYISOLATE_DEBUG_RPC=1
```

This logs all RPC messages sent and received, useful for diagnosing communication issues.

### PYISOLATE_ENABLE_CUDA_IPC

Enable CUDA IPC for zero-copy GPU tensor transfer:

```bash
export PYISOLATE_ENABLE_CUDA_IPC=1
```

Without this, CUDA tensors are copied to CPU for transfer.

### PYISOLATE_CHILD

Set automatically in child processes. Check this to determine if running in sandbox:

```python
import os
if os.environ.get("PYISOLATE_CHILD") == "1":
    print("Running in isolated sandbox")
```

## Common Issues

### "No such file or directory" for Shared Memory

**Symptom**: `RPC recv failed ... No such file or directory`

**Cause**: Tensor was garbage collected before remote process could access shared memory.

**Solution**: The `TensorKeeper` class holds references for 5.0 seconds by default. If you see this error:
1. Increase `TensorKeeper.retention_seconds` for slow networks
2. Ensure tensors aren't being explicitly deleted too early
3. Check that `/dev/shm` has sufficient space

### Sandbox Launch Fails

**Symptom**: Extension fails to start with bwrap errors

**Diagnosis**:
```bash
# Check if bwrap is available
which bwrap

# Check user namespace restrictions
cat /proc/sys/kernel/unprivileged_userns_clone
# Should be 1 for full isolation

# Check AppArmor restrictions (Ubuntu)
aa-status | grep bwrap
```

**Solutions**:
1. Install bubblewrap: `apt install bubblewrap`
2. Enable unprivileged user namespaces: `sysctl kernel.unprivileged_userns_clone=1`
3. For AppArmor issues, see [sandbox_detect.py](../pyisolate/_internal/sandbox_detect.py)

### RPC Timeout

**Symptom**: RPC calls hang or timeout

**Diagnosis**:
```python
import logging
logging.getLogger("pyisolate").setLevel(logging.DEBUG)
```

Check for:
1. Deadlocks in callback chains
2. Extension process crashed (check process status)
3. Socket connection issues

**Solutions**:
1. Check extension logs for exceptions
2. Verify socket path is accessible
3. Ensure no circular RPC calls

### Singleton Not Found

**Symptom**: `KeyError: <class 'MySingleton'>`

**Cause**: Singleton accessed before `use_remote()` was called.

**Solution**: Ensure `use_remote()` is called before any instantiation:
```python
# Correct order
MySingleton.use_remote(rpc)
instance = MySingleton()

# Wrong order - will fail
instance = MySingleton()  # Creates local instance
MySingleton.use_remote(rpc)  # Too late!
```

### CUDA IPC Failures

**Symptom**: CUDA tensors fail to transfer between processes

**Diagnosis**:
```python
import torch
# Check CUDA IPC support
print(torch.cuda.is_available())
print(torch.version.cuda)

# Test IPC handle creation
t = torch.zeros(10, device='cuda')
try:
    import torch.multiprocessing.reductions as r
    func, args = r.reduce_tensor(t)
    print("CUDA IPC supported")
except Exception as e:
    print(f"CUDA IPC failed: {e}")
```

**Common causes**:
1. Different CUDA versions between processes
2. Tensor received from another process (can't re-share)
3. CUDA driver issues

**Solutions**:
1. Clone tensors that were received via IPC before re-sharing
2. Ensure same CUDA/driver version in all processes
3. Fall back to CPU transfer: unset `PYISOLATE_ENABLE_CUDA_IPC`

## Logging Configuration

### Enable Debug Logging

```python
import logging

# Enable all pyisolate debug logging
logging.getLogger("pyisolate").setLevel(logging.DEBUG)

# Enable specific module logging
logging.getLogger("pyisolate._internal.rpc_protocol").setLevel(logging.DEBUG)
logging.getLogger("pyisolate._internal.sandbox").setLevel(logging.DEBUG)
```

### Pytest Debug Options

```bash
# Enable debug logging during tests
pytest --debug-pyisolate

# Log to file
pytest --pyisolate-log-file=debug.log

# Verbose output
pytest -v --tb=long
```

## Inspecting RPC State

### Check Registered Singletons

```python
from pyisolate._internal.rpc_protocol import SingletonMetaclass

# List all registered singletons
for cls, instance in SingletonMetaclass._instances.items():
    print(f"{cls.__name__}: {type(instance)}")
```

### Check Pending RPC Calls

```python
# In AsyncRPC instance
print(f"Pending calls: {len(rpc._pending_calls)}")
for call_id, pending in rpc._pending_calls.items():
    print(f"  {call_id}: {pending['method']} on {pending['object_id']}")
```

## Sandbox Debugging

### Inspect Sandbox Command

```python
from pyisolate._internal.sandbox import build_bwrap_command
from pyisolate._internal.sandbox_detect import detect_restriction_model

restriction = detect_restriction_model()
cmd = build_bwrap_command(
    python_exe="/path/to/python",
    module_path="/path/to/extension",
    venv_path="/path/to/venv",
    uds_address="/tmp/socket",
    allow_gpu=True,
    restriction_model=restriction
)
print(" ".join(cmd))
```

### Test Sandbox Manually

```bash
# Run sandbox command manually to see errors
bwrap --ro-bind /usr /usr --dev /dev --proc /proc \
    --ro-bind /path/to/venv /path/to/venv \
    /path/to/python -c "print('Hello from sandbox')"
```

## Memory Debugging

### Track Tensor References

```python
from pyisolate._internal.tensor_serializer import _tensor_keeper

# Check how many tensors are being kept
print(f"Tensors in keeper: {len(_tensor_keeper._keeper)}")
```

### Detect Memory Leaks

```python
import gc
import weakref

# Track singleton garbage collection
from pyisolate._internal.rpc_protocol import SingletonMetaclass

class MyService(ProxiedSingleton):
    pass

instance = MyService()
ref = weakref.ref(instance)

del instance
SingletonMetaclass._instances.clear()
gc.collect()

if ref() is None:
    print("Properly collected")
else:
    print("Memory leak detected!")
```

## Getting Help

1. Enable debug logging and capture output
2. Include pyisolate version: `python -c "import pyisolate; print(pyisolate.__version__)"`
3. Include Python version and platform info
4. Check existing issues at: https://github.com/anthropics/claude-code/issues
