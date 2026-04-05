# RPC Protocol Specification

This document specifies the Remote Procedure Call (RPC) protocol used by pyisolate for inter-process communication between host and isolated extension processes.

## Overview

PyIsolate uses a bidirectional JSON-based RPC protocol over Unix Domain Sockets (UDS) or multiprocessing queues. The protocol supports:

- Synchronous method calls with async execution
- Nested/recursive calls (callbacks from extension to host)
- Zero-copy tensor transfer via shared memory references
- Error propagation with remote tracebacks

## Message Types

All messages are JSON objects with a `kind` field indicating the message type.

### RPCRequest (kind: "call")

Initiates a method call on a remote object.

```json
{
  "kind": "call",
  "call_id": 1,
  "object_id": "ModelRegistry",
  "method": "get_model",
  "args": ["model_name"],
  "kwargs": {"version": "latest"},
  "parent_call_id": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `kind` | `"call"` | Message type identifier |
| `call_id` | `int` | Unique identifier for this call |
| `object_id` | `str` | Remote object identifier (typically class name) |
| `method` | `str` | Method name to invoke |
| `args` | `list` | Positional arguments |
| `kwargs` | `dict` | Keyword arguments |
| `parent_call_id` | `int \| null` | For nested calls, the parent's call_id |

### RPCResponse (kind: "response")

Response to a method call.

```json
{
  "kind": "response",
  "call_id": 1,
  "result": {"model_id": "abc123", "loaded": true},
  "error": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `kind` | `"response"` | Message type identifier |
| `call_id` | `int` | Matching call_id from request |
| `result` | `any` | Return value (null if error) |
| `error` | `str \| null` | Error message if call failed |

### RPCCallback (kind: "callback")

Callback from extension back to host during a method execution.

```json
{
  "kind": "callback",
  "callback_id": "progress_callback_0",
  "call_id": 2,
  "parent_call_id": 1,
  "args": [0.5],
  "kwargs": {}
}
```

| Field | Type | Description |
|-------|------|-------------|
| `kind` | `"callback"` | Message type identifier |
| `callback_id` | `str` | Callback identifier |
| `call_id` | `int` | Unique identifier for this callback |
| `parent_call_id` | `int` | The call_id of the method that initiated this callback |
| `args` | `list` | Positional arguments |
| `kwargs` | `dict` | Keyword arguments |

### RPCError (kind: "error")

Explicit error message (alternative to error field in response).

```json
{
  "kind": "error",
  "call_id": 1,
  "error": "ValueError: Invalid model name",
  "traceback": "Traceback (most recent call last):\n  ..."
}
```

### RPCStop (kind: "stop")

Signal to terminate the RPC connection.

```json
{
  "kind": "stop",
  "reason": "shutdown"
}
```

## Request/Response Lifecycle

### Simple Call Flow

```
Host                          Extension
  |                               |
  |  RPCRequest (call_id=1)       |
  |------------------------------>|
  |                               |  Execute method
  |                               |
  |  RPCResponse (call_id=1)      |
  |<------------------------------|
  |                               |
```

### Nested Call Flow (Callback)

```
Host                          Extension
  |                               |
  |  RPCRequest (call_id=1)       |
  |------------------------------>|
  |                               |  Start method
  |  RPCCallback (call_id=2,      |
  |    parent_call_id=1)          |
  |<------------------------------|
  |                               |
  |  Execute callback             |
  |                               |
  |  RPCResponse (call_id=2)      |
  |------------------------------>|
  |                               |  Continue method
  |  RPCResponse (call_id=1)      |
  |<------------------------------|
  |                               |
```

## Error Handling

### Error Propagation

When an exception occurs in the remote process:

1. Exception is caught and serialized
2. Remote traceback is captured
3. RPCResponse sent with `error` field populated
4. Host receives response and raises exception locally
5. Remote traceback is attached for debugging

### Error Response Format

```json
{
  "kind": "response",
  "call_id": 1,
  "result": null,
  "error": "ValueError: Model 'unknown' not found"
}
```

The host reconstructs the exception type from the error string prefix (e.g., `ValueError:`) and raises it with the remote traceback attached as `__pyisolate_remote_traceback__`.

## Tensor Serialization

PyTorch tensors are not serialized directly. Instead, they are converted to `TensorRef` references that point to shared memory:

### CPU Tensor Reference

```json
{
  "__type__": "TensorRef",
  "device": "cpu",
  "strategy": "file_system",
  "manager_path": "/dev/shm/torch_xxx",
  "storage_key": "abc123",
  "storage_size": 4096,
  "dtype": "torch.float32",
  "tensor_size": [2, 3, 4],
  "tensor_stride": [12, 4, 1],
  "tensor_offset": 0,
  "requires_grad": false
}
```

### CUDA Tensor Reference

```json
{
  "__type__": "TensorRef",
  "device": "cuda",
  "device_idx": 0,
  "handle": "<base64-encoded CUDA IPC handle>",
  "storage_size": 4096,
  "storage_offset": 0,
  "dtype": "torch.float32",
  "tensor_size": [2, 3, 4],
  "tensor_stride": [12, 4, 1],
  "tensor_offset": 0,
  "requires_grad": false,
  "ref_counter_handle": "<base64>",
  "ref_counter_offset": 0,
  "event_handle": "<base64>",
  "event_sync_required": true
}
```

## Transport Layer

The primary transport is `JSONSocketTransport` — length-prefixed JSON over Unix Domain Sockets (or TCP on Windows). All isolation modes use this transport.

### JSONSocketTransport (Primary)

Uses length-prefixed JSON-RPC over raw sockets. No pickle. This is the standard transport for all Linux isolation modes (sandboxed and non-sandboxed) and Windows TCP mode.

### QueueTransport (Legacy)

Uses `multiprocessing.Queue` for communication. This is a legacy backward-compatibility path and is not used in current isolation modes.

### Transport Interface

```python
class RPCTransport(Protocol):
    def send(self, message: RPCMessage) -> None:
        """Send message to remote endpoint."""
        ...

    def recv(self, timeout: float | None = None) -> RPCMessage | None:
        """Receive message from remote endpoint."""
        ...

    def close(self) -> None:
        """Close the transport."""
        ...
```

## ProxiedSingleton Pattern

The `ProxiedSingleton` metaclass enables transparent RPC by:

1. Maintaining a singleton registry of instances
2. Injecting RPC caller proxies via `use_remote()`
3. Supporting `@local_execution` for methods that run locally

### Registration Flow

```python
# Host side
class ModelRegistry(ProxiedSingleton):
    def get_model(self, name: str) -> Model:
        ...

# Extension side
ModelRegistry.use_remote(rpc)  # Injects proxy
registry = ModelRegistry()      # Returns proxy
result = registry.get_model("x")  # RPC call to host
```

## Security Considerations

1. **Message Validation**: All incoming messages are validated against expected TypedDicts
2. **Object ID Whitelisting**: Only registered object_ids can be called
3. **No Code Execution**: RPC only invokes pre-registered methods
4. **Sandbox Isolation**: Transport layer works within bubblewrap sandbox constraints
