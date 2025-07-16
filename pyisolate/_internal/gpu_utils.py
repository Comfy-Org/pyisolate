"""
pyisolate._internal.gpu_utils

GPU/XPU/torch-specific utilities for tensor serialization, DLPack conversion, and device handling.
These functions require torch (and sometimes numpy) to be installed.
"""

def maybe_to_dlpack(obj):
    """Convert XPU tensor to DLPack capsule if needed (requires torch)."""
    try:
        import torch
        from torch.utils import dlpack as _dlpack  # type: ignore[attr-defined]
    except ImportError as e:
        raise ImportError("pyisolate: 'torch' is required for maybe_to_dlpack but is not installed.") from e
    if isinstance(obj, torch.Tensor) and hasattr(obj, "device") and obj.device.type == "xpu":
        # If the input is a NumPy array and not writable, make it writable before converting
        if hasattr(obj, "numpy"):
            arr = obj.numpy()
            if not arr.flags.writeable:
                arr = arr.copy()
            return torch.from_numpy(arr).to("xpu")
        return _dlpack.to_dlpack(obj)  # type: ignore[attr-defined]
    return obj

def maybe_from_dlpack(obj):
    """Convert DLPack capsule to XPU tensor if needed (requires torch)."""
    try:
        import torch
        from torch.utils import dlpack as _dlpack  # type: ignore[attr-defined]
    except ImportError as e:
        raise ImportError("pyisolate: 'torch' is required for maybe_from_dlpack but is not installed.") from e
    # DLPack capsules are PyCapsule, not torch.Tensor
    if not isinstance(obj, torch.Tensor) and hasattr(obj, "__dlpack__"):
        return _dlpack.from_dlpack(obj)  # type: ignore[attr-defined]
    # For raw PyCapsule (older PyTorch), try fallback
    if type(obj).__name__ == "PyCapsule":
        return _dlpack.from_dlpack(obj)  # type: ignore[attr-defined]
    return obj

def maybe_serialize_tensor(obj):
    """Serialize XPU tensor for transport (requires torch)."""
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "pyisolate: 'torch' is required for maybe_serialize_tensor but is not installed."
        ) from e
    if isinstance(obj, torch.Tensor) and hasattr(obj, "device") and obj.device.type == "xpu":
        # Fallback: send as CPU buffer + metadata
        arr = obj.cpu().numpy()
        return ("xpu_tensor", arr.tobytes(), arr.shape, str(arr.dtype))
    return obj

def maybe_deserialize_tensor(obj):
    """Deserialize XPU tensor from transport (requires torch and numpy)."""
    try:
        import numpy as np
        import torch
    except ImportError as e:
        raise ImportError(
            "pyisolate: 'torch' and 'numpy' are required for maybe_deserialize_tensor but are not installed."
        ) from e
    if isinstance(obj, tuple) and len(obj) == 4 and obj[0] == "xpu_tensor":
        _, buf, shape, dtype = obj
        arr = np.frombuffer(buf, dtype=dtype).reshape(shape)
        return torch.from_numpy(arr).to("xpu")
    return obj
