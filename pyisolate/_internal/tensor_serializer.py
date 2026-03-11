import atexit
import base64
import collections
import contextlib
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

from .torch_gate import require_torch

logger = logging.getLogger(__name__)

# Minimum /dev/shm space required for tensor serialization (100MB)
MIN_SHM_SPACE_BYTES = 100 * 1024 * 1024

# Cache for /dev/shm availability check
_shm_available: bool | None = None
_shm_check_lock = threading.Lock()


def _check_shm_availability() -> bool:
    """Check if /dev/shm is available and has sufficient space.

    Returns:
        True if /dev/shm is usable, False otherwise.
    """
    global _shm_available

    with _shm_check_lock:
        if _shm_available is not None:
            return _shm_available

        shm_path = Path("/dev/shm")

        # Check if /dev/shm exists
        if not shm_path.exists():
            logger.warning(
                "/dev/shm not found. Tensor serialization may use slower file-based "
                "fallback. Consider mounting tmpfs at /dev/shm for better performance."
            )
            _shm_available = False
            return False

        # Check if we can write to it
        if not os.access(shm_path, os.W_OK):
            logger.warning(
                "/dev/shm is not writable. Tensor serialization may use slower "
                "fallback. Check permissions on /dev/shm."
            )
            _shm_available = False
            return False

        # Check available space
        try:
            stat = os.statvfs(shm_path)
            available_bytes = stat.f_bavail * stat.f_frsize
            if available_bytes < MIN_SHM_SPACE_BYTES:
                logger.warning(
                    "/dev/shm has low available space (%.1f MB). Large tensor "
                    "serialization may fail. Consider increasing /dev/shm size.",
                    available_bytes / (1024 * 1024),
                )
                # Still usable, just warn
        except OSError as e:
            logger.debug("Failed to check /dev/shm space: %s", e)

        _shm_available = True
        return True


def _reset_shm_check() -> None:
    """Reset the cached /dev/shm check (for testing)."""
    global _shm_available
    with _shm_check_lock:
        _shm_available = None


# ---------------------------------------------------------------------------
# Tensor Lifecycle Management
# ---------------------------------------------------------------------------


class TensorKeeper:
    """
    Keeps strong references to serialized tensors for a short window to prevent
    premature garbage collection and shared-memory file deletion before the
    remote side has a chance to open it.

    This fixes the 'RPC recv failed ... No such file or directory' race condition.
    """

    dest = ("TensorKeeper",)

    def __init__(self, retention_seconds: float = 5.0):
        self.retention_seconds = retention_seconds
        self._keeper: collections.deque = collections.deque()
        self._lock = threading.Lock()

    def keep(self, t: Any) -> None:
        now = time.time()
        with self._lock:
            self._keeper.append((now, t))
            logger.debug(
                f"TensorKeeper: KEEPING tensor {t.shape} (Total kept: {len(self._keeper)}). id={id(t)}"
            )

            # Cleanup old

            while self._keeper:
                timestamp, _ = self._keeper[0]
                if now - timestamp > self.retention_seconds:
                    self._keeper.popleft()
                else:
                    break

    def flush(self) -> int:
        """Release all currently held tensor references immediately."""
        with self._lock:
            count = len(self._keeper)
            self._keeper.clear()
            return count


_tensor_keeper = TensorKeeper()


def _flush_reduction_shared_cache(reductions: Any) -> None:
    shared_cache = getattr(reductions, "shared_cache", None)
    if shared_cache is None:
        return

    try:
        free_dead = getattr(shared_cache, "free_dead_references", None)
        if callable(free_dead):
            free_dead()
        clear_fn = getattr(shared_cache, "clear", None)
        if callable(clear_fn):
            clear_fn()
    except Exception:
        logger.debug("TensorKeeper flush: failed to purge shared_cache", exc_info=True)


def _flush_cuda_ipc(torch: Any) -> None:
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.ipc_collect()
    except Exception:
        logger.debug("TensorKeeper flush: cuda ipc_collect failed", exc_info=True)


def flush_tensor_keeper() -> int:
    """Release all tensors held by TensorKeeper and return the release count."""
    released = _tensor_keeper.flush()
    try:
        torch, reductions = require_torch("flush_tensor_keeper")
    except Exception:
        return released

    _flush_reduction_shared_cache(reductions)
    _flush_cuda_ipc(torch)
    return released


def purge_orphan_sender_shm_files(min_age_seconds: float = 1.0, force: bool = False) -> int:
    """Best-effort unlink of stale sender-side torch_* shm files for this PID.

    Guarded by PYISOLATE_PURGE_SENDER_SHM=1 to keep default behavior unchanged.
    """
    if not force and os.environ.get("PYISOLATE_PURGE_SENDER_SHM", "0") != "1":
        return 0

    shm_root = Path("/dev/shm")
    if not shm_root.exists():
        return 0

    now = time.time()
    prefix = f"torch_{os.getpid()}_"
    removed = 0
    for path in shm_root.glob(f"{prefix}*"):
        try:
            if min_age_seconds > 0:
                mtime = path.stat().st_mtime
                if (now - mtime) < min_age_seconds:
                    continue
            path.unlink()
            removed += 1
        except FileNotFoundError:
            continue
        except Exception:
            logger.debug("Failed to purge stale SHM file %s", path, exc_info=True)
    return removed


def _flush_tensor_keeper_on_exit() -> None:
    # purge_orphan_sender_shm_files is intentionally NOT called here.
    # Calling it with force=True races with the C++ RefcountedMapAllocator
    # destructor: SIGKILL'd child peers leave the refcount file at count > 1,
    # so the host's C++ close() decrements but does NOT unlink (refcount still > 0).
    # purge_orphan would then unlink the file while the C++ mmap is still open,
    # causing a double-unlink → ENOENT → SIGABRT at process exit.
    # Files are cleaned up by the C++ destructor when the last consumer closes.
    with contextlib.suppress(Exception):
        flush_tensor_keeper()


atexit.register(_flush_tensor_keeper_on_exit)


def _install_signal_cleanup_handlers() -> None:
    """Optional signal cleanup for harnesses that terminate via SIGHUP/SIGTERM."""
    if os.environ.get("PYISOLATE_SIGNAL_CLEANUP", "0") != "1":
        return

    def _handler(signum: int, _frame: Any) -> None:
        try:
            _flush_tensor_keeper_on_exit()
        finally:
            os._exit(128 + signum)

    for sig in (signal.SIGHUP, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            logger.debug("Failed to install signal cleanup handler for %s", sig, exc_info=True)


_install_signal_cleanup_handlers()


def serialize_tensor(t: Any) -> dict[str, Any]:
    """Serialize a tensor to JSON-compatible format using shared memory."""
    torch, _ = require_torch("serialize_tensor")
    if t.is_cuda:
        return _serialize_cuda_tensor(t)
    return _serialize_cpu_tensor(t)


def _serialize_cpu_tensor(t: Any) -> dict[str, Any]:
    """Serialize CPU tensor using file_system shared memory strategy.

    Falls back gracefully if /dev/shm is unavailable, though performance
    may be reduced.
    """
    torch, reductions = require_torch("CPU tensor serialization")

    # Keep strategy pinned to file_system for JSON-RPC transfer paths.
    # A fallback from file_descriptor -> file_system can leave behind SHM refs.
    try:
        if torch.multiprocessing.get_sharing_strategy() != "file_system":
            torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        logger.debug("Failed to enforce file_system sharing strategy", exc_info=True)

    # Check /dev/shm availability (cached after first check)
    _check_shm_availability()

    # Warn for large tensors that may impact performance
    tensor_size_mb = t.numel() * t.element_size() / (1024 * 1024)
    if tensor_size_mb > 500:  # 500MB threshold
        logger.warning(
            "PERFORMANCE: Serializing large CPU tensor (%.1f MB). Consider using "
            "CUDA tensors with PYISOLATE_ENABLE_CUDA_IPC=1 for zero-copy transfer.",
            tensor_size_mb,
        )
    elif tensor_size_mb > 100:
        logger.debug(
            "Serializing medium-sized CPU tensor (%.1f MB) via shared memory.",
            tensor_size_mb,
        )

    # Ensure the tensor is kept alive on this side until the remote side can open it.
    # Without this, the tensor might be garbage collected immediately after serialization returns,
    # causing the underlying shared memory file to be deleted before the receiver opens it.
    _tensor_keeper.keep(t)

    if not t.is_shared():
        t.share_memory_()

    storage = t.untyped_storage()
    sfunc, sargs = reductions.reduce_storage(storage)

    if sfunc.__name__ == "rebuild_storage_filename":
        use_borrowed = os.environ.get("PYISOLATE_CPU_BORROWED_SHM", "1") == "1"
        strategy = "file_system_borrowed" if use_borrowed else "file_system"
        if use_borrowed:
            # reduce_storage() increments sender-side refcount for transfer.
            # Undo transit incref immediately and use borrowed strategy payload.
            storage._shared_decref()
        # sargs: (cls, manager_path, storage_key, size)
        return {
            "__type__": "TensorRef",
            "device": "cpu",
            "strategy": strategy,
            "manager_path": sargs[1].decode("utf-8"),
            "storage_key": sargs[2].decode("utf-8"),
            "storage_size": sargs[3],
            "dtype": str(t.dtype),
            "tensor_size": list(t.size()),
            "tensor_stride": list(t.stride()),
            "tensor_offset": t.storage_offset(),
            "requires_grad": t.requires_grad,
        }
    elif sfunc.__name__ == "rebuild_storage_fd":
        # Force file_system strategy for JSON-RPC compatibility
        torch.multiprocessing.set_sharing_strategy("file_system")
        t.share_memory_()
        return _serialize_cpu_tensor(t)
    else:
        raise RuntimeError(f"Unsupported storage reduction: {sfunc.__name__}")


def _serialize_cuda_tensor(t: Any) -> dict[str, Any]:
    """Serialize CUDA tensor using CUDA IPC."""
    _, reductions = require_torch("CUDA tensor serialization")
    try:
        func, args = reductions.reduce_tensor(t)
    except RuntimeError as e:
        if "received from another process" in str(e):
            # This tensor was received via IPC and can't be re-shared.
            # This typically happens when a node returns an unmodified input tensor.
            # Clone is required but expensive for large tensors.
            tensor_size_mb = t.numel() * t.element_size() / (1024 * 1024)
            import logging

            logger = logging.getLogger(__name__)

            if tensor_size_mb > 100:  # 100MB threshold
                logger.warning(
                    "PERFORMANCE: Cloning large CUDA tensor (%.1fMB) received from another process. "
                    "Consider modifying the node to avoid returning unmodified input tensors.",
                    tensor_size_mb,
                )
            else:
                logger.debug("Cloning CUDA tensor (%.2fMB) received from another process", tensor_size_mb)

            t = t.clone()
            func, args = reductions.reduce_tensor(t)
        else:
            raise

    # Ensure the tensor is kept alive on this side until the remote side can open it.
    _tensor_keeper.keep(t)

    # args: (cls, size, stride, offset, storage_type, dtype, device_idx, handle, storage_size,
    #        storage_offset, requires_grad, ref_counter_handle, ref_counter_offset,
    #        event_handle, event_sync_required)
    return {
        "__type__": "TensorRef",
        "device": "cuda",
        "device_idx": args[6],  # int device index
        "tensor_size": list(args[1]),
        "tensor_stride": list(args[2]),
        "tensor_offset": args[3],
        "dtype": str(args[5]),
        "handle": base64.b64encode(args[7]).decode("ascii"),
        "storage_size": args[8],
        "storage_offset": args[9],
        "requires_grad": args[10],
        "ref_counter_handle": base64.b64encode(args[11]).decode("ascii"),
        "ref_counter_offset": args[12],
        "event_handle": base64.b64encode(args[13]).decode("ascii") if args[13] else None,
        "event_sync_required": args[14],
    }


def deserialize_tensor(data: dict[str, Any]) -> Any:
    """Deserialize a tensor from TensorRef format."""
    torch, _ = require_torch("deserialize_tensor")
    # If this is already a tensor (e.g., passed through by shared memory), return as-is
    if isinstance(data, torch.Tensor):
        return data
    # All formats now use TensorRef
    return _deserialize_legacy_tensor(data)


def _convert_lists_to_tuples(obj: Any) -> Any:
    """Recursively convert lists to tuples (PyTorch requires tuples for size/stride)."""
    if isinstance(obj, list):
        return tuple(_convert_lists_to_tuples(item) for item in obj)
    if isinstance(obj, dict):
        return {k: _convert_lists_to_tuples(v) for k, v in obj.items()}
    return obj


def _deserialize_legacy_tensor(data: dict[str, Any]) -> Any:
    """Handle legacy TensorRef format for backward compatibility."""
    torch, reductions = require_torch("legacy tensor deserialization")
    device = data["device"]
    dtype_str = data["dtype"]
    dtype = getattr(torch, dtype_str.split(".")[-1])

    if device == "cpu":
        strategy = data.get("strategy")
        if strategy not in ("file_system", "file_system_borrowed"):
            raise RuntimeError(f"Unsupported CPU strategy: {strategy}")

        manager_path = data["manager_path"].encode("utf-8")
        storage_key = data["storage_key"].encode("utf-8")
        storage_size = data["storage_size"]

        if strategy == "file_system_borrowed":
            # Rebuild directly without caching the borrowed storage handle in shared_cache.
            # Keeping borrowed refs in the global cache can pin SHM entries longer than needed.
            rebuilt_storage = torch.UntypedStorage._new_shared_filename_cpu(
                manager_path, storage_key, storage_size
            )
        else:
            # Legacy path for backward compatibility with old strategy payloads.
            rebuilt_storage = reductions.rebuild_storage_filename(
                torch.UntypedStorage, manager_path, storage_key, storage_size
            )

        # Wrap in TypedStorage (required by rebuild_tensor)
        typed_storage = torch.storage.TypedStorage(wrap_storage=rebuilt_storage, dtype=dtype, _internal=True)

        # Rebuild tensor using new signature: (cls, storage, metadata)
        # metadata is (offset, size, stride, requires_grad)
        metadata = (
            data["tensor_offset"],
            tuple(data["tensor_size"]),
            tuple(data["tensor_stride"]),
            data["requires_grad"],
        )
        cpu_tensor: Any = reductions.rebuild_tensor(  # type: ignore[assignment]
            torch.Tensor, typed_storage, metadata
        )
        # Diagnostic toggle: copy CPU tensors out of shared storage immediately.
        # If this removes residual SHM files, the remaining leak is receiver-lifetime related.
        if os.environ.get("PYISOLATE_CPU_TENSOR_FORCE_CLONE_ON_DESERIALIZE", "0") == "1":
            cloned_tensor = cpu_tensor.clone()
            if data["requires_grad"]:
                cloned_tensor.requires_grad_(True)
            cpu_tensor = cloned_tensor
        return cpu_tensor

    elif device == "cuda":
        handle = base64.b64decode(data["handle"])
        ref_counter_handle = base64.b64decode(data["ref_counter_handle"])
        event_handle = base64.b64decode(data["event_handle"]) if data["event_handle"] else None
        device_idx = data.get("device_idx", 0)  # int device index

        cuda_tensor: Any = reductions.rebuild_cuda_tensor(  # type: ignore[assignment]
            torch.Tensor,
            tuple(data["tensor_size"]),
            tuple(data["tensor_stride"]),
            data["tensor_offset"],
            torch.storage.TypedStorage,
            dtype,
            device_idx,  # int device index, not torch.device
            handle,
            data["storage_size"],
            data["storage_offset"],
            data["requires_grad"],
            ref_counter_handle,
            data["ref_counter_offset"],
            event_handle,
            data["event_sync_required"],
        )
        return cuda_tensor

    raise RuntimeError(f"Unsupported device: {device}")


def register_tensor_serializer(registry: Any) -> None:
    require_torch("register_tensor_serializer")
    # Register both "Tensor" (type name) and "torch.Tensor" (full name) just in case
    registry.register("Tensor", serialize_tensor, deserialize_tensor)
    registry.register("torch.Tensor", serialize_tensor, deserialize_tensor)
    # Also register TensorRef for deserialization
    registry.register("TensorRef", None, deserialize_tensor)
    # Register TorchReduction for recursive deserialization
    registry.register("TorchReduction", None, deserialize_tensor)

    # Register PyTorch atom types for recursive serialization
    def serialize_dtype(obj: Any) -> str:
        return str(obj)

    def deserialize_dtype(data: str) -> Any:
        import torch

        # Handle "torch.float32" -> torch.float32
        dtype_name = data.split(".")[-1]
        return getattr(torch, dtype_name)

    def serialize_device(obj: Any) -> str:
        return str(obj)

    def deserialize_device(data: str) -> Any:
        import torch

        return torch.device(data)

    def serialize_size(obj: Any) -> list:
        return list(obj)

    def deserialize_size(data: list) -> Any:
        import torch

        return torch.Size(data)

    registry.register("dtype", serialize_dtype, deserialize_dtype)
    registry.register("device", serialize_device, deserialize_device)
    registry.register("Size", serialize_size, deserialize_size)
