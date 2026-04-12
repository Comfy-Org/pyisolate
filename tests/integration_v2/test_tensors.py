from typing import Any
import gc

import pytest
import torch  # noqa: E402

try:
    import numpy as np  # noqa: F401

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@pytest.mark.asyncio
async def test_tensor_roundtrip_cpu(reference_host: Any) -> None:
    """
    Verify sending a CPU tensor to the child and getting it back.
    """
    print("\n[TEST] Starting CPU tensor roundtrip")
    ext = reference_host.load_test_extension("tensor_cpu", isolated=True)
    proxy = ext.get_proxy()

    # Create tensor
    t = torch.ones(5, 5)
    print(f"[TEST] Created tensor: {t.shape}")

    # Roundtrip
    print("[TEST] Sending tensor...")
    result = await proxy.echo_tensor(t)
    print("[TEST] Tensor echoed back.")

    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, t)
    print("[TEST] CPU tensor verification passed.")
    # Check if storage is shared or copied?
    # ReferenceHost usually uses file_system strategy for CPU.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.asyncio
async def test_cuda_allocation(reference_host: Any) -> None:
    """
    Verify child can allocate CUDA memory and return meta-data.
    """
    print("\n[TEST] Starting CUDA allocation test")
    ext = reference_host.load_test_extension("tensor_cuda", isolated=True)
    proxy = ext.get_proxy()

    # Allocate 10MB
    print("[TEST] Requesting allocation...")
    info = await proxy.allocate_cuda(10)
    print(f"[TEST] Allocation info: {info}")

    assert "device" in info
    assert "cuda" in info["device"]
    assert info["allocated_bytes"] >= 10 * 1024 * 1024
    del info
    gc.collect()
    torch.cuda.synchronize()
    print("[TEST] CUDA allocation verified.")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.asyncio
async def test_tensor_roundtrip_cuda(reference_host: Any) -> None:
    """
    Verify sending a CUDA tensor. Requires CUDA IPC if isolated.
    """
    print("\n[TEST] Starting CUDA IPC roundtrip")
    ext = reference_host.load_test_extension("tensor_cuda_ipc", isolated=True)
    proxy = ext.get_proxy()

    t = torch.ones(5, 5, device="cuda")
    print(f"[TEST] Created CUDA tensor: {t.shape}, device={t.device}")

    # Roundtrip
    print("[TEST] Sending tensor...")
    result = await proxy.echo_tensor(t)
    print(f"[TEST] Received tensor: device={result.device}")

    assert isinstance(result, torch.Tensor)
    assert result.device.type == "cuda"
    assert torch.equal(result.cpu(), t.cpu())
    del result
    del t
    gc.collect()
    torch.cuda.synchronize()
    print("[TEST] CUDA IPC verified.")
