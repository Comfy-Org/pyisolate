from __future__ import annotations

import pytest

from pyisolate._internal.tensor_serializer import _deserialize_json_tensor


def test_deserialize_json_tensor_rejects_unknown_dtype_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown TensorValue dtypes should fail fast in torchless workers."""

    def fail_require_torch(context: str) -> None:  # noqa: ARG001
        raise RuntimeError("torch unavailable")

    monkeypatch.setattr("pyisolate._internal.tensor_serializer.require_torch", fail_require_torch)

    with pytest.raises(ValueError, match="Unsupported TensorValue dtype"):
        _deserialize_json_tensor(
            {
                "dtype": "torch.complex64",
                "data": [1.0, 2.0],
                "tensor_size": [2],
                "requires_grad": False,
            }
        )
