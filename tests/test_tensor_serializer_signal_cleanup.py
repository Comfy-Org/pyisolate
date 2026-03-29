from __future__ import annotations

import importlib
import signal


def test_signal_cleanup_handler_tolerates_missing_sighup(monkeypatch) -> None:
    import pyisolate._internal.tensor_serializer as tensor_serializer

    monkeypatch.setenv("PYISOLATE_SIGNAL_CLEANUP", "1")
    monkeypatch.delattr(signal, "SIGHUP", raising=False)

    installed: list[object] = []

    def fake_signal(sig, _handler):
        installed.append(sig)

    monkeypatch.setattr(signal, "signal", fake_signal)

    importlib.reload(tensor_serializer)

    assert installed == [signal.SIGTERM]
