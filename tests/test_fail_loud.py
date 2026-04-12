from typing import Any
import pytest

from pyisolate._internal import bootstrap


def test_bootstrap_malformed_snapshot_fails(monkeypatch: Any) -> None:
    """Test that a malformed JSON snapshot raises ValueError."""
    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", "{invalid_json")

    with pytest.raises(ValueError, match="Failed to decode PYISOLATE_HOST_SNAPSHOT"):
        bootstrap.bootstrap_child()


def test_bootstrap_missing_adapter_ref_fails(monkeypatch: Any) -> None:
    """Test that valid JSON without adapter_ref returns None (no adapter loaded)."""
    # If no adapter_ref is present, bootstrap returns None, it doesn't fail unless
    # adapter_ref WAS present but failed to load.
    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", '{"sys_path": []}')

    adapter = bootstrap.bootstrap_child()
    assert adapter is None


def test_bootstrap_bad_adapter_ref_fails(monkeypatch: Any) -> None:
    """Test that a valid snapshot with a bad adapter_ref logs a warning
    but might not crash unless critical logic depends on it.
    """
    # The current logic in bootstrap.py catches Exception and logs a warning for rehydration failures.
    # It then raises ValueError if "snapshot contained adapter info but adapter could not be loaded".

    monkeypatch.setenv("PYISOLATE_HOST_SNAPSHOT", '{"adapter_ref": "bad.module:BadClass"}')

    # We expect a ValueError because adapter_ref was provided but failed to load
    with pytest.raises(ValueError, match="Snapshot contained adapter info but adapter could not be loaded"):
        bootstrap.bootstrap_child()
