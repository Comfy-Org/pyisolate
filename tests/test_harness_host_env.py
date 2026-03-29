from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.harness.host import ReferenceHost


@pytest.mark.asyncio
async def test_reference_host_cleanup_restores_tmpdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_tmpdir = tmp_path / "original-tmpdir"
    original_tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(original_tmpdir))

    host = ReferenceHost()

    assert os.environ["TMPDIR"] != str(original_tmpdir)

    await host.cleanup()

    assert os.environ["TMPDIR"] == str(original_tmpdir)
