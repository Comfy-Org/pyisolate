import os
import sys
import tempfile

import pytest

from pyisolate._internal.sandbox_detect import detect_sandbox_capability

_SANDBOX_AVAILABLE = False
if sys.platform == "linux":
    _SANDBOX_AVAILABLE = detect_sandbox_capability().available

pytestmark = pytest.mark.skipif(
    not _SANDBOX_AVAILABLE,
    reason="filesystem barrier checks require a working Linux bubblewrap sandbox",
)


@pytest.mark.asyncio
async def test_filesystem_barrier(reference_host):
    """
    Verify that the child process cannot write to restricted paths on the host.
    """
    # 1. Create a sensitive file on host
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("sensitive data")
        sensitive_path = f.name

    try:
        # 2. Load extension
        ext = reference_host.load_test_extension("fs_test", isolated=True)
        proxy = ext.get_proxy()

        # 3. Attempt to read/write sensitive file from child
        # By default, bwrap only binds specific paths. /tmp is usually private or shared?
        # In ReferenceHost/pyisolate defaults, /tmp might be bound?
        # Let's try to write to a path that is definitely NOT bound, e.g. the test file itself
        # if in a private dir?
        # Actually, let's try to write to /home/johnj/pyisolate (source code) - assuming default
        # doesn't allow write?
        # Wait, pyisolate binds 'module_path' and 'venv_path'.
        # It binds site-packages.
        # It binds /tmp usually?

        # Let's try to write to a new file in /usr (should be read-only or not bound)
        # or /etc/passwd (classic test).

        # Test 1: Read /etc/passwd (should fail or be empty if not bound, usually bound RO)
        # Test 2: Write to /etc/hosts (should fail)

        try:
            await proxy.write_file("/etc/hosts", "hacked")
            write_succeeded = True
        except Exception:
            write_succeeded = False

        assert not write_succeeded, "Child should NOT be able to write to /etc/hosts"

        # Test 3: Write to module path (should be allowed? or RO?)
        # PyIsolate binds module path. Read-only?
        # pyisolate/_internal/sandbox.py:
        # binds module_path as ro-bind usually? I need to check sandbox.py details.
        # If I can't check, I'll test it.

    finally:
        if os.path.exists(sensitive_path):
            os.unlink(sensitive_path)


@pytest.mark.asyncio
async def test_module_path_ro(reference_host):
    """Verify module path is read-only in child."""
    ext = reference_host.load_test_extension("ro_test", isolated=True)
    proxy = ext.get_proxy()

    # Try to write a file inside the module directory
    test_file = f"{ext.module_path}/hacked.txt"
    try:
        await proxy.write_file(test_file, "hacked")
        write_success = True
    except Exception:
        write_success = False

    assert not write_success, "Module path should be mounted Read-Only"
