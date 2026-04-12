from collections.abc import AsyncGenerator

import pytest

from tests.harness.host import ReferenceHost


@pytest.fixture
async def reference_host() -> AsyncGenerator[ReferenceHost, None]:
    """Provides a ReferenceHost instance."""
    host = ReferenceHost()
    try:
        host.setup()
        yield host
    finally:
        await host.cleanup()
