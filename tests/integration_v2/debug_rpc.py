from typing import Any
import logging
import sys

import pytest

# Configure logging to see what's happening
logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
logging.getLogger("pyisolate").setLevel(logging.DEBUG)


@pytest.mark.asyncio
async def test_debug_ping(reference_host: Any) -> None:
    print("\n--- Starting Debug Ping ---")
    ext = reference_host.load_test_extension("debug_ping", isolated=True)

    print(f"Extension loaded: {ext}")
    proxy = ext.get_proxy()
    print(f"Proxy obtained: {proxy}")

    try:
        response = await proxy.ping()
        print(f"Ping response: {response}")
        assert response == "pong"
    except Exception as e:
        print(f"Ping failed with: {e}")
        raise
