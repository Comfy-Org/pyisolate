"""
Pytest configuration and fixtures.

Add any shared fixtures or pytest configuration here.
"""

import logging
import sys
from types import SimpleNamespace

import pytest

from pyisolate._internal.singleton_context import singleton_scope


@pytest.fixture(autouse=True)
def clean_singletons():
    """Auto-cleanup fixture for singleton isolation between tests.

    This fixture runs automatically for all tests and ensures that:
    - Each test starts with a clean singleton state
    - Singletons created during a test are cleaned up afterward
    - Previous singleton state is restored after each test

    This eliminates the need for manual SingletonMetaclass._instances.clear()
    calls in individual tests.
    """
    with singleton_scope():
        yield


@pytest.fixture
def patch_extension_launch(monkeypatch):
    """Prevent real subprocess launches during unit tests.

    NOTE: This fixture is NOT autouse - integration tests should NOT use it.
    Unit tests that need mocked launch should explicitly request this fixture.
    """
    from pyisolate._internal import host as host_internal

    original_launch = host_internal.Extension._Extension__launch
    host_internal.Extension._orig_launch = original_launch  # type: ignore[attr-defined]

    def dummy_launch(self):
        return SimpleNamespace(
            is_alive=lambda: False,
            terminate=lambda: None,
            join=lambda timeout=None: None,
            kill=lambda: None,
        )

    monkeypatch.setattr(host_internal.Extension, "_Extension__launch", dummy_launch)
    yield
    monkeypatch.setattr(host_internal.Extension, "_Extension__launch", original_launch)


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Register custom markers
    config.addinivalue_line("markers", "slow: marks tests as slow (>5s, deselect with -m 'not slow')")

    # Set up logging
    log_level = logging.DEBUG if config.getoption("--debug-pyisolate") else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Set specific logger levels
    logging.getLogger("pyisolate").setLevel(log_level)
    logging.getLogger("asyncio").setLevel(log_level)

    # If custom log file is specified, add file handler
    custom_log_file = config.getoption("--pyisolate-log-file")
    if custom_log_file:
        file_handler = logging.FileHandler(custom_log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
            )
        )
        logging.getLogger().addHandler(file_handler)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--debug-pyisolate",
        action="store_true",
        default=False,
        help="Enable debug logging for pyisolate (shows detailed execution flow)",
    )
    parser.addoption(
        "--pyisolate-log-file",
        action="store",
        default=None,
        help="Log pyisolate debug output to specified file",
    )
