import argparse
import asyncio
import inspect
import logging
import os
import sys
from typing import TypedDict, cast

import yaml
from shared import DatabaseSingleton, ExampleExtensionBase

import pyisolate
from pyisolate._internal.sandbox_detect import detect_sandbox_capability


# ANSI color codes for terminal output (using 256-color mode for better compatibility)
class Colors:
    GREEN = "\033[38;5;34m"  # Bright green that stands out
    RED = "\033[38;5;196m"  # Bright red
    YELLOW = "\033[38;5;220m"  # Bright yellow
    BLUE = "\033[38;5;33m"  # Bright blue
    BOLD = "\033[1m"
    RESET = "\033[0m"


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        if verbose
        else "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Override any existing configuration
    )


logger = logging.getLogger(__name__)


async def async_main():
    # Since this is just an example, we'll install `pyisolate` in edit mode
    pyisolate_dir = os.path.dirname(os.path.dirname(os.path.realpath(pyisolate.__file__)))

    base_path = os.path.dirname(os.path.realpath(__file__))

    # Create a 'manager' to load extensions of a particular type
    config = pyisolate.ExtensionManagerConfig(venv_root_path=os.path.join(base_path, "extension-venvs"))
    manager = pyisolate.ExtensionManager(ExampleExtensionBase, config)

    sandbox_mode = pyisolate.SandboxMode.REQUIRED
    if sys.platform == "linux":
        cap = detect_sandbox_capability()
        if not cap.available:
            sandbox_mode = pyisolate.SandboxMode.DISABLED
            logger.warning(
                "Sandbox unavailable in example environment (%s); using sandbox_mode=disabled",
                cap.restriction_model,
            )

    extensions: list[ExampleExtensionBase] = []
    extension_dir = os.path.join(base_path, "extensions")
    for extension in os.listdir(extension_dir):
        if os.path.isdir(os.path.join(extension_dir, extension)):
            module_path = os.path.join(extension_dir, extension)

            yaml_path = os.path.join(module_path, "manifest.yaml")

            class CustomConfig(TypedDict):
                enabled: bool
                isolated: bool
                dependencies: list[str]
                share_torch: bool

            # Load the extension configuration from a YAML file
            with open(yaml_path) as f:
                manifest = cast(CustomConfig, yaml.safe_load(f))

            # Skip disabled extensions
            if not manifest.get("enabled", True):
                logger.info(f"Skipping disabled extension: {extension}")
                continue

            # On Windows, edit mode doesn't really work.
            # pyisolate_install = [ pyisolate_dir ] if os.name == "nt" else [ "-e", pyisolate_dir, ]
            pyisolate_install = [
                "-e",
                pyisolate_dir,
            ]

            # Each extension can have its own configuration and dependencies
            config = pyisolate.ExtensionConfig(
                name=extension,
                module_path=module_path,
                isolated=manifest["isolated"],
                dependencies=manifest["dependencies"] + pyisolate_install,
                apis=[DatabaseSingleton],
                share_torch=manifest["share_torch"],
                sandbox_mode=sandbox_mode,
            )

            extension = manager.load_extension(config)
            extensions.append(extension)
            logger.debug(f"Loaded extension: {extension}")

    # Execute extensions
    logger.debug("Executing extensions...")
    for index, extension in enumerate(extensions):
        await extension.do_stuff(f"Hello from extension {index}!")

    # Test verification
    logger.info("Running extension tests...")
    db = DatabaseSingleton()
    test_results = []

    # Test Extension 1
    ext1_result = await db.get_value("extension1_result")
    if (
        ext1_result
        and ext1_result.get("extension") == "extension1"
        and ext1_result.get("data_rows") == 3
        and ext1_result.get("numpy_version").startswith("1.")
    ):
        test_results.append(("Extension1", "PASSED", "Data processing with pandas/numpy 1.x"))
        logger.debug(f"Extension1 result: {ext1_result}")
    else:
        test_results.append(
            ("Extension1", "FAILED", f"Expected data processing result not found: {ext1_result}")
        )

    # Test Extension 2
    ext2_result = await db.get_value("extension2_result")
    if ext2_result and ext2_result.get("extension") == "extension2" and ext2_result.get("array_sum") == 17.5:
        test_results.append(("Extension2", "PASSED", "Array processing with numpy 2.x"))
        logger.debug(f"Extension2 result: {ext2_result}")
    else:
        test_results.append(
            ("Extension2", "FAILED", f"Expected array processing result not found: {ext2_result}")
        )

    # Test Extension 3
    ext3_result = await db.get_value("extension3_result")
    if ext3_result and ext3_result.get("extension") == "extension3" and ext3_result.get("count") == 5:
        test_results.append(("Extension3", "PASSED", "HTML parsing with BeautifulSoup/scipy"))
        logger.debug(f"Extension3 result: {ext3_result}")
    else:
        test_results.append(("Extension3", "FAILED", "Expected HTML parsing result not found"))

    # Display test results
    logger.info(Colors.BOLD + "=" * 60 + Colors.RESET)
    logger.info(Colors.BOLD + "EXTENSION TEST RESULTS" + Colors.RESET)
    logger.info(Colors.BOLD + "=" * 60 + Colors.RESET)

    failed_tests = 0
    for name, status, description in test_results:
        if status == "PASSED":
            status_colored = f"{Colors.GREEN}✓ PASSED{Colors.RESET}"
            failed_tests += 0
        else:
            status_colored = f"{Colors.RED}✗ FAILED{Colors.RESET}"
            failed_tests += 1

        logger.info(f"{name:12} | {status_colored:15} | {description}")

    logger.info(Colors.BOLD + "=" * 60 + Colors.RESET)

    if failed_tests == 0:
        summary_color = Colors.GREEN
        summary_text = f"Tests passed: {len(test_results) - failed_tests}/{len(test_results)}"
    else:
        summary_color = Colors.RED
        summary_text = f"Tests passed: {len(test_results) - failed_tests}/{len(test_results)}"

    logger.info(summary_color + summary_text + Colors.RESET)

    # Shutdown extensions
    logger.debug("Shutting down extensions...")
    for extension in extensions:
        stop_result = extension.stop()
        if inspect.isawaitable(stop_result):
            await stop_result

    # Exit with appropriate code
    if failed_tests > 0:
        logger.error(f"{Colors.RED}Example failed with {failed_tests} test failure(s){Colors.RESET}")
        sys.exit(1)
    else:
        logger.info(f"{Colors.GREEN}All tests passed successfully!{Colors.RESET}")
        sys.exit(0)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Pyisolate Example - demonstrates isolated extension environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py         # Run with normal output
  python main.py -v      # Run with verbose debug output
        """,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger.debug(f"Using pyisolate from: {pyisolate.__file__}")
    if args.verbose:
        logger.info("Debug logging enabled")

    asyncio.run(async_main())
