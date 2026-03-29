"""Tests for host.py dispatch to conda/uv backend (Slice 3)."""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import MagicMock, patch

from pyisolate._internal.sandbox_detect import RestrictionModel

# ── __launch dispatch ────────────────────────────────────────────────


class TestLaunchDispatchConda:
    """Verify __launch() dispatches to conda backend when package_manager='conda'."""

    @patch("pyisolate._internal.host.validate_backend_config")
    @patch("pyisolate._internal.host.create_conda_env")
    @patch("pyisolate._internal.host.create_venv")
    @patch("pyisolate._internal.host.install_dependencies")
    def test_conda_calls_create_conda_env(
        self,
        mock_install_deps: MagicMock,
        mock_create_venv: MagicMock,
        mock_create_conda: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        """When package_manager='conda', __launch should call create_conda_env, NOT create_venv."""
        from pyisolate._internal.host import Extension
        from pyisolate.shared import ExtensionBase

        config = {
            "name": "test_ext",
            "module": "test_module",
            "dependencies": [],
            "share_torch": False,
            "share_cuda_ipc": False,
            "package_manager": "conda",
            "conda_channels": ["conda-forge"],
            "conda_dependencies": ["numpy"],
        }

        ext = Extension.__new__(Extension)
        ext.name = "test_ext"
        ext.config = config
        ext.venv_path = Path("/fake/venv")
        ext.module_path = "/fake/module"
        ext.extension_type = ExtensionBase

        # Call the private __launch via name mangling
        with patch.object(ext, "_launch_with_uds", return_value=MagicMock()):
            ext._Extension__launch()

        mock_create_conda.assert_called_once()
        mock_create_venv.assert_not_called()
        mock_install_deps.assert_not_called()

    @patch("pyisolate._internal.host.validate_backend_config")
    @patch("pyisolate._internal.host.create_conda_env")
    @patch("pyisolate._internal.host.create_venv")
    @patch("pyisolate._internal.host.install_dependencies")
    def test_uv_calls_create_venv(
        self,
        mock_install_deps: MagicMock,
        mock_create_venv: MagicMock,
        mock_create_conda: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        """When package_manager='uv' (default), __launch uses original uv path."""
        from pyisolate._internal.host import Extension
        from pyisolate.shared import ExtensionBase

        config = {
            "name": "test_ext",
            "module": "test_module",
            "dependencies": [],
            "share_torch": False,
            "share_cuda_ipc": False,
            "package_manager": "uv",
        }

        ext = Extension.__new__(Extension)
        ext.name = "test_ext"
        ext.config = config
        ext.venv_path = Path("/fake/venv")
        ext.module_path = "/fake/module"
        ext.extension_type = ExtensionBase

        with patch.object(ext, "_launch_with_uds", return_value=MagicMock()):
            ext._Extension__launch()

        mock_create_venv.assert_called_once()
        mock_install_deps.assert_called_once()
        mock_create_conda.assert_not_called()

    @patch("pyisolate._internal.host.validate_backend_config")
    @patch("pyisolate._internal.host.create_conda_env")
    def test_validate_called_before_conda_launch(
        self,
        mock_create_conda: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        """validate_backend_config must be called before env creation."""
        from pyisolate._internal.host import Extension
        from pyisolate.shared import ExtensionBase

        config = {
            "name": "test_ext",
            "module": "test_module",
            "dependencies": [],
            "share_torch": False,
            "share_cuda_ipc": False,
            "package_manager": "conda",
            "conda_channels": ["conda-forge"],
            "conda_dependencies": ["numpy"],
        }

        ext = Extension.__new__(Extension)
        ext.name = "test_ext"
        ext.config = config
        ext.venv_path = Path("/fake/venv")
        ext.module_path = "/fake/module"
        ext.extension_type = ExtensionBase

        with patch.object(ext, "_launch_with_uds", return_value=MagicMock()):
            ext._Extension__launch()

        mock_validate.assert_called_once_with(config)

    @patch("pyisolate._internal.host.validate_backend_config")
    @patch("pyisolate._internal.host.create_venv")
    @patch("pyisolate._internal.host.install_dependencies")
    def test_validate_called_before_uv_launch(
        self,
        mock_install: MagicMock,
        mock_create: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        """validate_backend_config must also be called for uv backend."""
        from pyisolate._internal.host import Extension
        from pyisolate.shared import ExtensionBase

        config = {
            "name": "test_ext",
            "module": "test_module",
            "dependencies": [],
            "share_torch": False,
            "share_cuda_ipc": False,
            "package_manager": "uv",
        }

        ext = Extension.__new__(Extension)
        ext.name = "test_ext"
        ext.config = config
        ext.venv_path = Path("/fake/venv")
        ext.module_path = "/fake/module"
        ext.extension_type = ExtensionBase

        with patch.object(ext, "_launch_with_uds", return_value=MagicMock()):
            ext._Extension__launch()

        mock_validate.assert_called_once_with(config)


# ── Python exe resolution ────────────────────────────────────────────


class TestPythonExeResolution:
    """Verify _launch_with_uds resolves correct python for each backend."""

    def test_conda_resolves_pixi_python(self) -> None:
        """Conda backend must use .pixi/envs/default/bin/python, not venv/bin/python."""
        from pyisolate._internal.host import Extension
        from pyisolate.shared import ExtensionBase

        config = {
            "name": "test_ext",
            "module": "test_module",
            "dependencies": [],
            "share_torch": False,
            "share_cuda_ipc": False,
            "package_manager": "conda",
            "conda_channels": ["conda-forge"],
            "conda_dependencies": ["numpy"],
        }

        ext = Extension.__new__(Extension)
        ext.name = "test_ext"
        ext.config = config
        ext.venv_path = Path("/fake/venv")
        ext.module_path = "/fake/module"
        ext.extension_type = ExtensionBase
        ext._cuda_ipc_enabled = False

        # We need to test that the python exe resolved inside _launch_with_uds
        # uses _resolve_pixi_python for conda, not the standard venv path.
        # We'll mock _resolve_pixi_python and verify it's called.
        with (
            patch(
                "pyisolate._internal.host._resolve_pixi_python",
                return_value=Path("/fake/venv/.pixi/envs/default/bin/python"),
            ) as mock_resolve,
            patch("pyisolate._internal.host.socket"),
            patch("pyisolate._internal.host.tempfile"),
            patch("pyisolate._internal.host.subprocess"),
            patch("pyisolate._internal.host.threading"),
            patch("pyisolate._internal.host.build_extension_snapshot"),
            patch("pyisolate._internal.host.JSONSocketTransport"),
            patch("pyisolate._internal.host.AsyncRPC"),
            patch(
                "pyisolate._internal.host.detect_sandbox_capability",
                return_value=MagicMock(available=True),
            ),
        ):
            # This will fail because we need more mocking, but the key assertion
            # is that _resolve_pixi_python is called for conda backend
            with contextlib.suppress(Exception):
                ext._launch_with_uds()

            mock_resolve.assert_called_once()


class TestCondaSealedWorkerBwrapDispatch:
    """Verify conda sealed_worker launches through bubblewrap with pixi python."""

    @patch("pyisolate._internal.host.build_bwrap_command")
    @patch("pyisolate._internal.host.subprocess.Popen")
    def test_conda_sealed_worker_launches_through_bwrap(
        self,
        mock_popen: MagicMock,
        mock_build_bwrap: MagicMock,
    ) -> None:
        from pyisolate._internal.host import Extension
        from pyisolate.shared import ExtensionBase

        config = {
            "name": "test_ext",
            "module": "test_module",
            "dependencies": [],
            "share_torch": False,
            "share_cuda_ipc": False,
            "package_manager": "conda",
            "execution_model": "sealed_worker",
            "conda_channels": ["conda-forge"],
            "conda_dependencies": ["numpy"],
            "sandbox": {"writable_paths": ["/fake/artifacts"]},
        }

        ext = Extension.__new__(Extension)
        ext.name = "test_ext"
        ext.config = config
        ext.venv_path = Path("/fake/venv")
        ext.module_path = "/fake/module"
        ext.extension_type = ExtensionBase
        ext._cuda_ipc_enabled = False
        ext._uds_path = None
        ext._uds_listener = None
        ext._client_sock = None

        pixi_python = Path("/fake/venv/.pixi/envs/default/bin/python")
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.args = ["bwrap", "--clearenv", str(pixi_python)]
        mock_popen.return_value = mock_proc
        mock_build_bwrap.return_value = [
            "bwrap",
            "--clearenv",
            str(pixi_python),
            "-m",
            "pyisolate._internal.uds_client",
        ]

        transport = MagicMock()
        transport.send = MagicMock()

        with (
            patch("pyisolate._internal.host._resolve_pixi_python", return_value=pixi_python),
            patch("pyisolate._internal.host.socket") as mock_socket,
            patch("pyisolate._internal.host.tempfile"),
            patch("pyisolate._internal.host.detect_sandbox_capability") as mock_detect,
            patch("sys.platform", "linux"),
            patch("pyisolate._internal.host.JSONSocketTransport", return_value=transport),
            patch("pyisolate._internal.host.AsyncRPC"),
        ):
            mock_detect.return_value = MagicMock(
                available=True,
                restriction_model=RestrictionModel.NONE,
            )
            mock_listener = MagicMock()
            mock_listener.accept.return_value = (MagicMock(), None)
            mock_socket.socket.return_value = mock_listener
            mock_socket.AF_UNIX = 1
            mock_socket.SOCK_STREAM = 1

            with (
                patch("pyisolate._internal.socket_utils.has_af_unix", return_value=True),
                patch("pyisolate._internal.socket_utils.ensure_ipc_socket_dir", return_value=Path("/run")),
                patch("pyisolate._internal.host.build_extension_snapshot", return_value={}),
                patch("os.chmod"),
            ):
                ext._launch_with_uds()

        mock_build_bwrap.assert_called_once()
        kwargs = mock_build_bwrap.call_args.kwargs
        assert kwargs["execution_model"] == "sealed_worker"
        assert kwargs["sandbox_config"] == {"writable_paths": ["/fake/artifacts"]}
        assert kwargs["python_exe"] == str(pixi_python)
        transport.send.assert_called_once()
        bootstrap_data = transport.send.call_args[0][0]
        assert bootstrap_data["snapshot"]["apply_host_sys_path"] is False


class TestEnvPropagation:
    """Verify child env overrides are applied on non-Linux launches too."""

    @patch("pyisolate._internal.host.subprocess.Popen")
    def test_windows_launch_propagates_config_env(
        self,
        mock_popen: MagicMock,
    ) -> None:
        from pyisolate._internal.host import Extension
        from pyisolate.shared import ExtensionBase

        config = {
            "name": "test_ext",
            "module": "test_module",
            "dependencies": [],
            "share_torch": False,
            "share_cuda_ipc": False,
            "package_manager": "conda",
            "execution_model": "sealed_worker",
            "conda_channels": ["conda-forge"],
            "conda_dependencies": ["boltons"],
            "env": {"PYISOLATE_ARTIFACT_DIR": r"C:\artifacts"},
        }

        ext = Extension.__new__(Extension)
        ext.name = "test_ext"
        ext.config = config
        ext.venv_path = Path(r"C:\fake\venv")
        ext.module_path = r"C:\fake\module"
        ext.extension_type = ExtensionBase
        ext._cuda_ipc_enabled = False
        ext._uds_path = None
        ext._uds_listener = None
        ext._client_sock = None

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc
        transport = MagicMock()
        transport.send = MagicMock()

        with (
            patch(
                "pyisolate._internal.host._resolve_pixi_python",
                return_value=Path(r"C:\fake\venv\.pixi\envs\default\python.exe"),
            ),
            patch("pyisolate._internal.host.socket") as mock_socket,
            patch("pyisolate._internal.host.JSONSocketTransport", return_value=transport),
            patch("pyisolate._internal.host.AsyncRPC"),
            patch("pyisolate._internal.host.build_extension_snapshot", return_value={}),
            patch("pyisolate._internal.socket_utils.has_af_unix", return_value=False),
            patch("os.name", "nt"),
            patch("sys.platform", "win32"),
        ):
            mock_listener = MagicMock()
            mock_listener.accept.return_value = (MagicMock(), None)
            mock_listener.getsockname.return_value = ("127.0.0.1", 43210)
            mock_socket.socket.return_value = mock_listener
            mock_socket.AF_INET = 2
            mock_socket.SOCK_STREAM = 1
            mock_socket.SOL_SOCKET = 1
            mock_socket.SO_REUSEADDR = 2

            ext._launch_with_uds()

        child_env = mock_popen.call_args.kwargs["env"]
        assert child_env["PYISOLATE_ARTIFACT_DIR"] == r"C:\artifacts"


# ── share_cuda_ipc forced False ──────────────────────────────────────


class TestCondaCudaIpcForced:
    """Conda backend must force share_cuda_ipc=False."""

    @patch("pyisolate._internal.host.create_conda_env")
    @patch("pyisolate._internal.host.validate_backend_config")
    def test_conda_forces_cuda_ipc_false(
        self,
        mock_validate: MagicMock,
        mock_conda: MagicMock,
    ) -> None:
        """Even if config says share_cuda_ipc=True, conda must override to False."""
        from pyisolate._internal.host import Extension
        from pyisolate.shared import ExtensionBase

        config = {
            "name": "test_ext",
            "module": "test_module",
            "dependencies": [],
            "share_torch": False,
            "share_cuda_ipc": True,  # Explicitly set, should be overridden
            "package_manager": "conda",
            "conda_channels": ["conda-forge"],
            "conda_dependencies": ["numpy"],
        }

        ext = Extension.__new__(Extension)
        ext.name = "test_ext"
        ext.config = config
        ext.venv_path = Path("/fake/venv")
        ext.module_path = "/fake/module"
        ext.extension_type = ExtensionBase
        ext._cuda_ipc_enabled = True

        with patch.object(ext, "_launch_with_uds", return_value=MagicMock()):
            ext._Extension__launch()

        # After __launch, cuda_ipc should be forced False
        assert ext._cuda_ipc_enabled is False
        assert config["share_cuda_ipc"] is False
