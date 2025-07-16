from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import queue
import threading
import traceback
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypedDict,
    TypeVar,
    Union,
    cast,
    get_type_hints,
)

# We only import this to get type hinting working. It can also be a torch.multiprocessing
if TYPE_CHECKING:
    import multiprocessing as typehint_mp

else:
    import multiprocessing
    typehint_mp = multiprocessing

# GPU-specific utilities have moved to pyisolate._internal.gpu_utils
if TYPE_CHECKING:
    from .gpu_utils import (
        maybe_deserialize_tensor,
        maybe_serialize_tensor,
    )
else:
    from .gpu_utils import (
        maybe_deserialize_tensor,
        maybe_serialize_tensor,
    )

logger = logging.getLogger(__name__)

# TODO - Remove me
debug_all_messages = False


def debugprint(*args, **kwargs):
    if debug_all_messages:
        logger.debug(" ".join(str(arg) for arg in args))


def local_execution(func):
    """Decorator to mark a ProxiedSingleton method for local execution.

    By default, all methods in a ProxiedSingleton are executed on the host
    process via RPC. Use this decorator to mark methods that should run
    locally in each process instead.

    This is useful for methods that:
    - Need to access process-local state (e.g., caches, metrics)
    - Don't need to be synchronized across processes
    - Would have poor performance if executed via RPC

    Args:
        func: The method to mark for local execution.

    Returns:
        The decorated method that will execute locally.

    Example:
        >>> class CachedService(ProxiedSingleton):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self._local_cache = {}
        ...         self.shared_data = {}
        ...
        ...     async def get_shared(self, key: str) -> Any:
        ...         # This runs on the host via RPC
        ...         return self.shared_data.get(key)
        ...
        ...     @local_execution
        ...     def get_cache_size(self) -> int:
        ...         # This runs locally in each process
        ...         return len(self._local_cache)

    Note:
        Local methods can be synchronous or asynchronous, but they cannot
        access shared state from the host process.
    """
    func._is_local_execution = True
    return func


class LocalMethodRegistry:
    """Registry for local method implementations in proxied singletons"""

    _instance: LocalMethodRegistry | None = None
    _lock = threading.Lock()

    def __init__(self):
        self._local_implementations: dict[type, object] = {}
        self._local_methods: dict[type, set[str]] = {}

    @classmethod
    def get_instance(cls) -> LocalMethodRegistry:
        """Get the singleton instance of LocalMethodRegistry"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register_class(self, cls: type) -> None:
        """Register a class with its local method implementations"""
        # Create a local instance by bypassing the singleton mechanism
        # We call the base object.__new__ directly to avoid getting the existing singleton
        local_instance = object.__new__(cls)  # type: ignore[misc]
        cls.__init__(local_instance)
        self._local_implementations[cls] = local_instance

        # Track which methods are marked for local execution
        local_methods = set()
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if getattr(method, "_is_local_execution", False):
                local_methods.add(name)

        # Also check instance methods
        for name in dir(cls):
            if not name.startswith("_"):
                attr = getattr(cls, name, None)
                if callable(attr) and getattr(attr, "_is_local_execution", False):
                    local_methods.add(name)

        self._local_methods[cls] = local_methods

    def is_local_method(self, cls: type, method_name: str) -> bool:
        """Check if a method should be executed locally"""
        return cls in self._local_methods and method_name in self._local_methods[cls]

    def get_local_method(self, cls: type, method_name: str):
        """Get the local implementation of a method"""
        if cls not in self._local_implementations:
            raise ValueError(f"Class {cls} not registered for local execution")

        local_instance = self._local_implementations[cls]
        return getattr(local_instance, method_name)


class RPCRequest(TypedDict):
    kind: Literal["call"]
    object_id: str
    call_id: int
    parent_call_id: int | None
    method: str
    args: tuple
    kwargs: dict


class RPCResponse(TypedDict):
    kind: Literal["response"]
    call_id: int
    result: Any
    error: str | None


RPCMessage = Union[RPCRequest, RPCResponse]


class RPCPendingRequest(TypedDict):
    kind: Literal["call"]
    object_id: str
    parent_call_id: int | None
    calling_loop: asyncio.AbstractEventLoop
    future: asyncio.Future
    method: str
    args: tuple
    kwargs: dict


RPCPendingMessage = Union[RPCPendingRequest, RPCResponse]


proxied_type = TypeVar("proxied_type", bound=object)


class AsyncRPC:
    def __init__(
        self,
        recv_queue: typehint_mp.Queue[RPCMessage],
        send_queue: typehint_mp.Queue[RPCMessage],
    ):
        self.id = str(uuid.uuid4())
        self.handling_call_id: contextvars.ContextVar[int | None]
        self.handling_call_id = contextvars.ContextVar(self.id + "_handling_call_id", default=None)
        self.recv_queue = recv_queue
        self.send_queue = send_queue
        self.lock = threading.Lock()
        self.pending: dict[int, RPCPendingRequest] = {}
        self.default_loop = asyncio.get_event_loop()
        self.callees: dict[str, object] = {}
        self.blocking_future: asyncio.Future | None = None

        # Use an outbox to avoid blocking when we try to send
        self.outbox: queue.Queue[RPCPendingRequest] = queue.Queue()

    def create_caller(self, abc: type[proxied_type], object_id: str) -> proxied_type:
        this = self

        class CallWrapper:
            def __init__(self):
                pass

            def __getattr__(self, name):
                attr = getattr(abc, name, None)
                if not callable(attr) or name.startswith("_"):
                    raise AttributeError(f"{name} is not a valid method")

                # Check if this method should run locally
                registry = LocalMethodRegistry.get_instance()
                if registry.is_local_method(abc, name):
                    return registry.get_local_method(abc, name)

                # Original RPC logic for remote methods
                if not inspect.iscoroutinefunction(attr):
                    raise ValueError(f"{name} is not a coroutine function")

                async def method(*args, **kwargs):
                    loop = asyncio.get_event_loop()
                    pending_request = RPCPendingRequest(
                        kind="call",
                        object_id=object_id,
                        parent_call_id=this.handling_call_id.get(),
                        calling_loop=loop,
                        future=loop.create_future(),
                        method=name,
                        args=args,
                        kwargs=kwargs,
                    )
                    this.outbox.put(pending_request)
                    result = await pending_request["future"]
                    return result

                return method

        return cast(proxied_type, CallWrapper())

    def register_callee(self, object_instance: object, object_id: str):
        with self.lock:
            if object_id in self.callees:
                raise ValueError(f"Object ID {object_id} already registered")
            self.callees[object_id] = object_instance

    async def run_until_stopped(self):
        # Start the threads
        if self.blocking_future is None:
            self.run()
        assert self.blocking_future is not None, "RPC must be running to wait"
        await self.blocking_future

    async def stop(self):
        # Stop the threads by sending None to the queues
        assert self.blocking_future is not None, "RPC must be running to stop"
        self.blocking_future.set_result(None)

    def run(self):
        self.blocking_future = self.default_loop.create_future()
        self._threads = [
            threading.Thread(target=self._recv_thread, daemon=True),
            threading.Thread(target=self._send_thread, daemon=True),
        ]
        for t in self._threads:
            t.start()

    async def dispatch_request(self, request: RPCRequest):
        try:
            object_id = request["object_id"]
            method = request["method"]
            args = request["args"]
            kwargs = request["kwargs"]

            callee = None
            with self.lock:
                callee = self.callees.get(object_id, None)

            if callee is None:
                raise ValueError(f"Object ID {object_id} not registered for remote calls")

            # Call the method on the callee
            debugprint("Dispatching request: ", request)
            func = getattr(callee, method)
            result = (
                (await func(*args, **kwargs)) if inspect.iscoroutinefunction(func) else func(*args, **kwargs)
            )
            response = RPCResponse(
                kind="response",
                call_id=request["call_id"],
                result=result,
                error=None,
            )
        except Exception as e:
            error_msg = str(e)
            # Check for CUDA OOM errors specifically
            if "CUDA error: out of memory" in error_msg or "out of memory" in error_msg.lower():
                print(f"CUDA OOM error in RPC dispatch for {request.get('method', 'unknown')}: {error_msg}")
                traceback.print_exc()
            else:
                traceback.print_exc()
            response = RPCResponse(
                kind="response",
                call_id=request["call_id"],
                result=None,
                error=error_msg,
            )

        debugprint("Sending response: ", response)
        try:
            self.send_queue.put(response)
        except Exception as e:
            error_msg = str(e)
            if "CUDA error: out of memory" in error_msg or "out of memory" in error_msg.lower():
                print(f"CUDA OOM error while sending RPC response: {error_msg}")
                # Try to send a simpler error response
                try:
                    simple_response = RPCResponse(
                        kind="response",
                        call_id=request["call_id"],
                        result=None,
                        error=f"CUDA out of memory during response transmission: {error_msg}",
                    )
                    self.send_queue.put(simple_response)
                except Exception:
                    print("Failed to send even a simple error response - process may be stuck")
            else:
                print(f"Error sending RPC response: {error_msg}")
                raise

    def _recv_thread(self):
        while True:
            try:
                item = self.recv_queue.get()
            except Exception as e:
                error_msg = str(e)
                if "CUDA error: out of memory" in error_msg or "out of memory" in error_msg.lower():
                    print(f"CUDA OOM error while receiving RPC message: {error_msg}")
                    # Try to continue receiving other messages
                    continue
                else:
                    print(f"Error receiving RPC message: {error_msg}")
                    continue

            debugprint("Got recv: ", item)
            if item is None:
                if self.blocking_future:
                    self.default_loop.call_soon_threadsafe(self.blocking_future.set_result, None)
                break

            # Device-aware deserialization for args/kwargs/result
            if item["kind"] == "response":
                debugprint("Got response: ", item)
                call_id = item["call_id"]
                pending_request = None
                with self.lock:
                    pending_request = self.pending.pop(call_id, None)
                debugprint("Pending request: ", pending_request)
                if pending_request:
                    if "error" in item and item["error"] is not None:
                        debugprint("Error in response: ", item["error"])
                        pending_request["calling_loop"].call_soon_threadsafe(
                            pending_request["future"].set_exception,
                            Exception(item["error"]),
                        )
                    else:
                        debugprint("Got result: ", item["result"])
                        set_result = pending_request["future"].set_result
                        result = maybe_deserialize_tensor(item["result"])
                        pending_request["calling_loop"].call_soon_threadsafe(set_result, result)
                else:
                    continue
            elif item["kind"] == "call":
                request = cast(RPCRequest, item)
                debugprint("Got call: ", request)
                request_parent = request.get("parent_call_id", None)
                call_id = request["call_id"]

                # Device-aware deserialization for args/kwargs
                args = tuple(maybe_deserialize_tensor(arg) for arg in request["args"])
                kwargs = {k: maybe_deserialize_tensor(v) for k, v in request["kwargs"].items()}
                request_mod = dict(request)
                request_mod["args"] = args
                request_mod["kwargs"] = kwargs

                call_on_loop = self.default_loop
                if request_parent is not None:
                    pending_request = None
                    with self.lock:
                        pending_request = self.pending.get(request_parent, None)
                    if pending_request:
                        call_on_loop = pending_request["calling_loop"]

                async def call_with_context(captured_request):
                    token = self.handling_call_id.set(captured_request["call_id"])
                    try:
                        return await self.dispatch_request(captured_request)
                    finally:
                        self.handling_call_id.reset(token)

                asyncio.run_coroutine_threadsafe(coro=call_with_context(request_mod), loop=call_on_loop)
            else:
                raise ValueError(f"Unknown item type: {type(item)}")

    def _send_thread(self):
        id_gen = 0
        while True:
            item = self.outbox.get()
            if item is None:
                break

            debugprint("Got send: ", item)
            if item["kind"] == "call":
                call_id = id_gen
                id_gen += 1
                with self.lock:
                    self.pending[call_id] = item
                # Device-aware serialization for args/kwargs
                args = tuple(maybe_serialize_tensor(arg) for arg in item["args"])
                kwargs = {k: maybe_serialize_tensor(v) for k, v in item["kwargs"].items()}
                request = RPCRequest(
                    kind="call",
                    object_id=item["object_id"],
                    call_id=call_id,
                    parent_call_id=item["parent_call_id"],
                    method=item["method"],
                    args=args,
                    kwargs=kwargs,
                )
                try:
                    self.send_queue.put(request)
                except Exception as e:
                    error_msg = str(e)
                    if "CUDA error: out of memory" in error_msg or "out of memory" in error_msg.lower():
                        print(f"CUDA OOM error while sending RPC request for {item['method']}: {error_msg}")
                        try:
                            simple_response = RPCRequest(
                                kind="call",
                                object_id=item["object_id"],
                                call_id=call_id,
                                parent_call_id=item["parent_call_id"],
                                method=item["method"],
                                args=(),
                                kwargs={},
                            )
                            self.send_queue.put(simple_response)
                        except Exception:
                            print("Failed to send even a simple error request - process may be stuck")
                    else:
                        print(f"Error sending RPC request: {error_msg}")
                        raise
            else:
                # For responses, patch result for device-aware serialization
                response = item
                if "result" in response:
                    response_mod = dict(response)
                    response_mod["result"] = maybe_serialize_tensor(response["result"])
                    self.send_queue.put(cast(RPCResponse, response_mod))
                else:
                    self.send_queue.put(response)


class SingletonMetaclass(type):
    T = TypeVar("T", bound="SingletonMetaclass")
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    def inject_instance(cls: type[T], instance: T) -> None:
        assert cls not in SingletonMetaclass._instances, "Cannot inject instance after first instantiation"
        SingletonMetaclass._instances[cls] = instance

    def get_instance(cls: type[T], *args, **kwargs) -> T:
        """
        Gets the singleton instance of the class, creating it if it doesn't exist.
        """
        if cls not in SingletonMetaclass._instances:
            SingletonMetaclass._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    def use_remote(cls, rpc: AsyncRPC) -> None:
        assert issubclass(cls, ProxiedSingleton), (
            "Class must be a subclass of ProxiedSingleton to be made remote"
        )
        id = cls.get_remote_id()
        remote = rpc.create_caller(cls, id)

        # Register local implementations for methods marked with @local_execution
        registry = LocalMethodRegistry.get_instance()
        registry.register_class(cls)

        cls.inject_instance(remote)  # type: ignore

        for name, t in get_type_hints(cls).items():
            if isinstance(t, type) and issubclass(t, ProxiedSingleton) and not name.startswith("_"):
                # If the type is a ProxiedSingleton, we need to register it as well
                assert issubclass(t, ProxiedSingleton), f"{t} must be a subclass of ProxiedObject"
                caller = rpc.create_caller(t, t.get_remote_id())
                setattr(remote, name, caller)


class ProxiedSingleton(metaclass=SingletonMetaclass):
    """Base class for creating shared singleton services across processes.

    ProxiedSingleton enables you to create services that have a single instance
    shared across all extensions and the host process. When an extension accesses
    a ProxiedSingleton, it automatically gets a proxy to the singleton instance
    in the host process, ensuring all processes share the same state.

    This is particularly useful for shared resources like databases, configuration
    managers, or any service that should maintain consistent state across all
    extensions.

    Advanced usage: Methods can be marked to run locally in each process instead
    of being proxied to the host (see internal documentation for details).

    Example:
        >>> from pyisolate import ProxiedSingleton
        >>>
        >>> class DatabaseService(ProxiedSingleton):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.data = {}
        ...
        ...     async def get(self, key: str) -> Any:
        ...         return self.data.get(key)
        ...
        ...     async def set(self, key: str, value: Any) -> None:
        ...         self.data[key] = value
        ...
        >>>
        >>> # In extension configuration:
        >>> config = ExtensionConfig(
        ...     name="my_extension",
        ...     module_path="./extension.py",
        ...     apis=[DatabaseService],  # Grant access to this singleton
        ...     # ... other config
        ... )

    Note:
        All methods that should be accessible via RPC must be async methods.
        Synchronous methods can only be used if marked with @local_execution.
    """

    def __init__(self):
        """Initialize the ProxiedSingleton.

        This constructor is called only once per singleton class in the host
        process. Extensions will receive a proxy instead of creating new instances.
        """
        super().__init__()

    @classmethod
    def get_remote_id(cls) -> str:
        """Get the unique identifier for this singleton in the RPC system.

        By default, this returns the class name. Override this method if you
        need a different identifier (e.g., to avoid naming conflicts).

        You probably don't need to override this.

        Returns:
            The string identifier used to register and look up this singleton
            in the RPC system.
        """
        return cls.__name__

    def _register(self, rpc: AsyncRPC):
        """Register this singleton instance with the RPC system.

        This method is called automatically by the framework to make this
        singleton available for remote calls. It should not be called directly
        by user code.

        Args:
            rpc: The AsyncRPC instance to register with.
        """
        id = self.get_remote_id()
        rpc.register_callee(self, id)

        # Iterate through all attributes on the class and register any that are also ProxiedSingleton
        for name, attr in self.__class__.__dict__.items():
            if isinstance(attr, ProxiedSingleton) and not name.startswith("_"):
                attr._register(rpc)
