"""Preserve provider execution contracts and isolate scopes between async steps."""

import inspect
import sys
from collections.abc import Callable
from functools import wraps
from typing import Any, NamedTuple

import torch


class _LoaderState(NamedTuple):
    device: Any
    text_encoder_device: Any
    unet_offload_device: Any
    cuda_device: int | None
    core_patcher: Any


def _snapshot() -> _LoaderState:
    from . import (
        get_current_device,
        get_current_text_encoder_device,
        get_current_unet_offload_device,
    )

    patcher = sys.modules.get("comfy.model_patcher")
    return _LoaderState(
        get_current_device(),
        get_current_text_encoder_device(),
        get_current_unet_offload_device(),
        torch.cuda.current_device() if torch.cuda.is_initialized() else None,
        getattr(patcher, "CoreModelPatcher", None),
    )


def _restore(state: _LoaderState):
    from . import (
        set_current_device,
        set_current_text_encoder_device,
        set_current_unet_offload_device,
    )

    set_current_device(state.device)
    set_current_text_encoder_device(state.text_encoder_device)
    set_current_unet_offload_device(state.unet_offload_device)
    if state.cuda_device is not None:
        torch.cuda.set_device(state.cuda_device)
    patcher: Any = sys.modules.get("comfy.model_patcher")
    if patcher is not None and state.core_patcher is not None:
        patcher.CoreModelPatcher = state.core_patcher


class _ScopedAwaitable:
    """Save task-local loader state while returning ambient state to the executor.

    CUDA's current device and CoreModelPatcher are thread/module state, so a
    ContextVar alone cannot isolate them when a provider yields to another node.
    """

    def __init__(self, coroutine):
        self.coroutine = coroutine

    def __await__(self):
        iterator = self.coroutine.__await__()
        state = _snapshot()
        value = None
        error = None
        try:
            while True:
                ambient = _snapshot()
                try:
                    _restore(state)
                    try:
                        pending = (
                            iterator.throw(error) if error is not None else iterator.send(value)
                        )
                    except StopIteration as done:
                        return done.value
                    finally:
                        state = _snapshot()
                finally:
                    _restore(ambient)
                try:
                    value = yield pending
                    error = None
                except BaseException as caught:
                    value = None
                    error = caught
        finally:
            ambient = _snapshot()
            try:
                _restore(state)
                iterator.close()
            finally:
                _restore(ambient)


def adapt_loader_execution(provider, scoped_loader) -> Callable[..., Any]:
    """Keep synchronous providers synchronous; await async providers before output handling."""

    if inspect.iscoroutinefunction(getattr(provider, getattr(provider, "FUNCTION", ""), None)):

        async def run(self, *args, **kwargs):
            scope = scoped_loader(self, *args, **kwargs)
            try:
                fn, call_args, call_kwargs = next(scope)
                try:
                    result = await fn(*call_args, **call_kwargs)
                except BaseException as error:
                    scope.throw(error)
                    raise
                try:
                    scope.send(result)
                except StopIteration as done:
                    return done.value
                raise RuntimeError("Loader scope yielded more than one provider call")
            finally:
                scope.close()

        @wraps(scoped_loader)
        async def asynchronous(self, *args, **kwargs):
            return await _ScopedAwaitable(run(self, *args, **kwargs))

        return asynchronous

    @wraps(scoped_loader)
    def synchronous(self, *args, **kwargs):
        scope = scoped_loader(self, *args, **kwargs)
        try:
            fn, call_args, call_kwargs = next(scope)
            try:
                result = fn(*call_args, **call_kwargs)
            except BaseException as error:
                scope.throw(error)
                raise
            try:
                scope.send(result)
            except StopIteration as done:
                return done.value
            raise RuntimeError("Loader scope yielded more than one provider call")
        finally:
            scope.close()

    return synchronous
