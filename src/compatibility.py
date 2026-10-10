"""Small compatibility decisions kept separate from ComfyUI startup patches."""

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from functools import wraps
from typing import Any, cast

from packaging.version import InvalidVersion, Version


def needs_legacy_dlpack_guard(version: str) -> bool:
    try:
        return Version(version) < Version("0.2.35")
    except InvalidVersion:
        # An unknown/new backend should keep its own export implementation.
        return False


@contextmanager
def legacy_patcher_for_distorch(model_patcher, enabled: bool):
    original = model_patcher.CoreModelPatcher
    if enabled:
        model_patcher.CoreModelPatcher = model_patcher.ModelPatcher
    try:
        yield
    finally:
        model_patcher.CoreModelPatcher = original


def preserve_core_model_loading(core_loader, distributed_loader):
    """Use DisTorch's loader only for annotated models or legacy provider contracts.

    Empty-allocation CLIP nodes must keep core's DynamicVRAM eviction, pinning,
    and loading order, even after a DisTorch node has installed its patches.
    """

    @wraps(core_loader)
    def load_models_gpu(models, *args, **kwargs):
        models = list(models)
        pending = list(models)
        seen = set()
        while pending:
            model = pending.pop()
            if id(model) in seen:
                continue
            seen.add(id(model))
            metadata = getattr(getattr(model, "model", None), "_distorch_v2_meta", None)
            if metadata and metadata.get("full_allocation"):
                return distributed_loader(models, *args, **kwargs)
            patches = getattr(model, "model_patches_models", None)
            if not callable(patches):
                # Old GGUF providers need the fallback in the distributed loader.
                return distributed_loader(models, *args, **kwargs)
            pending.extend(cast(Callable[[], Iterable[Any]], patches)())
        return core_loader(models, *args, **kwargs)

    return load_models_gpu
