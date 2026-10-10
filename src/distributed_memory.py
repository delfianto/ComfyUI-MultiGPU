"""Physical storage and eviction accounting for models spanning devices."""

from collections import defaultdict
from collections.abc import Callable
from functools import wraps
from typing import Any, cast

import torch


def resident_storage_bytes(model):
    """Count unique backing storage, including packed weights and their metadata."""
    memory = defaultdict(int)
    visited = set()
    storages = set()

    def visit(tensor):
        if id(tensor) in visited:
            return
        visited.add(id(tensor))
        flatten = getattr(tensor, "__tensor_flatten__", None)
        if callable(flatten):
            names, _context = cast(Callable[[], tuple[list[str], Any]], flatten)()
            for name in names:
                visit(getattr(tensor, name))
            return
        storage = tensor.untyped_storage()
        device = torch.device(tensor.device)
        key = (device, storage.data_ptr(), storage.nbytes())
        if key not in storages:
            storages.add(key)
            memory[device] += storage.nbytes()

    for parameter in model.parameters():
        visit(parameter)
    for buffer in model.buffers():
        visit(buffer)
    return dict(memory)


def allocation_memory_required(plan):
    """Reserve only storage that must still move to each assigned device."""
    required = defaultdict(int)
    for device, blocks in plan["device_assignments"].items():
        target = torch.device(device)
        for _name, module, _type, size in blocks:
            resident = resident_storage_bytes(module).get(target, 0)
            required[target] += max(0, size - resident)
    return dict(required)


def preserve_distributed_eviction(core_free_memory, model_management):
    """Let core evict normal models first, then unused cross-device residents."""

    @wraps(core_free_memory)
    def free_memory(memory_required, device, keep_loaded=(), *args, **kwargs):
        keep_loaded = list(keep_loaded)
        unloaded = core_free_memory(memory_required, device, keep_loaded, *args, **kwargs)
        if device is None or torch.device(device).type != "cuda":
            return unloaded
        target = torch.device(device)
        for loaded in list(reversed(model_management.current_loaded_models)):
            if model_management.get_free_memory(target) >= memory_required:
                break
            if loaded in keep_loaded or torch.device(loaded.device) == target or loaded.is_dead():
                continue
            inner = loaded.model.model
            metadata = getattr(inner, "_distorch_v2_meta", None)
            if not metadata or not metadata.get("full_allocation"):
                continue
            if resident_storage_bytes(inner).get(target, 0) == 0:
                continue
            # Core's per-device index omits weights held on donor GPUs. A full
            # detach is necessary: partial unload accounts only for load_device.
            if loaded.model_unload():
                model_management.current_loaded_models.remove(loaded)
                unloaded.append(loaded)
                model_management.soft_empty_cache()
        return unloaded

    return free_memory


def guard_quantized_empty_like(core_handler, pause_malloc_graph):
    """Keep donor metadata clones outside a compute-device allocation graph."""

    @wraps(core_handler)
    def empty_like(weight, args, kwargs):
        source = torch.device(weight.device)
        device = kwargs.get("device")
        target = torch.device(device) if device is not None else source
        if source.type == target.type == "cuda" and source != target:
            # Kitchen clones scale/rotation metadata on the source GPU before
            # moving it to the target. The recorder belongs to the target GPU.
            with pause_malloc_graph():
                return core_handler(weight, args, kwargs)
        return core_handler(weight, args, kwargs)

    setattr(empty_like, "_multigpu_malloc_graph_guard", True)
    return empty_like


def register_quantized_allocation_guard(tensor_base, pause_malloc_graph):
    """Patch the generic handler once, without touching core's protected cast_to."""
    table = getattr(tensor_base, "_DISPATCH_TABLE", None)
    if not isinstance(table, dict):
        return False
    operation = torch.ops.aten.empty_like.default
    handler = table.get(operation)
    if not callable(handler):
        return False
    if getattr(handler, "_multigpu_malloc_graph_guard", False):
        return True
    table[operation] = guard_quantized_empty_like(handler, pause_malloc_graph)
    return True
