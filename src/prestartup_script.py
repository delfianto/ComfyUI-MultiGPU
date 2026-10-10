"""Initialize aimdo's visible devices together, before ComfyUI creates patchers."""

import logging
from functools import wraps


def install_aimdo_device_init_guard(control, cuda=None) -> bool:
    original = control.init_device
    if getattr(original, "_multigpu_all_devices", False):
        return False

    @wraps(original)
    def init_device(device_id, extra_vram_headroom=0):
        if cuda is None:
            import torch

            cuda_api = torch.cuda
        else:
            cuda_api = cuda
        # aimdo only supports one initialization of its device-context table.
        # Never deinitialize live contexts or claim a missing context is ready.
        if not cuda_api.is_available() or cuda_api.device_count() <= 1 or control.devctxs:
            return original(device_id, extra_vram_headroom=extra_vram_headroom)
        devices = [
            (index, extra_vram_headroom) if index == device_id else index
            for index in range(cuda_api.device_count())
        ]
        return control.init_devices(devices)

    setattr(init_device, "_multigpu_all_devices", True)
    control.init_device = init_device
    return True


try:
    from comfy_aimdo import control

    install_aimdo_device_init_guard(control)
except (ImportError, AttributeError):
    logging.getLogger("MultiGPU").debug("aimdo multi-device initialization is unavailable")
