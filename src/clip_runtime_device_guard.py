"""Keep MultiGPU's logical execution device aligned with the CLIP patcher."""

from functools import wraps

import comfy.sd


def install_clip_runtime_device_guard(runtime_guard):
    # Loader overrides choose CLIP's load_device, but restore the global device
    # afterward. FP8 comfy_kitchen operations consult that global device while
    # encoding; it must follow CLIP rather than the diffusion model's GPU.
    for method_name in ("encode_from_tokens_scheduled", "encode_from_tokens"):
        original = getattr(comfy.sd.CLIP, method_name)
        if getattr(original, "_multigpu_clip_runtime_guard", False):
            continue

        def wrap_encode(fn):
            @wraps(fn)
            def encode(self, *args, **kwargs):
                with runtime_guard(self.patcher.load_device, reason=f"CLIP.{fn.__name__}"):
                    return fn(self, *args, **kwargs)

            setattr(encode, "_multigpu_clip_runtime_guard", True)
            return encode

        setattr(comfy.sd.CLIP, method_name, wrap_encode(original))


def install_vae_runtime_device_guard(runtime_guard):
    for method_name in ("encode", "decode", "encode_tiled", "decode_tiled"):
        original = getattr(comfy.sd.VAE, method_name, None)
        if original is None or getattr(original, "_multigpu_vae_runtime_guard", False):
            continue

        def wrap_vae(fn):
            @wraps(fn)
            def run(self, *args, **kwargs):
                with runtime_guard(self.device, reason=f"VAE.{fn.__name__}"):
                    return fn(self, *args, **kwargs)

            setattr(run, "_multigpu_vae_runtime_guard", True)
            return run

        setattr(comfy.sd.VAE, method_name, wrap_vae(original))
