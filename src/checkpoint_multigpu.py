import hashlib
import logging
from typing import Any

import comfy.clip_vision
import comfy.model_detection
import comfy.model_management as mm
import comfy.sd
import comfy.storage
import comfy.utils
import torch
from comfy.sd import CLIP, VAE

from .device_utils import get_device_list, soft_empty_cache_multigpu
from .distorch_2 import register_patched_safetensor_modelpatcher
from .model_management_mgpu import multigpu_memory_log

logger: Any = logging.getLogger("MultiGPU")


def load_checkpoint_with_placement(
    ckpt_path, device_config, distorch_config=None, component=None, disable_dynamic=False
):
    """Load once and retain placement in core's per-device reload factory contract."""
    import folder_paths

    sd, metadata = comfy.utils.load_torch_file(ckpt_path, return_metadata=True)
    out = patched_load_state_dict_guess_config(
        sd,
        output_model=component in (None, 0),
        output_clip=component in (None, 1),
        output_vae=component in (None, 2),
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        metadata=metadata,
        disable_dynamic=disable_dynamic,
        device_config=device_config,
        distorch_config=distorch_config,
    )
    if out is None:
        raise RuntimeError(f"Could not detect checkpoint model type: {ckpt_path}")
    for index, value in enumerate(out[:3]):
        if value is None:
            continue
        patcher = value if index == 0 else getattr(value, "patcher", None)
        if patcher is None:
            continue
        patcher.cached_patcher_init = (
            load_checkpoint_with_placement,
            (ckpt_path, dict(device_config), dict(distorch_config or {}), index),
        )
    if component is not None:
        value = out[component]
        return value if component == 0 else value.patcher
    return out[:3]


def patched_load_state_dict_guess_config(
    sd,
    output_vae=True,
    output_clip=True,
    output_clipvision=False,
    embedding_directory=None,
    output_model=True,
    model_options=None,
    te_model_options=None,
    metadata=None,
    disable_dynamic=False,
    *,
    device_config=None,
    distorch_config=None,
):
    """Checkpoint loader with MultiGPU and DisTorch2 device placement support."""
    from . import (
        get_current_device,
        get_current_text_encoder_device,
        set_current_device,
        set_current_text_encoder_device,
    )

    model_options = {} if model_options is None else model_options
    te_model_options = {} if te_model_options is None else te_model_options
    logger.debug("[MultiGPU Checkpoint] ENTERING Patched Checkpoint Loader")
    logger.debug(f"[MultiGPU Checkpoint] Received Device Config: {device_config}")
    logger.debug(f"[MultiGPU Checkpoint] Received DisTorch2 Config: {distorch_config}")

    device_config = device_config or {}
    unet_disable_dynamic = disable_dynamic or bool(
        distorch_config and distorch_config.get("unet_allocation")
    )
    clip_disable_dynamic = disable_dynamic or bool(
        distorch_config and distorch_config.get("clip_allocation")
    )
    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None

    # Capture the current devices at runtime so we can restore them after loading
    original_main_device = get_current_device()
    original_clip_device = get_current_text_encoder_device()

    try:
        diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
        parameters = comfy.utils.calculate_parameters(sd, diffusion_model_prefix)
        weight_dtype = comfy.utils.weight_dtype(sd, diffusion_model_prefix)

        custom_operations = model_options.get("custom_operations", None)
        if custom_operations is None:
            sd, metadata = comfy.utils.convert_old_quants(
                sd, diffusion_model_prefix, metadata=metadata
            )

        model_config = comfy.model_detection.model_config_from_unet(
            sd, diffusion_model_prefix, metadata=metadata
        )

        if model_config is None:
            logger.warning(
                "[MultiGPU] Warning: Not a standard checkpoint file. Trying to load as diffusion model only."
            )
            # Simplified fallback for non-checkpoints
            set_current_device(device_config.get("unet_device", original_main_device))
            diffusion_model = comfy.sd.load_diffusion_model_state_dict(
                sd,
                model_options=model_options,
                metadata=metadata,
                disable_dynamic=unet_disable_dynamic,
            )
            if diffusion_model is None:
                return None
            return (diffusion_model, None, VAE(sd={}), None)

        logger.debug(
            f"[MultiGPU] Detected Model Config: {type(model_config).__name__}, Parameters: {parameters / 10**9:.2f}B"
        )

        unet_weight_dtype = list(model_config.supported_inference_dtypes)
        if model_config.quant_config is not None:
            weight_dtype = None

        if custom_operations is not None:
            model_config.custom_operations = custom_operations
        unet_compute_device = torch.device(
            device_config.get("unet_device", model_options.get("load_device", original_main_device))
        )
        unet_dtype = model_options.get("dtype", model_options.get("weight_dtype", None))
        if unet_dtype is None:
            unet_dtype = mm.unet_dtype(
                device=unet_compute_device,
                model_params=parameters,
                supported_dtypes=unet_weight_dtype,
                weight_dtype=weight_dtype,
            )

        if model_config.quant_config is not None:
            manual_cast_dtype = mm.unet_manual_cast(
                None, unet_compute_device, model_config.supported_inference_dtypes
            )
        else:
            manual_cast_dtype = mm.unet_manual_cast(
                unet_dtype, unet_compute_device, model_config.supported_inference_dtypes
            )
        model_config.set_inference_dtype(unet_dtype, manual_cast_dtype, device=unet_compute_device)
        logger.info(f"UNet DType: {unet_dtype}, Manual Cast: {manual_cast_dtype}")

        if model_config.clip_vision_prefix is not None and output_clipvision:
            clipvision = comfy.clip_vision.load_clipvision_from_sd(
                sd, model_config.clip_vision_prefix, True
            )

        if output_model:
            set_current_device(unet_compute_device)
            inital_load_device = mm.unet_inital_load_device(parameters, unet_dtype)

            multigpu_memory_log("checkpoint_unet", "pre-load")

            model = model_config.get_model(sd, diffusion_model_prefix, device=inital_load_device)
            model_patcher_class = (
                comfy.model_patcher.ModelPatcher
                if unet_disable_dynamic
                else comfy.model_patcher.CoreModelPatcher
            )
            model_patcher = model_patcher_class(
                model,
                load_device=unet_compute_device,
                offload_device=model_options.get("offload_device", mm.unet_offload_device()),
                fast_disk=comfy.storage.state_dict_fast_disk(sd),
            )
            model.load_model_weights(sd, diffusion_model_prefix, assign=model_patcher.is_dynamic())
            multigpu_memory_log("checkpoint_unet", "post-weights")

            logger.mgpu_mm_log("Invoking soft_empty_cache_multigpu before UNet ModelPatcher setup")
            soft_empty_cache_multigpu()
            multigpu_memory_log("checkpoint_unet", "post-model")

            if distorch_config and "unet_allocation" in distorch_config:
                unet_alloc = distorch_config["unet_allocation"]
                if unet_alloc:
                    register_patched_safetensor_modelpatcher()
                    inner_model = model_patcher.model
                    inner_model._distorch_v2_meta = {
                        "full_allocation": unet_alloc,
                        "eject_models": distorch_config.get("eject_models", True),
                    }
                    logger.info(f"[CHECKPOINT_META] UNET inner_model id=0x{id(inner_model):x}")
                    model._distorch_high_precision_loras = distorch_config.get(
                        "high_precision_loras", True
                    )

        if output_vae:
            vae_target_device = torch.device(device_config.get("vae_device", original_main_device))
            set_current_device(vae_target_device)  # Use main device context for VAE
            multigpu_memory_log("checkpoint_vae", "pre-load")

            vae_sd = comfy.utils.state_dict_prefix_replace(
                sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True
            )
            vae_sd = model_config.process_vae_state_dict(vae_sd)
            vae = VAE(sd=vae_sd, metadata=metadata, device=vae_target_device)
            multigpu_memory_log("checkpoint_vae", "post-load")

        if output_clip:
            if te_model_options.get("custom_operations", None) is None:
                scaled_fp8_list = []
                for k in list(sd.keys()):  # Convert scaled fp8 to mixed ops
                    if k.endswith(".scaled_fp8"):
                        scaled_fp8_list.append(k[: -len("scaled_fp8")])

                if len(scaled_fp8_list) > 0:
                    out_sd = {}
                    for k in sd:
                        skip = False
                        for pref in scaled_fp8_list:
                            skip = skip or k.startswith(pref)
                        if not skip:
                            out_sd[k] = sd[k]

                    for pref in scaled_fp8_list:
                        quant_sd, _qmetadata = comfy.utils.convert_old_quants(sd, pref, metadata={})
                        for k in quant_sd:
                            out_sd[k] = quant_sd[k]
                        sd = out_sd

            clip_target_device = torch.device(
                device_config.get("clip_device", original_clip_device)
            )
            set_current_text_encoder_device(clip_target_device)

            clip_target = model_config.clip_target(state_dict=sd)
            if clip_target is not None:
                clip_sd = model_config.process_clip_state_dict(sd)
                if len(clip_sd) > 0:
                    logger.debug(
                        "[MultiGPU Checkpoint] Invoking soft_empty_cache_multigpu before CLIP construction"
                    )
                    multigpu_memory_log("checkpoint_clip", "pre-load")
                    soft_empty_cache_multigpu()
                    clip_params = comfy.utils.calculate_parameters(clip_sd)
                    clip = CLIP(
                        clip_target,
                        embedding_directory=embedding_directory,
                        tokenizer_data=clip_sd,
                        parameters=clip_params,
                        state_dict=clip_sd,
                        model_options=te_model_options,
                        disable_dynamic=clip_disable_dynamic,
                    )

                    if distorch_config and "clip_allocation" in distorch_config:
                        clip_alloc = distorch_config["clip_allocation"]
                        if clip_alloc and hasattr(clip, "patcher"):
                            register_patched_safetensor_modelpatcher()
                            inner_clip = clip.patcher.model
                            inner_clip._distorch_v2_meta = {
                                "full_allocation": clip_alloc,
                                "eject_models": False,
                            }
                            logger.info(
                                f"[CHECKPOINT_META] CLIP inner_model id=0x{id(inner_clip):x}"
                            )
                            clip.patcher.model._distorch_high_precision_loras = distorch_config.get(
                                "high_precision_loras", True
                            )

                    logger.info("CLIP Loaded.")
                    multigpu_memory_log("checkpoint_clip", "post-load")
                else:
                    logger.warning("No CLIP/text encoder weights in checkpoint.")
            else:
                logger.warning("CLIP target not found in model config.")

        if output_model and inital_load_device != torch.device("cpu"):
            mm.load_models_gpu([model_patcher], force_full_load=True)

    finally:
        set_current_device(original_main_device)
        set_current_text_encoder_device(original_clip_device)
    return (model_patcher, clip, vae, clipvision)


class CheckpointLoaderAdvancedMultiGPU:
    @classmethod
    def INPUT_TYPES(node_cls):
        import folder_paths

        devices = get_device_list()
        default_device = devices[1] if len(devices) > 1 else devices[0]

        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "unet_device": (devices, {"default": default_device}),
                "clip_device": (devices, {"default": default_device}),
                "vae_device": (devices, {"default": default_device}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "multigpu"
    TITLE = "Checkpoint Loader Advanced (MultiGPU)"

    def load_checkpoint(self, ckpt_name, unet_device, clip_device, vae_device):
        import folder_paths

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        device_config = {
            "unet_device": unet_device,
            "clip_device": clip_device,
            "vae_device": vae_device,
        }

        return load_checkpoint_with_placement(ckpt_path, device_config)


class CheckpointLoaderAdvancedDisTorch2MultiGPU:
    @classmethod
    def INPUT_TYPES(node_cls):
        import folder_paths

        devices = get_device_list()
        compute_device = devices[1] if len(devices) > 1 else devices[0]

        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "unet_compute_device": (devices, {"default": compute_device}),
                "unet_virtual_vram_gb": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.0, "max": 128.0, "step": 0.1},
                ),
                "unet_donor_device": (devices, {"default": "cpu"}),
                "clip_compute_device": (devices, {"default": "cpu"}),
                "clip_virtual_vram_gb": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.0, "max": 128.0, "step": 0.1},
                ),
                "clip_donor_device": (devices, {"default": "cpu"}),
                "vae_device": (devices, {"default": compute_device}),
            },
            "optional": {
                "unet_expert_mode_allocations": ("STRING", {"multiline": False, "default": ""}),
                "clip_expert_mode_allocations": ("STRING", {"multiline": False, "default": ""}),
                "high_precision_loras": ("BOOLEAN", {"default": True}),
                "eject_models": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "multigpu/distorch_2"
    TITLE = "Checkpoint Loader Advanced (DisTorch2)"

    def load_checkpoint(
        self,
        ckpt_name,
        unet_compute_device,
        unet_virtual_vram_gb,
        unet_donor_device,
        clip_compute_device,
        clip_virtual_vram_gb,
        clip_donor_device,
        vae_device,
        unet_expert_mode_allocations="",
        clip_expert_mode_allocations="",
        high_precision_loras=True,
        eject_models=True,
    ):

        import folder_paths

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        device_config = {
            "unet_device": unet_compute_device,
            "clip_device": clip_compute_device,
            "vae_device": vae_device,
        }

        unet_vram_str = ""
        if unet_virtual_vram_gb > 0:
            unet_vram_str = f"{unet_compute_device};{unet_virtual_vram_gb};{unet_donor_device}"
        elif unet_expert_mode_allocations:
            unet_vram_str = unet_compute_device
        unet_alloc = (
            f"{unet_expert_mode_allocations}#{unet_vram_str}"
            if unet_expert_mode_allocations or unet_vram_str
            else ""
        )

        clip_vram_str = ""
        if clip_virtual_vram_gb > 0:
            clip_vram_str = f"{clip_compute_device};{clip_virtual_vram_gb};{clip_donor_device}"
        elif clip_expert_mode_allocations:
            clip_vram_str = clip_compute_device
        clip_alloc = (
            f"{clip_expert_mode_allocations}#{clip_vram_str}"
            if clip_expert_mode_allocations or clip_vram_str
            else ""
        )

        distorch_config = {
            "unet_allocation": unet_alloc,
            "clip_allocation": clip_alloc,
            "high_precision_loras": high_precision_loras,
            "eject_models": eject_models,
            "unet_settings": hashlib.sha256(
                f"{unet_alloc}{high_precision_loras}".encode()
            ).hexdigest(),
            "clip_settings": hashlib.sha256(
                f"{clip_alloc}{high_precision_loras}".encode()
            ).hexdigest(),
        }

        return load_checkpoint_with_placement(ckpt_path, device_config, distorch_config)
