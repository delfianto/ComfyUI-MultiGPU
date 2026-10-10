"""Import the fork against a real stable ComfyUI checkout and validate patched APIs."""

import argparse
import asyncio
import inspect
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("comfyui", type=Path)
    args = parser.parse_args()
    core = args.comfyui.resolve()
    sys.path.insert(0, str(core))
    # Core parses argv while importing model management. No GPU is required here.
    sys.argv = [str(core / "main.py"), "--cpu"]
    import comfy.model_management as mm
    import comfy.model_patcher as mp
    import comfy.sd
    import folder_paths
    import torch
    from comfy.supported_models_base import BASE

    import nodes

    required = {
        "CheckpointLoaderAdvancedMultiGPU",
        "CheckpointLoaderAdvancedDisTorch2MultiGPU",
        "UNETLoaderMultiGPU",
        "VAELoaderMultiGPU",
        "CLIPLoaderMultiGPU",
        "DualCLIPLoaderMultiGPU",
        "TripleCLIPLoaderMultiGPU",
        "QuadrupleCLIPLoaderMultiGPU",
        "CLIPVisionLoaderMultiGPU",
        "UNETLoaderDisTorch2MultiGPU",
        "CLIPLoaderDisTorch2MultiGPU",
        "CheckpointLoaderSimpleDisTorch2MultiGPU",
    }
    with tempfile.TemporaryDirectory(prefix="mgpu-api-") as directory:
        Path(directory, "ComfyUI-MultiGPU").symlink_to(Path(__file__).resolve().parents[1])
        folder_paths.add_model_folder_path("custom_nodes", directory)
        asyncio.run(nodes.init_extra_nodes(init_api_nodes=False))
    missing = required - nodes.NODE_CLASS_MAPPINGS.keys()
    if missing:
        raise AssertionError(f"Missing MultiGPU loaders: {sorted(missing)}")
    for name in required:
        node = nodes.NODE_CLASS_MAPPINGS[name]
        schema = node.INPUT_TYPES()
        assert schema["required"], name
        assert callable(getattr(node, node.FUNCTION)), name
    implementation = next(
        module
        for name, module in tuple(sys.modules.items())
        if name.endswith("ComfyUI-MultiGPU.src")
    )
    implementation.register_patched_safetensor_modelpatcher()
    assert set(inspect.signature(mm.load_models_gpu).parameters) == {
        "models",
        "memory_required",
        "force_patch_weights",
        "minimum_memory_required",
        "force_full_load",
    }
    assert set(inspect.signature(mm.free_memory).parameters) == {
        "memory_required",
        "device",
        "keep_loaded",
        "for_dynamic",
        "pins_required",
        "ram_required",
    }
    assert set(inspect.signature(mp.ModelPatcher.partially_load).parameters) == {
        "self",
        "device_to",
        "extra_memory",
        "force_patch_weights",
    }
    assert "fast_disk" in inspect.signature(mp.ModelPatcher).parameters
    assert "device" in inspect.signature(BASE.set_inference_dtype).parameters
    assert hasattr(BASE, "quant_config")
    assert "device" in inspect.signature(comfy.sd.VAE).parameters
    # This must still delegate to core, despite registering DisTorch's patches.
    mm.load_models_gpu([])
    # Execute the annotated branch with real core patchers and tiny CPU weights.
    # Signatures alone cannot detect calls made on the wrong wrapper object.
    model = torch.nn.Linear(2, 2)
    setattr(model, "_distorch_v2_meta", {"full_allocation": "cpu,1#cpu", "eject_models": False})
    patcher = mp.ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    mm.load_models_gpu([patcher], force_full_load=True)
    assert model.weight.device.type == "cpu"
    mm.unload_all_models()
    print(f"Stable API import passed: {len(required)} loader schemas and patched core signatures")


if __name__ == "__main__":
    main()
