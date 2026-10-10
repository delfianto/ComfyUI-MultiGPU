# Wan Animate 2 cache regression

These are **ComfyUI API prompts**, adapted from the official
[Motion Transfer (Wan Animate 2 Distilled) blueprint](https://github.com/Comfy-Org/workflow_templates/blob/main/blueprints/motion_transfer_wan_animate_2_distilled.json)
to investigate upstream [#219](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/219).
They use native WanAnimate nodes and this fork's model loaders. No WanVideoWrapper
or pose-extraction provider is required.

## Files and inputs

The prompts use this personal setup's slug filenames:

| Model | Directory under ComfyUI `models/` |
| --- | --- |
| `wan-animate-2-distill-int8-convrot.safetensors` | `diffusion_models/` |
| `umt5-xxl-fp8-e4m3fn-scaled.safetensors` | `clip/` or `text_encoders/` |
| `clip-vision-h.safetensors` | `clip_vision/` |
| `wan-v2.1-vae-bf16.safetensors` | `vae/` |

Download the weights from [Comfy-Org/Wan-Animate-2](https://huggingface.co/Comfy-Org/Wan-Animate-2)
and rename them to match, or edit the four loader nodes in each prompt.

Place the official example inputs in ComfyUI's `input/` directory using these names:

- [pink_hair_mech_arms_ref.png](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/pink_hair_mech_arms_ref.png)
  as `wananimate-test-pink_hair_mech_arms_ref.png`.
- [street_dance_drive.mp4](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/input/street_dance_drive.mp4)
  as `wananimate-test-street_dance_drive.mp4`.

## Comparison cases

All cases use seed 219, 480x832, ten LCM/simple steps, CFG 1, and sampling shift 5.
Diffusion computes on cuda:0; UMT5, CLIP vision, and the VAE use cuda:1.
The driving video's first 49 or 81 frames are sampled. Each output is an animated
WebP under `output/wananimate-issue219/`.

- [cpu-default-49f.json](cpu-default-49f.json): 49-frame CPU/default cache control.
- [gpu-int8-49f.json](gpu-int8-49f.json): same 49-frame graph with GPU/int8 cache.
- [cpu-default-81f.json](cpu-default-81f.json): larger CPU/default cache control.

Run the cases sequentially. Our larger GPU/int8 case with the ordinary loader
exhausted the 4080's VRAM. Explicit DisTorch2 splits also exhausted VRAM during
activation allocation, despite reporting about 8 GiB of weights assigned to
the 3060. Neither failing configuration is included as a recommended example.
Use CPU/default cache for 81 frames on this setup. These are tested personal
configurations, not general memory guarantees or performance benchmarks.

The existing `ci/run_workflows.py` runner accepts these prompts. Its environment
needs `requests` and `websocket-client`; run it against the existing server rather
than importing these files as Python tests. For example, from the repository root:

```sh
python ci/run_workflows.py --host 127.0.0.1 --port 8188 --workflow-timeout 1800 \
  docs/example_workflows/wan-animate-2/cpu-default-49f.json \
  docs/example_workflows/wan-animate-2/gpu-int8-49f.json
```

Hardware runs are manual and are not part of the CPU-only GitHub quality job.
Results and limits are recorded in [MAINTENANCE.md](../../MAINTENANCE.md).
