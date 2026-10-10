# ComfyUI-MultiGPU

This is my **personal fork** of the archived
[pollockjj/ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU),
maintained for my own ComfyUI setup.

## What this does

Adds device selection to ComfyUI loaders so diffusion, CLIP/text encoders, and
VAEs can run on different GPUs. DisTorch2 can store parts of a model on another
GPU or in system RAM to free VRAM on the compute GPU. Workflow steps still run
in sequence.

## Use and support

**DISCLAIMER: I am NOT BY ANY MEANS a Python guru or GPU kernel hacker. Most fixes
were made with an LLM assistant (GPT-6.1 Sol). This works on my machine; support
for other setups or checkpoints is outside scope.**

I run ComfyUI on CachyOS Linux with rootless Docker: an **RTX 4080 16 GB** for
diffusion and an **RTX 3060 12 GB** for CLIP/VAE and donor storage.

Tested configurations include Krea2 RAW FP8 with LoRA and DyPE, FLUX.1-dev FP8
with DisTorch2 and DyPE, FLUX.2 Klein 9B FP8 with repeated encoder switching,
Wan Animate 2 with CPU/default and GPU/int8 cache, MiniMax H3 mixed INT4/INT8,
and LTX-2.5 distilled INT8 ConvRot with audio and two-stage spatial upscaling.
Hardware, runtime versions, model filenames, results, and remaining issues are in
[the maintenance notes](docs/MAINTENANCE.md).

## Installation

Clone into ComfyUI's `custom_nodes` directory, then restart ComfyUI:

```sh
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/delfianto/ComfyUI-MultiGPU.git
```

Keep only one MultiGPU checkout; remove any upstream or image-bundled copy to
avoid duplicate nodes and patches. Optional integrations such as GGUF require
their provider nodes.

## Documentation

- [Usage guide](docs/README.md): the cleaned original README, node catalog,
  allocation modes, and examples.
- [Maintenance notes](docs/MAINTENANCE.md): setup, fixes, issue tracker,
  validation, source layout, and development checks.

## Credits and license

Created by [Alexander Dzhoganov](https://github.com/AlexanderDzhoganov), maintained
upstream by [pollockjj](https://github.com/pollockjj), with contributions from
City96 and others. Licensed under [GPL-3.0](LICENSE).
