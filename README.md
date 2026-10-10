# ComfyUI-MultiGPU — personal maintained fork

This is my **PERSONAL FORK** of
[pollockjj/ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU),
maintained for my own ComfyUI setup. Upstream is archived. I am keeping this copy
working on my machine and fixing the problems I actually run into.

**DISCLAIMER: I am NOT BY ANY MEANS a Python guru or GPU kernel hacker. Most fixes
here were made with an LLM assistant (GPT-6.1 Sol). This works on my machine.
That does not mean it will work on yours.**

## What this does

ComfyUI-MultiGPU adds device selection to model loaders so you can put the
diffusion model, CLIP/text encoder, and VAE on different GPUs. DisTorch2 also lets
you move portions of model storage to another GPU or system RAM to free VRAM on
the compute GPU. It helps with memory capacity; workflow steps still run in
sequence.

My setup uses an **RTX 4080 16 GB** for diffusion and an **RTX 3060 12 GB** for
CLIP/VAE, with the 3060 also acting as a DisTorch2 donor when needed. So far, I
have tested Krea2 RAW FP8 with a character LoRA and DyPE, and FLUX.1-dev FP8 with
DisTorch2 and DyPE. Details, model filenames, software versions, and test limits
are in [docs/MAINTENANCE.md](docs/MAINTENANCE.md).

The local fixes cover encoder execution devices, CUDA/P2P probing, modern
comfy-kitchen compatibility, DynamicVRAM initialization, DisTorch2 allocation
and memory accounting, and some loader compatibility issues. Ruff, basedpyright,
and regression tests keep the changes reviewable. This is ongoing personal
maintenance, with no release schedule or promise to fix every upstream bug.

## Personal use and support

**THIS IS A PERSONAL FORK. If it does not work with your workflow, hardware, or
whatever esoteric checkpoint you use, please do not complain or expect me to
provide support. Use it at your own risk.**

You are welcome to read the code, try it, and fork it. I am sharing what works
here, with no commitment to debug someone else's setup or add support for every
model and custom node. The tested configurations are the scope of my claims.

## Installation

Clone this fork into your ComfyUI `custom_nodes` directory, then restart ComfyUI:

```sh
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/delfianto/ComfyUI-MultiGPU.git
```

Use one MultiGPU checkout. Remove the upstream or image-bundled copy before
loading this fork so duplicate nodes and global patches do not conflict.
Optional integrations such as GGUF require their own provider nodes.

## Source layout

The Python implementation lives in `src/`. The small root `__init__.py` and
`prestartup_script.py` files are ComfyUI entry points. Tests live in `tests/`,
CI utilities in `ci/`, and all documentation in `docs/`. Node help pages in
`docs/web/docs/` are also available through ComfyUI's in-app help. Documentation
images and other assets live in `docs/assets/`; example workflows and their
previews live in `docs/example_workflows/`.

## Documentation

- [docs/README.md](docs/README.md): cleaned original usage guide, node catalog,
  allocation modes, and example workflows. Historical upstream testing is
  identified separately from this fork's tests.
- [docs/MAINTENANCE.md](docs/MAINTENANCE.md): changes, bug fixes, remaining issues,
  hardware, tested models, and development checks.

## Credits and license

Originally created by [Alexander Dzhoganov](https://github.com/AlexanderDzhoganov)
and maintained upstream by [pollockjj](https://github.com/pollockjj), with
contributions from City96 and other upstream contributors. This fork keeps the
original [GPL-3.0 license](LICENSE) and attribution.
