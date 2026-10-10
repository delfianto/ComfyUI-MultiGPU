# Personal fork maintenance

[Main README](../README.md) · [Upstream usage guide](README.md)

This checkout belongs to https://github.com/delfianto/ComfyUI-MultiGPU.
The original project is archived; preserve its GPL-3.0 license and attribution.
The original registry PublisherId remains historical metadata, not a claim that
this fork can publish releases under the original maintainer's identity.

## Hardware and runtime

This is the local machine used for the generation tests below:

| Component | Local setup |
| --- | --- |
| CPU | AMD Ryzen 9 9950X, 16 cores / 32 threads |
| System RAM | About 60 GiB usable, as reported by Linux |
| Main GPU (`cuda:0`) | NVIDIA GeForce RTX 4080, 16 GB VRAM |
| Encoder/donor GPU (`cuda:1`) | NVIDIA GeForce RTX 3060, 12 GB VRAM |
| NVIDIA driver | 615.78.08, observed when documenting this setup |
| Host OS / deployment | CachyOS Linux, rootless Docker, both GPUs visible to ComfyUI |
| CUDA image | NVIDIA CUDA 13.4.2 base, Ubuntu 24.04 |
| PyTorch | 2.11.0+cu130 (wheel CUDA runtime 13.0) |
| ComfyUI | 0.39.0 |
| comfy-kitchen / comfy-aimdo | 0.2.37 / 0.5.6 |

Diffusion normally runs on the RTX 4080; CLIP and VAE run on the RTX 3060.
DisTorch2 can also borrow model storage from the 3060. These cards have no usable
CUDA peer-access path in this setup, so transfer overhead matters.

## Tested model configurations

These are completed generation tests, not a list of everything installed:

| Configuration | Models and components | Test and result |
| --- | --- | --- |
| Krea2 RAW FP8 + LoRA + DyPE | `krea2-raw-fp8.safetensors`; `qwen3-vl-4b-fp8-scaled.safetensors`; `qwen-image-vae.safetensors`; Karina LoRA `aespakarina_krea2v3_27_qfloat8.safetensors` at 0.9 | 512×512 smoke test and 896×1200, 25-step generation passed; full run took 68.14 seconds after switching from FLUX. |
| FLUX.1-dev FP8 + DisTorch2 + DyPE | `flux1-dev-fp8.safetensors`; `clip-l.safetensors` + `t5xxl-fp8-e4m3fn-scaled.safetensors`; `ae.safetensors` | 512×512, 2-step generation passed; measured donor placement was 4080.60 MiB on the 3060, with 7674.75 MiB on the 4080. |

CLIP/VAE used `cuda:1` in both configurations. The FLUX test confirms execution
and donor placement; a 2-step smoke test does not establish final image quality.
Qwen Image 2.1, other checkpoints, Windows hardware, and optional video providers
have not been validated by these generation runs.

## Development checks

Run from the **repository root**, using a Python environment with Torch and
packaging (e.g. ComfyUI's environment).

```sh
uv sync --group dev --no-install-project
ruff check .
ruff format --check .
basedpyright -p pyproject.toml --warnings
python -m unittest discover -s tests -v
```

The dev group supplies lint/type tools; install Torch separately or use the host
application's environment. Ruff checks imports, undefined names, common errors,
bugbear and Python modernization. basedpyright uses basic mode across the Python
codebase, keeping argument, return, optional-value and attribute checks enabled.
Missing host/optional-provider imports are permitted. [typings/nodes.pyi](../typings/nodes.pyi) describes
the core registry so the local `nodes.py` is not mistaken for ComfyUI's module.
Dynamic logger methods use an explicit Any boundary; errors are not globally
suppressed. The classmethod receiver naming rule is disabled for wrapper classes
that capture a different class in their closure.

## Upstream triage

[Archive thread #223](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/223)
identifies proposed PRs, not validated fixes. Priority is process crashes first,
then incorrect GPU allocation, then optional-provider compatibility.

| Priority | Reports | Change and validation |
| --- | --- | --- |
| Critical | #187, #210, #218, #220; PR #216 | Prefer Torch's peer API; resolve versioned/Linux and Windows runtimes; all probe failures return false. CPU regressions cover success, failures, caching and platform candidates. Windows runtime execution still needs a Windows rig. |
| Critical | #211, #213, #222 | Keep comfy-kitchen >=0.2.35 native DLPack exports. Moving tensors inside a per-argument export wrapper can conflict with the kernel's actual device/stream. Older kitchen versions retain the guarded compatibility path and are not validated here. |
| Critical | Local FP8 CLIP crash; #189/#185 related symptoms | Guard direct/scheduled CLIP and standard/tiled VAE inference with their real load device, restoring device state on exceptions. CLIP nesting and VAE regressions plus local dual-GPU generation. Similar symptoms alone do not establish that every upstream reproduction is fixed. |
| High | #202/#186; PR #195 | Initialize visible aimdo devices together before core startup. Preserve the primary GPU's headroom; never reinitialize live contexts or report missing devices as initialized. Defer Torch import until core calls init_device. Existing per-device legacy fallback remains. |
| High | #191 | DisTorch2 uses legacy ModelPatcher while loading an explicitly distributed model; DynamicVRAM otherwise bypasses donor placement in the patched legacy loading path. Ordinary loaders keep DynamicVRAM. Preserve patcher alias and loader device state on errors. |
| High | #196; PR #198 | Count unique underlying model parameters by physical nbytes without evaluating parametrized weight properties. Tests cover weight_norm, tied parameters, and packed storage. Buffers and allocator overhead are not included in the parameter estimate. |
| Medium | #188; PR #200 | Florence2 delegates INPUT_TYPES and forwards args/kwargs so schema changes and upstream model_paths initialization stay intact. Mock provider regressions; Florence2 is not installed locally. |
| Medium | PR #217 | Recognize ComfyUI-GGUF-Loader as an additional provider. Existing city96 provider remains supported. The alternative provider is not installed locally. |
| Medium | PR #182 | Copy the schema and modified optional map without recursively copying metadata. Regression checks upstream schema isolation with noncopyable metadata. |

Other open reports remain tracked upstream. #219's WanAnimate2Cache int8 crash,
#203's async-provider signature changes, repeat-run model corruption (#185),
and WanVideo/LTX feature requests need their specific providers/models and
reproductions before claiming fixes. This fork does not claim to resolve all open
issues. The LTXV2 feature PR #164 is not merged as part of these crash fixes.

## Source layout

Runtime Python modules live in `src/`; `src/__init__.py` registers nodes and
installs the compatibility patches. Tests remain in `tests/`, CI utilities in
`ci/`, and type contracts in `typings/`.

Two small entry files remain at the repository root because ComfyUI discovers
them there:

- `__init__.py` exposes the node mappings and forwards historical helper
  attributes to `src`. Its `WEB_DIRECTORY` points to `docs/web/`, with node
  help pages in `docs/web/docs/` to preserve ComfyUI's `docs/<node>.md` help
  route. Documentation assets live in `docs/assets/`.
- `prestartup_script.py` executes `src/prestartup_script.py` directly. It avoids
  importing the runtime package or Torch before core configures CUDA visibility.

Serialized node names and workflow loader names are unchanged.

## Deployment

The service bind-mounts this directory at `/data/custom_nodes/ComfyUI-MultiGPU`.
Python changes require `composectl restart ai-comfyui --json`; image rebuilding is
unnecessary for node edits. The Docker image owns GGUF only. Preserve serialized
node names so existing workflows remain compatible. Model runs use RTX 4080
(cuda:0), with CLIP/VAE on RTX 3060 (cuda:1) wherever practical.

Reviewed proposals: [#216](https://github.com/pollockjj/ComfyUI-MultiGPU/pull/216),
[#198](https://github.com/pollockjj/ComfyUI-MultiGPU/pull/198),
[#200](https://github.com/pollockjj/ComfyUI-MultiGPU/pull/200),
[#195](https://github.com/pollockjj/ComfyUI-MultiGPU/pull/195),
[#182](https://github.com/pollockjj/ComfyUI-MultiGPU/pull/182), and
[#217](https://github.com/pollockjj/ComfyUI-MultiGPU/pull/217).
The P2P, Florence2, and schema-copy proposals were adapted; memory accounting and
prestartup initialization were implemented with additional local safeguards.

## Local validation (2026-10-10)

Rebuilt image `78549b8cbea2`, ComfyUI 0.39.0, Torch 2.11.0+cu130,
comfy-kitchen 0.2.37, comfy-aimdo 0.5.6. Service restarted through composectl;
Docker health is healthy. MultiGPU loads only from the persistent checkout.
Both GPUs have initialized aimdo contexts, and modern DLPack is unpatched.

- Ruff check and format check pass; basedpyright reports zero errors/warnings.
- All 22 CPU regressions pass both on the host and inside the container.
- FLUX.1-dev FP8 + DyPE, 512x512, 2 steps: succeeds. Actual layer assignment
  places 4080.60 MiB on cuda:1 and 7674.75 MiB on cuda:0. CLIP/VAE use cuda:1.
- Krea2 RAW FP8 + Karina LoRA 0.9 + DyPE, 896x1200, 25 steps: succeeds in
  68.14 seconds after switching from FLUX; inspected output is a valid portrait.
- The FLUX workflow now uses a diffusion loader plus separate CLIP-L/T5 and ae
  VAE loaders. Its previous checkpoint loader referenced a diffusion-only file.

Prompts, history, validation results, and the portrait are saved under
`/srv/appdata/comfyui/user/default/workflow-backups/multigpu-fork-migration/`.
These runs cover the installed image models, not every optional video provider.

Editor diagnostic cleanup: CLI/CI now enable unused-variable, unused-parameter,
and deprecated-API warnings, matching the tagged hints exposed by Zed. Intentional
compatibility parameters retain their public keyword names. LTX helper loaders
now forward config instead of discarding it, covered by a regression test.

After the src/ reorganization, ComfyUI was restarted and all 44 MultiGPU node
names, schemas, outputs and categories matched the previous registration. The
root web documentation route returned HTTP 200. All 22 regression tests passed
on the host and in the container; Ruff/basedpyright and the editor language server
were clean. A fresh 512×512 Krea2 + LoRA + DyPE smoke generation completed in
8.65 seconds with the model on cuda:0 and CLIP/VAE on cuda:1.
