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
| Krea2 RAW FP8 + LoRA + DyPE | `krea2-raw-fp8.safetensors`; `qwen3-vl-4b-fp8-scaled.safetensors`; `qwen-image-vae-bf16.safetensors` (renamed after the test); Karina LoRA `aespakarina_krea2v3_27_qfloat8.safetensors` at 0.9 | 512×512 smoke test and 896×1200, 25-step generation passed; full run took 68.14 seconds after switching from FLUX. |
| FLUX.1-dev FP8 + DisTorch2 + DyPE | `flux1-dev-fp8.safetensors`; `clip-l.safetensors` + `t5xxl-fp8-e4m3fn-scaled.safetensors`; `ae.safetensors` | 512×512, 2-step generation passed; measured donor placement was 4080.60 MiB on the 3060, with 7674.75 MiB on the 4080. |
| Wan Animate 2 distilled INT8 ConvRot | `wan-animate-2-distill-int8-convrot.safetensors`; `umt5-xxl-fp8-e4m3fn-scaled.safetensors`; `clip-vision-h.safetensors`; `wan-v2.1-vae-bf16.safetensors` | 480×832, 10 steps: 49 frames passed with CPU/default and GPU/int8 cache; 81 frames passed with CPU/default cache. See the cache validation below for OOM cases and limits. |

CLIP/VAE used `cuda:1` in these configurations. The FLUX test confirms execution
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
identifies proposed PRs, not validated fixes. Reviewed on 2026-10-10.
Severity is this fork's assessment; upstream does not assign severity labels.
An open upstream report does not establish that the current fork reproduces it.

Status describes this fork: `fixed` means the targeted defect is addressed and
validated; `partial` means related changes exist but coverage or the original
reproduction remains unverified; `open` means no validated fix is available.
The table tracks implemented changes and remaining work together.

| Priority | Reports | Status | Problem | Fix and validation | Next step |
| --- | --- | --- | --- | --- | --- |
| Critical | #187, #210, #218, #220; PR #216 | partial | CUDA peer-access probing can fail or crash. | Prefer Torch's peer API; resolve versioned Linux and Windows runtimes; probe failures return false. CPU regressions cover success, failure, caching, and platform candidates. | Validate runtime execution on Windows hardware. |
| Critical | #211, #213, #222 | partial | Cross-device DLPack exports can conflict with kernel device/stream handling. | Keep comfy-kitchen >=0.2.35 native exports. Older versions retain the compatibility patch and are unvalidated here. | Reproduce affected older-kitchen configurations and inspect tensor lifetime, device, and stream handling. |
| Critical | Local FP8 CLIP crash; #189 related symptoms | partial | CLIP/VAE inference can execute under the wrong current device. | Guard direct/scheduled CLIP and standard/tiled VAE inference with their real load device; restore state on exceptions. CPU regressions and local dual-GPU generation pass. | Validate the original upstream reproduction; similar symptoms do not establish resolution. Repeated-run #185 is tracked separately below. |
| Critical | [#184](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/184); PR #183 | partial | Windows DynamicVRAM stack overflow, including graphs without MultiGPU nodes. | Install the traversal guard at startup for DisTorch2 and core loaders; correct parent classification and shared-module traversal. Deep-graph CPU tests and local empty-allocation DisTorch2 generation pass. See [validation](#dynamic-loading-validation-2026-10-10). | Reproduce the native C-stack failure and verify the fix on the original Windows/ZImage setup. |
| Critical | [#219](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/219) | partial | WanAnimate GPU/int8 cache reportedly kills the process without a Python traceback on Windows/kitchen 0.2.31. | Native DLPack is preserved on kitchen 0.2.37. Local 49-frame GPU/int8 and 81-frame CPU/default generation pass; 81-frame GPU/int8 returns recoverable OOM. No targeted fix for the original native crash is established. See [validation](#wananimate-cache-validation-2026-10-10). | Reproduce the Windows/older-kitchen case in an isolated process; capture native crash diagnostics and inspect active patches, tensor lifetime, device, and streams. |
| Critical | [#212](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/212) | open | A GPU reportedly disappears during MiniMax H3 generation on four RTX 3090s; cause is uncertain. | No targeted fix established. Cross-device fixes may be related, but current evidence cannot establish causation or resolution. | Obtain a reproducible graph, driver/Xid logs, and comparisons with core device-selection nodes on the affected hardware. |
| High | Local code finding: older GGUF wrappers | open | Provider calls occur before device-restoration try/finally; failures can leave global device state changed. | No fix yet. This is a confirmed code finding, not an established cause of a specific upstream report. | Protect the entire operation in src/wrappers.py; test restoration after provider and result-processing failures. **Next implementation item.** |
| High | [#203](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/203) | open | Async/await errors prevent generation; wrappers assume synchronous results. | No fix yet; the report lacks a precise reproducer, so its cause remains unconfirmed. | Identify the affected provider/version; preserve its execution contract and device guards through completion, failure, and cancellation. Test synchronous and async providers. |
| High | [#185](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/185) | partial | Switching encoders on GPU 1 reportedly produces garbled output after two or three runs, particularly Flux.2 Klein FP8. | Runtime device guards address related symptoms; this repeated-run scenario is untested. | Test FP8/GGUF encoder switching, eviction, and reloads; compare conditioning consistency and device state. |
| High | [#209](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/209) | partial | MiniMax H3 mixed INT8/INT4 allocation is accepted while donor VRAM reportedly stays nearly empty. | DynamicVRAM bypass and physical-storage accounting address likely contributors; mixed-quantization checkpoints remain unvalidated. | Measure actual live donor tensor storage and load/unload behavior. Allocation logs and low donor compute utilization do not prove placement. |
| High | #202/#186; PR #195 | fixed | aimdo initialization can omit visible GPUs or disturb existing contexts/headroom. | Initialize visible devices together before core startup; preserve primary headroom; do not reinitialize live contexts or mark missing devices initialized. Defer Torch import until core init_device; retain legacy fallback. CPU regressions pass. | None for the validated initialization defect. |
| High | #191 | fixed | DynamicVRAM bypasses DisTorch2 donor placement through the patched legacy loading path. | Use legacy ModelPatcher when loading explicitly distributed models; ordinary loaders keep DynamicVRAM. Restore patcher alias and loader device state on errors. Local FLUX donor placement and regressions pass. | Mixed-quantization validation is tracked separately under #209. |
| High | #196; PR #198 | fixed | Parameter estimates evaluate parametrized weights or miscount tied/packed storage. | Count unique underlying parameters by physical nbytes. Tests cover weight_norm, tied parameters, and packed storage; buffers and allocator overhead remain outside this estimate. | None for the validated parameter-accounting defect. |
| Medium | #188; PR #200 | partial | Florence2 wrapper schema/arguments drift from the provider. | Delegate INPUT_TYPES and forward args/kwargs, including model_paths. Mock provider regressions pass; Florence2 is not installed locally. | Validate an actual Florence2 provider workflow. |
| Medium | PR #217 | partial | Alternative ComfyUI-GGUF-Loader provider is not recognized. | Recognize the additional provider while retaining city96 support. The alternative provider is not installed locally. | Validate loading with the alternative provider. |
| Medium | PR #182 | fixed | Recursive schema copying can fail on metadata that cannot be deep-copied. | Copy the schema and modified optional map shallowly. Regression verifies upstream schema isolation with noncopyable metadata. | None for the validated schema-copying defect. |

Suggested implementation order: exception-safe GGUF wrappers, async provider
compatibility, repeated-run CLIP validation, then mixed-quantization MiniMax
validation. Confirming the remaining Windows crashes and GPU-disappearance
report requires their specific environments and evidence. The local WanAnimate
validation is documented below; its original native crash remains unverified.

WanVideo/LTX feature requests are outside this crash-fix tracker. LTXV2 PR #164
has not been merged. This fork does not claim to resolve all upstream issues;
this is a personal maintenance backlog, not a support or delivery promise.

## Source layout

Runtime Python modules live in `src/`; `src/__init__.py` registers nodes and
installs the compatibility patches. Tests remain in `tests/`, CI utilities in
`ci/`, and type contracts in `typings/`.

Two small entry files remain at the repository root because ComfyUI discovers
them there:

- `__init__.py` exposes the node mappings and forwards historical helper
  attributes to `src`. Its `WEB_DIRECTORY` points to `docs/web/`, with node
  help pages in `docs/web/docs/` to preserve ComfyUI's `docs/<node>.md` help
  route. Documentation assets live in `docs/assets/`, and the original example
  workflows and previews live in `docs/example_workflows/`.
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
These initial runs cover the installed image models, not every optional video
provider. Native WanAnimate coverage was added separately below.

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

## WanAnimate cache validation (2026-10-10)

Tested upstream [#219](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/219)
with the current fork and runtime above. This tests native WanAnimate 2, not
the older Wan 2.2 Animate model or WanVideoWrapper. The API graph is adapted
from the official distilled motion-transfer blueprint, using its public
reference image and driving video. Context windows are not enabled.

The diffusion model computes on cuda:0; UMT5, CLIP vision, and VAE encoding and
decoding use cuda:1. Full runs use 480×832, seed 219, 10 LCM/simple steps,
CFG 1, and shift 5. The active allocator is cudaMallocAsync, with DynamicVRAM
enabled for ordinary loaders. About 2.5 GiB of primary-GPU memory was already
in use before the tests. Other services were not stopped to free that memory.

| Frames | Cache | Loader / storage | Result | Server execution time |
| --- | --- | --- | --- | --- |
| 49 | CPU / default | Ordinary MultiGPU loader | Passed; 49 output frames | 115.91 s |
| 49 | GPU / int8 | Ordinary MultiGPU loader | Passed; 49 output frames | 98.15 s |
| 81 | CPU / default | Ordinary MultiGPU loader | Passed; 81 output frames | 202.67 s |
| 81 | GPU / int8 | Ordinary MultiGPU loader | Recoverable CUDA OOM during activation allocation | 48.43 s |
| 81 | GPU / int8 | DisTorch2: 3 GiB cuda:0, 8 GiB cuda:1, remainder CPU | Recoverable CUDA OOM | 25.60 s |
| 81 | GPU / int8 | DisTorch2: 1 GiB cuda:0, 8 GiB cuda:1, remainder CPU | Recoverable CUDA OOM | 18.51 s |

The first DisTorch2 run assigned 3094.15 MiB of layers to cuda:0, 8168.53 MiB
to cuda:1, and 4614.47 MiB to CPU, plus 3.38 MiB of small layers on cuda:0.
The loader reported completing this assignment, but sampling failed. These
logs do not independently measure live donor storage or validate the complete
configuration. The larger GPU cache and activation working set still exhausted
VRAM on this setup.

Successful outputs have the expected frame counts and dimensions. Selected
frames were checked for nonblank content, and the 49-frame CPU/GPU contact
sheet was visually inspected. Mean absolute RGB differences at frames
0/16/32/48 were 3.42/2.64/1.83/2.65 on the 0–255 scale. This is a diagnostic
comparison from one seed, not a general quality or performance benchmark.

ComfyUI remained responsive after each OOM. A fresh-seed 192×320, 17-frame,
two-step GPU/int8 run then passed in 9.17 s without restarting the service.
No silent process death or CUDA illegal-address failure was observed. The
original report used Windows, Blackwell GPUs, kitchen 0.2.31, and DynamicVRAM
disabled, so this does not establish that its original reproduction is fixed.
No additional runtime patch was made from these results.

For this hardware, use GPU/int8 cache at the tested 49-frame size and
CPU/default cache at 81 frames. The three passing API prompts and setup
instructions are in [the WanAnimate examples](example_workflows/wan-animate-2/README.md).
They are manual GPU regression cases; CPU-only GitHub CI cannot validate them.

Local prompts, histories, server logs, comparison previews, and the validation
summary are saved under
`/srv/appdata/comfyui/user/default/workflow-backups/wananimate-issue219/`.
Animated outputs are under
`/home/geist/Pictures/StableDiffusion/wananimate-issue219/`.

## Dynamic loading validation (2026-10-10)

Upstream PR #183 installed the nonrecursive `_load_list` guard only inside the
standard MultiGPU CLIP wrappers. DisTorch2 with an empty allocation string keeps
DynamicVRAM enabled and could reach the unguarded loader when those wrappers
had never run. The fork now installs the guard once during custom-node startup;
per-node installation is removed. Older ComfyUI builds without
`ModelPatcherDynamic` skip registration, and ordinary small graphs and
non-dynamic calls still delegate to core.

The fallback now checks descendant parameters even when a parent has no direct
parameters, matching ComfyUI 0.39.0. This prevents a cast-enabled parent from
being mistaken for a leaf and passing its entire subtree into recursive
`state_dict()` sizing. The parameter iterator also visits each shared module
once, preserving Torch's first-path and tied-parameter behavior and terminating
on a module back reference. This does not promise support for cyclic models
through all of ComfyUI's other operations.

Seven added CPU tests cover a chain deeper than Python's recursion limit,
parent classification and dtype movement, shared/tied parameters, walker back
references, idempotence, native delegation, and older-core compatibility.
All 29 tests pass in both the Python 3.12 CPU environment and the production
container. Ruff and basedpyright pass without warnings. A representative model's
fallback load list matches the installed core's dynamic load list.

After restarting the idle service, all registered node IDs and input schemas
remain unchanged. A fresh Wan Animate INT8 ConvRot test used
`CLIPLoaderDisTorch2MultiGPU` with no donor allocation, UMT5 and VAE on the 3060,
and generation on the 4080: 192×320, 17 frames, 2 steps, GPU/int8 cache. It
completed successfully, and the saved animation contains 17 nonblank frames.
Startup and generation logs confirm the guard was installed before any loader
ran and was used for the empty-allocation text encoder. Models were unloaded
afterward and the service remained healthy with an empty queue.

These results validate the registration gap and fallback behavior on Linux.
They do not reproduce or establish resolution of #184's Windows native C-stack
failure, nor validate its ZImage checkpoint or every ComfyUI version. Keep the
TODO open until the original Windows case is tested.
