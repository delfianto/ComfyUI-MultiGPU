# Maintenance

[Main README](../README.md) · [Upstream usage guide](README.md)

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

**I use Linux. I do not have a Windows machine and have no interest in maintaining
one. A `fixed` issue means the fix is validated, or the failure does not reproduce
in the documented tests, on this machine running my
[ComfyUI Docker image and Compose setup](https://github.com/delfianto/compose/tree/main/ai/comfyui)
(`ghcr.io/delfianto/comfyui-nvidia-cuda`).**

Diffusion normally runs on the RTX 4080; CLIP and VAE run on the RTX 3060.
DisTorch2 can also borrow model storage from the 3060. These cards have no usable
CUDA peer-access path in this setup, so transfer overhead matters.

## Tested model configurations

Completed generation tests:

| Configuration | Models and components | Test and result |
| --- | --- | --- |
| Krea2 RAW FP8 + LoRA + DyPE | `krea2-raw-fp8.safetensors`; `qwen3-vl-4b-fp8-scaled.safetensors`; `qwen-image-vae-bf16.safetensors` (renamed after the test); Karina LoRA `aespakarina_krea2v3_27_qfloat8.safetensors` at 0.9 | 512×512 smoke test and 896×1200, 25-step generation passed; full run took 68.14 seconds after switching from FLUX. |
| FLUX.1-dev FP8 + DisTorch2 + DyPE | `flux1-dev-fp8.safetensors`; `clip-l.safetensors` + `t5xxl-fp8-e4m3fn-scaled.safetensors`; `ae.safetensors` | 512×512, 2-step generation passed; measured donor placement was 4080.60 MiB on the 3060, with 7674.75 MiB on the 4080. |
| Wan Animate 2 distilled INT8 ConvRot | `wan-animate-2-distill-int8-convrot.safetensors`; `umt5-xxl-fp8-e4m3fn-scaled.safetensors`; `clip-vision-h.safetensors`; `wan-v2.1-vae-bf16.safetensors` | 480×832, 10 steps: 49 frames passed with CPU/default and GPU/int8 cache; 81 frames passed with CPU/default cache. See the cache validation below for OOM cases and limits. |
| MiniMax H3 mixed INT4/INT8 + INT4 encoder | FL2VA and REF2VA pruned ConvRot checkpoints; Qwen3-VL-32B INT4 ConvRot; MiniMax video FP16 and audio FP32 VAEs | Both variants passed 128×128, 5-frame, 2-step AV sampling and decoding. Donor storage remains about 5.96 GiB throughout sampling; CLIP and both VAEs execute on the 3060. See [validation](#minimax-mixed-quantization-validation-2026-10-10). |

| LTX-2.5 distilled INT8 ConvRot | Gemma4-12B INT8 ConvRot; video/audio BF16 VAEs; spatial 2× latent upscaler | 9-frame smoke test and two 33-frame, 256×256 → 512×512 AV generations passed. Diffusion uses both GPUs; encoder uses the 3060 and CPU. See [validation](#ltx-25-validation-2026-10-10). |

A 2-step smoke test checks execution, not final image quality.

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

The dev group supplies Ruff and basedpyright; install Torch separately or use
ComfyUI's environment. Ruff checks imports, common errors, bugbear, and Python
modernization. basedpyright uses basic mode with unused-variable,
unused-parameter, and deprecated-API warnings enabled. Host and optional-provider
imports may be absent; [typings/nodes.pyi](../typings/nodes.pyi) supplies the core
registry contract. Dynamic logger methods use an explicit `Any` boundary.

As of 2026-10-10, all **53 regression tests** pass on the host and in the
production container. Ruff, formatting, basedpyright, and GitHub CI pass.

## Upstream triage

[Archive thread #223](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/223)
identifies proposed PRs, not validated fixes. Reviewed on 2026-10-10.
Severity is this fork's assessment; upstream does not assign severity labels.

| Priority | Reports | Problem | Fix and validation |
| --- | --- | --- | --- |
| Critical | [#187](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/187), [#210](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/210), [#218](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/218), [#220](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/220); [PR #216](https://github.com/pollockjj/ComfyUI-MultiGPU/pull/216) | CUDA peer-access probing can fail or crash. | Prefer Torch's peer API; resolve CUDA runtime libraries; probe failures return false. CPU regressions cover success, failure, caching, and platform candidates. |
| Critical | [#211](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/211), [#213](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/213), [#222](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/222) | Cross-device DLPack exports can conflict with kernel device/stream handling. | Keep comfy-kitchen >=0.2.35 native exports. Older versions retain the compatibility patch and are unvalidated here. |
| Critical | Local FP8 CLIP crash; [#189](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/189) related symptoms | CLIP/VAE inference can execute under the wrong current device. | Guard direct/scheduled CLIP and standard/tiled VAE inference with their real load device; restore state on exceptions. CPU regressions and local dual-GPU generation pass. |
| Critical | [#184](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/184); [PR #183](https://github.com/pollockjj/ComfyUI-MultiGPU/pull/183) | DynamicVRAM stack overflow, including graphs without MultiGPU nodes. | Install the traversal guard at startup for DisTorch2 and core loaders; correct parent classification and shared-module traversal. Deep-graph CPU tests and local empty-allocation DisTorch2 generation pass. See [validation](#dynamic-loading-validation-2026-10-10). |
| Critical | [#219](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/219) | With kitchen 0.2.31, WanAnimate GPU/int8 cache reportedly kills the process without a Python traceback. | Native DLPack is preserved on kitchen 0.2.37. Local 49-frame GPU/int8 and 81-frame CPU/default generation pass; 81-frame GPU/int8 returns recoverable OOM. See [validation](#wananimate-cache-validation-2026-10-10). |
| High | Local code finding: older GGUF wrappers | Provider calls and result processing could fail before device restoration. | Protect setup, loading, and metadata handling in all four older wrappers. Regressions cover setup/provider/output errors and preserve the previous device. |
| High | [#203](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/203) | Async/await errors prevent generation; wrappers assumed synchronous providers. | Preserve synchronous/async execution contracts across all loader wrapper families; await before processing results. Restore device and patcher state between async steps, on errors, and on cancellation. Local executor probes pass; the report supplies no exact graph/provider version to replay. See [validation](#loader-execution-validation-2026-10-10). |
| High | [#185](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/185) | Switching encoders on GPU 1 reportedly produces garbled output after two or three runs, particularly Flux.2 Klein FP8. | Preserve core loading for ordinary/empty-allocation requests after DisTorch installs its patches; restore CLIP loader device selection on failures. Repeated encoder and two-stage generation tests pass with mixed-FP8 Qwen3-8B: conditioning and output pixels match exactly across runs and loader modes. See [validation](#repeated-clip-validation-2026-10-10). |
| High | [#209](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/209) | MiniMax H3 mixed INT8/INT4 donor allocation reportedly leaves GPU 1 nearly empty. Combined local runs also exposed donor-memory OOM and a native allocation-recording crash. | Legacy patcher allocation places real packed weights and metadata on the donor. Reserve storage on every assigned GPU; evict unused donor residents; pause recording around cross-GPU quantized metadata allocation. Both installed diffusion variants and the INT4 encoder pass placement/reload tests; consecutive AV sampling and decoding pass. See [validation](#minimax-mixed-quantization-validation-2026-10-10). |
| High | [#202](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/202)/[#186](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/186); [PR #195](https://github.com/pollockjj/ComfyUI-MultiGPU/pull/195) | aimdo initialization can omit visible GPUs or disturb existing contexts/headroom. | Initialize visible devices together before core startup; preserve primary headroom; do not reinitialize live contexts or mark missing devices initialized. Defer Torch import until core init_device; retain legacy fallback. CPU regressions pass. |
| High | [#191](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/191) | DynamicVRAM bypasses DisTorch2 donor placement through the patched legacy loading path. | Use legacy ModelPatcher when loading explicitly distributed models; ordinary loaders keep DynamicVRAM. Restore patcher alias and loader device state on errors. Local FLUX donor placement and regressions pass. |
| High | [#196](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/196); [PR #198](https://github.com/pollockjj/ComfyUI-MultiGPU/pull/198) | Parameter estimates evaluate parametrized weights or miscount tied/packed storage. | Count unique underlying parameters by physical nbytes. Tests cover weight_norm, tied parameters, and packed storage; buffers and allocator overhead remain outside this estimate. |
| Medium | [#188](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/188); [PR #200](https://github.com/pollockjj/ComfyUI-MultiGPU/pull/200) | Florence2 wrapper schema/arguments drift from the provider; current loaders also consult the text-encoder device. | Delegate provider schema and arguments; set and restore both device selectors. Both actual Florence2-base loaders generate captions on cuda:1. See [validation](#optional-provider-validation-2026-10-10). |
| Medium | [PR #217](https://github.com/pollockjj/ComfyUI-MultiGPU/pull/217) | Alternative ComfyUI-GGUF-Loader provider is not recognized. | Recognize the additional provider while retaining city96 support. Its actual Qwen3-VL Q4 loader and text encoding pass on cuda:1. See [validation](#optional-provider-validation-2026-10-10). |
| Medium | [PR #182](https://github.com/pollockjj/ComfyUI-MultiGPU/pull/182) | Recursive schema copying can fail on metadata that cannot be deep-copied. | Copy the schema and modified optional map shallowly. Regression verifies upstream schema isolation with noncopyable metadata. |

This table contains fixes made and tested on this machine with the container
image documented above. Validation details and limits are recorded in each row
and the sections below.

WanVideo/LTX feature requests are outside this crash-fix tracker;
[LTXV2 PR #164](https://github.com/pollockjj/ComfyUI-MultiGPU/pull/164) is not merged.

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

## Deployment and initial validation (2026-10-10)

The service bind-mounts this checkout at `/data/custom_nodes/ComfyUI-MultiGPU`.
Apply Python edits with `composectl restart ai-comfyui --json`; rebuilding the
image is unnecessary. GGUF is bundled in the image; MultiGPU loads from the
persistent checkout. Keep serialized node names compatible with existing flows.
The registry PublisherId is retained as historical metadata.

Initial deployment used image `78549b8cbea2`. Both GPUs had initialized aimdo
contexts and modern DLPack exports stayed native. After moving runtime code into
`src/`, all 44 MultiGPU node names, schemas, outputs, and categories matched the
previous registration; in-app documentation returned HTTP 200. A fresh Krea2 +
LoRA + DyPE smoke generation completed in 8.65 seconds.

The FLUX workflow's checkpoint loader was replaced with separate diffusion,
CLIP-L/T5, and VAE loaders because the model file contains diffusion weights only.
LTX helper loaders now forward `config` instead of discarding it. Full Krea2 and
FLUX generation results are in the model table above. Migration prompts,
histories, and outputs are under
`/srv/appdata/comfyui/user/default/workflow-backups/multigpu-fork-migration/`.

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
No process death or CUDA illegal-address failure was observed, and no additional
runtime patch was needed.

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
A representative model's fallback load list matches the installed core's
dynamic load list.

After restarting the idle service, all registered node IDs and input schemas
remain unchanged. A fresh Wan Animate INT8 ConvRot test used
`CLIPLoaderDisTorch2MultiGPU` with no donor allocation, UMT5 and VAE on the 3060,
and generation on the 4080: 192×320, 17 frames, 2 steps, GPU/int8 cache. It
completed successfully, and the saved animation contains 17 nonblank frames.
Startup and generation logs confirm the guard was installed before any loader
ran and was used for the empty-allocation text encoder. Models were unloaded
afterward and the service remained healthy with an empty queue.

## Repeated CLIP validation (2026-10-10)

[#185](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/185) was previously
partial because the existing runtime guards had only been tested in individual
generations. The report describes second/third-run corruption with two encoders
on the secondary GPU, particularly Klein 9B's mixed-FP8 Qwen3 encoder. Its graph
is not attached, so validation here exercises the relevant encoders and loading
paths rather than claiming to replay the reporter's exact workflow.

Two code findings were addressed during the investigation:

- Installing DisTorch previously replaced core `load_models_gpu` for every
  subsequent request with an older copied implementation. Ordinary models and
  empty-allocation DisTorch2 CLIP models now delegate to the installed core
  loader, preserving its DynamicVRAM eviction, pin-budget handling, loading
  order, and argument contract. Explicitly annotated distributed models,
  including attached models, retain the DisTorch path; older provider contracts
  retain their existing fallback.
- Standard CLIP wrappers previously called the provider before entering their
  device-restoration `try/finally`. Both standard wrapper variants now restore
  the previous text-encoder selection if loading fails.

The reported corruption was not reproduced on the current stack, including
before these two additional changes. Existing runtime guards already kept the
logical device and actual CUDA device aligned during the tested encoding paths.
The new changes fix the loading-contract and failure-restoration defects;
these findings do not establish the cause of the original upstream corruption.

Encoder-only tests alternated Chroma's T5 FP8 and
`qwen3-8b-fp8-mixed.safetensors` (`flux2`) on `cuda:1`, using the same prompt and
comparing every conditioning tensor and tensor-valued metadata against saved
CPU references. The tests used the same CLIP objects through each cycle.

| Loader mode | Repeated warm cycles | Forced unload/reload cycles | Result |
| --- | --- | --- | --- |
| Standard MultiGPU | 4 | 3 | All tensors finite and exactly equal to the first-cycle reference. |
| DisTorch2, empty allocation | 4 | Not separately tested | All tensors finite and exactly equal to the standard-loader reference. |
| DisTorch2, 4 GiB virtual VRAM with CPU donor | 4 | 3 | All tensors finite and exactly equal to the standard-loader reference after eviction/reload. |

The full generation test used two sequential sampling stages: FLUX.1-dev FP8
with CLIP-L/T5 FP8, then FLUX.2 Klein 9B FP8 with the mixed-FP8 Qwen3-8B encoder.
Both CLIPs and both VAEs used the 3060; native diffusion loaders targeted the
4080. Each run generated 512×512 outputs at seed 185, with 2 steps for FLUX.1
and 4 steps for Klein. The second-stage encoding depended on first-stage VAE
decoding, ensuring sampling and VAE work happened between the two encoders.
The two samplers used independent latents; this tests sequential model/encoder
switching rather than image-to-image refinement.

Temporary server-side probes forced positive and negative re-encoding on every
queued run, avoiding false passes from ComfyUI's conditioning cache. Four
consecutive runs were completed for each configuration:

| First encoder loader | Second encoder loader | Runs | Conditioning and images |
| --- | --- | --- | --- |
| Standard MultiGPU | Standard MultiGPU | 4 | Exact tensor and image-pixel matches across all four runs. |
| Standard MultiGPU | DisTorch2, empty allocation | 4 | Exact matches against the standard-loader reference. |
| Standard MultiGPU | DisTorch2, 4 GiB virtual VRAM with CPU donor | 4 | Exact matches against the standard-loader reference. |

Across 12 runs, all 48 positive/negative encodes were finite and had maximum
absolute difference **0** from their references. All 24 output images matched
their respective stage's reference pixels, including across loader modes.
Klein produced a coherent red teapot image, verified visually. Neither stage
showed second/third-run corruption. CUDA current-device state returned to
GPU 0 after every encoding call.

Seven added CPU regressions cover core/delegated loading, empty and explicit
allocations, annotated attached models, legacy provider fallback, shared patch
references, exception propagation, standard CLIP loader failure restoration,
and per-instance encoder device selection. A separate CUDA-runtime
candidate test covers missing `torch.__file__`. Zed type warnings were corrected
by annotating `_coerce_torch_device` as
`torch.device | None` and checking `torch.__file__` before `dirname`.

The two-GGUF variant was not tested: Chroma T5 GGUF and Qwen3-8B Q8_0 files
are not installed.

The temporary probe nodes were removed after testing; diagnostic prompts,
reference tensors, metrics, histories, and images remain in local test backups.
After probe removal, a fresh 512×512, 2-step FLUX.1 DisTorch2 generation with
4 GiB virtual VRAM and the 3060 as donor also passed. This checks that preserving
core loading for ordinary requests does not prevent the distributed generation
path from executing; it is not an independent measurement of donor storage.

## Loader execution validation (2026-10-10)

`src/loader_execution.py` adapts all 11 wrapper variants to the provider's
execution contract. Synchronous providers return directly; async providers
remain coroutine functions that ComfyUI can await. The existing loader scopes
resume result processing only after the provider finishes. Between async steps,
the adapter restores the surrounding logical devices, actual CUDA device, and
`CoreModelPatcher` alias, so another node does not inherit suspended loader state.

Regressions exercise successful results, provider/setup/output failures,
cancellation, independent concurrent device selections, empty allocations, and
DisTorch metadata on model and CLIP-style patcher results. Temporary nodes also
passed through the installed ComfyUI executor for async standard, CLIP, legacy
GGUF, and DisTorch2 wrappers. These controlled providers validate the execution
contract; they are not the unavailable original #203 reproducer.

A production-node FLUX.1-dev FP8 + DyPE generation completed in 9 seconds at
512×512 with two sampling steps: compute on the 4080, 4 GiB donor storage on the
3060, and CLIP/VAE on the 3060. Temporary probe nodes were removed afterward;
production node IDs and input schemas were checked against the pre-change API.
Prompts and histories are saved under
`/srv/appdata/comfyui/user/default/workflow-backups/issue185/loader-*`.

## MiniMax mixed-quantization validation (2026-10-10)

Tested `minimax-h3-fl2va-pruned-mixed-int4-int8-convrot.safetensors`,
`minimax-h3-ref2va-pruned-mixed-int4-int8-convrot.safetensors`, and
`qwen3-vl-32b-minimax-h3-int4-convrot.safetensors`. Measurements count unique
backing storage, including packed weights, quantization metadata, and buffers.
Each standalone model completed two load/unload cycles with a 6 GiB donation
from `cuda:1` to compute device `cuda:0`:

| Model | 4080 storage (MiB) | 3060 donor storage (MiB) |
| --- | --- | --- |
| FL2VA mixed INT4/INT8 | 9062.38 | 6101.13 |
| REF2VA mixed INT4/INT8 | 8290.75 | 6101.01 |
| Qwen3-VL-32B INT4 | 8121.57 | 6137.23 |

Placement survived reloads; unloading returned model storage to CPU. The encoder
also passed two encodes. Whole-layer allocation explains the small difference
from the requested 6 GiB.

Combined sampling initially hit OOM because eviction considered only the
compute GPU. After reserving donor memory, sampling exposed a process abort
while cloning donor quantization metadata inside the compute GPU's allocation
recorder. Fixes in `src/distributed_memory.py`:

- Reserve the pending storage of actual assigned blocks on every device and
  protect the models in the current load request. Preserve the explicit
  `eject_models` toggle in modern loader-wrapper metadata.
- After core eviction, detach unused distributed models holding storage on the
  requested donor GPU, even when their compute device differs.
- Pause native allocation recording around cross-GPU quantized `empty_like`
  handling. This kitchen handler guard survives core's per-prompt function
  restoration; same-device and CPU transfers retain their normal handling.

Consecutive requests passed with `eject_models=False`, allocation recording
still enabled, diffusion on the 4080 with 6 GiB requested from the 3060, and the
encoder on the 3060 with 6 GiB offloaded to CPU. Both diffusion variants produced
finite AV latents and decoded five 128×128 frames plus 32 kHz audio using the
3060's VAEs. REF2VA decoding completed in 10.57 seconds; FL2VA in 26.43 seconds.
Sampling measurements stayed at the donor storage shown above through both
steps. These are execution tests with no reference images, not quality tests.

The guard is validated with comfy-kitchen 0.2.37's generic dispatch table.
Prompts, storage measurements, histories, and crash traces are saved under
`/srv/appdata/comfyui/user/default/workflow-backups/issue209/`; generated frames
and audio are in the output directory's `issue209/` folder. Temporary probe
nodes were removed after validation.

## LTX-2.5 validation (2026-10-10)

Tested the split ComfyUI models:

- `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`
- `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`
- `ltx-2.5-video-vae-conv-bf16.safetensors`
- `ltx-2.5-audio-vae-bf16.safetensors`
- `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`

All five files match the official Lightricks revision's SHA256 hashes. The graph
uses native LTX nodes with this fork's generic loaders; PR #164 adapters are
unnecessary for this split-model configuration. Optional prompt enhancement
was skipped.

Diffusion uses `UNETLoaderDisTorch2MultiGPU`: compute on cuda:0, 9 GiB virtual
VRAM donated from cuda:1. The Gemma encoder uses `CLIPLoaderDisTorch2MultiGPU`
with type `ltxv`, compute on cuda:1 and 6 GiB donated from CPU. Both VAEs use
`VAELoaderMultiGPU` on cuda:1. All loaders retain `eject_models=False` where
available.

| Test | Sampling | Result | Execution time |
| --- | --- | --- | --- |
| Smoke | 128×128, 9 frames, 8 steps | Video and audio passed | 44.47 s |
| Two-stage | 256×256, 33 frames, 8 steps; spatial 2× upscale and 3-step refinement | 512×512 video and audio passed | 65.39 s |
| Fresh-process repeat | Same two-stage graph after restarting ComfyUI | Decoded frames and audio finite | 69.28 s wall time |

Sampling uses the official distilled template's sigma schedules,
`euler_ancestral`, video/audio CFG 1 and 24 fps. Measured transformer storage
remained 11,335.21 MiB on the 4080 and 9,151.95 MiB on the 3060 across all
11 sampling calls in both two-stage runs. Encoder storage during encoding was
about 8.28 GiB on the 3060 and 6.01 GiB on CPU. Donor-aware eviction allowed
encoding, sampling, upscaling and decoding to share the secondary GPU.

The saved MP4 contains 33 frames at 512×512 with an audio stream. Visual
inspection shows a coherent blue teapot scene; repeat checks found finite,
nonblank decoded images and finite, nonzero 48 kHz stereo audio. These short,
low-resolution runs validate execution rather than production quality. No
additional runtime patch was needed.

Follow-up motion-artifact tests found ghosting around moving edges with
`VAEDecodeTiled` at temporal size 16, overlap 8. Decoding the **same final latent**
with `VAEDecode` removed the doubled edges in inspected frames at both 512×512
and 768×768, without OOM or automatic tiled fallback on the 3060. The higher
resolution run used 384×384 base latents and completed in 203.60 seconds.
Spatial tiling at size 256, overlap 64, with temporal size **64** also removed
the ghosting in the inspected 33-frame clip; only decoding reran (5.73 seconds).
This isolates the short temporal window as the cause in this test, rather than
MP4 encoding or GPU placement. Longer clips need separate validation.
The saved production prompts now use untiled decoding; a 768×768 variant and
both comparison graphs are retained with the local test artifacts.

Temporary probes were removed afterward. Prompts, histories, storage and
finite-value measurements, and a probe-free `two-stage-production-prompt.json`
are saved under
`/srv/appdata/comfyui/user/default/workflow-backups/ltx25-validation/`.
Videos, frames and audio are under
`/home/geist/Pictures/StableDiffusion/ltx25-validation/`.

## Workflow-switching stress test (2026-10-10)

Completed three rounds across 11 model families, with an initial LTX reference
and an immediate LTX repeat at the end: **35 successful generations**. Private
API variants use the saved workflows' loaders, LoRAs and DyPE branches at moderate
sizes. Temporary nodes force fresh text encoding and check tensors before saving;
the running process is retained between workflows.

| Workflows | Stress settings |
| --- | --- |
| Krea2 RAW + Karina LoRA + DyPE | 512×768, 20 steps |
| FLUX.1-dev + DisTorch2 + DyPE | 512×512, 12 steps, 4 GiB requested from the 3060 |
| SDXL IntoRealism / ReaPony | 640×768, 20 steps each, checkpoint CLIP/VAE on the 3060 |
| Z-Image Moody / Qwen Image 2.1 Noct Q + DyPE | 512×768, 12 steps each |
| FLUX.2 Klein 9B | 512×512, 4 steps, mixed-FP8 Qwen3 encoder on the 3060 |
| Ideogram 4 precise core | 512×512, 12 steps, main and unconditional diffusion models |
| Wan Animate 2 | 480×832, 49 frames, 10 steps, GPU/int8 cache |
| MiniMax H3 FL2VA / REF2VA | 128×128, 5 frames, 2 steps; REF2VA in round two |
| LTX-2.5 | 33 frames, 256×256 → 512×512, 8+3 steps, untiled decode |

The original settings did **not** pass the stress test:

- With the default RAM-pressure node cache, the second consecutive LTX request
  caused sustained host memory pressure. `systemd-oomd` killed ComfyUI; both GPUs
  remained visible and no CUDA fault was recorded.
- Disabling pinned memory alone still crossed the test watchdog's 6 GiB RAM
  floor on the next LTX request. Classic caching crossed that floor while loading
  Ideogram after seven image-model runs. These watchdog stops are incomplete
  tests, not native OOM or CUDA failures.
- `--cache-none` completed a full switching round. In round two, changing Wan to
  CPU/default cache crossed the RAM floor; the watchdog stopped the service.
  The remaining cases resumed after restarting with GPU/int8 cache. Thus the
  three rounds were **not one uninterrupted run**.

The completed profile uses `--cache-none`, normal pinned-memory settings and
Wan GPU/int8 cache. It produced **145 passing finite-value checks**: 67
conditioning, 35 latent, 35 image and 8 audio checks. Successful runs retained
at least 10.45 GiB available host RAM in the two-second samples. No generation
error, CUDA fault or watchdog stop occurred in the remaining GPU/int8 cases.
The final switched LTX video and its immediate repeat match the fresh reference
exactly across all 33 PNG frames. Decoded audio is finite; 16-bit PCM differences
are at most two units (normalized difference 0.0000611), rather than bit-identical.
Representative image outputs were inspected visually; MiniMax's two-step case
checks execution only.

The Compose service now uses `--cache-none` to limit retained node results.
This costs loader reuse and re-executes unchanged nodes. It is a tested workaround,
not a root-cause fix for RAM retention, and does not validate cached-model reuse
under the failing policies. Follow-up work is to identify the retained allocations
and test bounded caching without recreating host-wide memory stalls.

Temporary probes were removed and the original production node registry restored.
Prompts, histories, finite checks, sampled GPU/host memory, watchdog stops,
`systemd-oomd` logs and comparisons are under
`/srv/appdata/comfyui/user/default/workflow-backups/model-switch-stress/`.
Outputs are under `/home/geist/Pictures/StableDiffusion/model-switch-stress/`.

## Optional provider validation (2026-10-10)

Tested real provider checkouts in a separate process inside the production
container, using isolated dependencies. These checkouts remain outside
`custom_nodes`; the running service retains its city96 GGUF provider.

- [kijai/ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2) at
  `9ece3de`: both local and download/load MultiGPU nodes loaded
  `microsoft/Florence-2-base` in fp16 and generated a caption from a 128×128
  image. Only the MultiGPU schema was invoked before loading, exercising the
  provider's `model_paths` initialization. Both patchers and all model
  parameters used cuda:1. Before the device-selector fix, both patchers used
  cuda:0 despite selecting cuda:1. Florence2 wrappers now set the text-encoder
  selector alongside the UNet selector, preserving older provider behavior;
  both selectors restore after success or failure. Regressions also cover
  reordered optional arguments and async execution.
- [ChrisColeTech/ComfyUI-GGUF-Loader](https://github.com/ChrisColeTech/ComfyUI-GGUF-Loader)
  at `499df2a`: provider discovery registered the GGUF wrappers, and
  `CLIPLoaderGGUFMultiGPU` delegated to that provider's class. Loading
  `qwen3-vl-8b-uncensored-aggressive-q4-k-m.gguf` with type `qwen_image`
  produced finite text embeddings of shape `[1, 512, 12288]`. The patcher
  and all model parameters used cuda:1. This validates registration and
  actual quantized text loading/encoding; it does not test the provider's
  additional generation nodes or the Qwen vision projector.

Private validation scripts, provider revisions and results are under
`user/default/workflow-backups/provider-validation/`. The Florence2 base
checkpoint remains in `models/LLM/Florence-2-base/` for reuse.
