"""Checkpoint placement, quantization and reload contracts without GPU allocation."""

import importlib.util
import logging
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch

SOURCE = Path(__file__).resolve().parents[1] / "src"


class CheckpointAPITests(unittest.TestCase):
    def setUp(self):
        self.state = SimpleNamespace(device="cuda:0", clip="cuda:0")
        package: Any = ModuleType("mgpu_checkpoint_test")
        package.__path__ = [str(SOURCE)]
        package.get_current_device = lambda: self.state.device
        package.get_current_text_encoder_device = lambda: self.state.clip
        package.set_current_device = lambda d: setattr(self.state, "device", d)
        package.set_current_text_encoder_device = lambda d: setattr(self.state, "clip", d)
        self.model = SimpleNamespace(load_model_weights=Mock())
        self.config = SimpleNamespace(
            quant_config={"format": "int8"},
            supported_inference_dtypes=[torch.float16],
            clip_vision_prefix=None,
            vae_key_prefix=["vae."],
            set_inference_dtype=Mock(),
            get_model=Mock(return_value=self.model),
            process_vae_state_dict=lambda sd: sd,
            clip_target=lambda **kw: "clip-target",
            process_clip_state_dict=lambda sd: sd,
        )
        self.patcher = Mock(side_effect=self.make_patcher)
        self.management = SimpleNamespace(
            unet_dtype=Mock(return_value=torch.float16),
            unet_manual_cast=Mock(return_value=torch.float16),
            unet_inital_load_device=Mock(return_value=torch.device("cpu")),
            unet_offload_device=lambda: torch.device("cpu"),
            load_models_gpu=Mock(),
        )
        self.vae = Mock(side_effect=lambda **kw: SimpleNamespace(patcher=self.make_patcher(None)))
        self.clip = Mock(
            side_effect=lambda *a, **kw: SimpleNamespace(patcher=self.make_patcher(None))
        )
        self.weights = {"weight": torch.ones(2)}
        self.utils = SimpleNamespace(
            load_torch_file=Mock(return_value=(self.weights, {"quantization": "test"})),
            calculate_parameters=lambda *a: 2,
            weight_dtype=lambda *a: torch.float16,
            convert_old_quants=lambda sd, *a, **kw: (sd, kw.get("metadata")),
            state_dict_prefix_replace=lambda sd, *a, **kw: sd,
        )
        self.sd = SimpleNamespace(
            CLIP=self.clip, VAE=self.vae, load_diffusion_model_state_dict=Mock()
        )
        comfy: Any = ModuleType("comfy")
        comfy.model_management = self.management
        comfy.model_patcher = SimpleNamespace(
            ModelPatcher=self.patcher, CoreModelPatcher=self.patcher
        )
        comfy.model_detection = SimpleNamespace(
            unet_prefix_from_state_dict=lambda sd: "model.",
            model_config_from_unet=Mock(return_value=self.config),
        )
        comfy.clip_vision = SimpleNamespace()
        comfy.sd = self.sd
        comfy.utils = self.utils
        comfy.storage = SimpleNamespace(state_dict_fast_disk=lambda sd: True)
        device: Any = ModuleType("mgpu_checkpoint_test.device_utils")
        device.get_device_list = lambda: ["cpu"]
        device.soft_empty_cache_multigpu = Mock()
        distorch: Any = ModuleType("mgpu_checkpoint_test.distorch_2")
        distorch.register_patched_safetensor_modelpatcher = Mock()
        memory: Any = ModuleType("mgpu_checkpoint_test.model_management_mgpu")
        memory.multigpu_memory_log = Mock()
        modules = {"comfy": comfy, "mgpu_checkpoint_test": package}
        for name in (
            "model_management",
            "model_patcher",
            "model_detection",
            "clip_vision",
            "sd",
            "utils",
            "storage",
        ):
            modules[f"comfy.{name}"] = getattr(comfy, name)
        modules.update(
            {
                device.__name__: device,
                distorch.__name__: distorch,
                memory.__name__: memory,
                "folder_paths": SimpleNamespace(get_folder_paths=lambda name: ["embeddings"]),
            }
        )
        self.imports = patch.dict(sys.modules, modules)
        self.imports.start()
        self.addCleanup(self.imports.stop)
        spec = importlib.util.spec_from_file_location(
            "mgpu_checkpoint_test.checkpoint_multigpu", SOURCE / "checkpoint_multigpu.py"
        )
        assert spec and spec.loader
        self.loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.loader)
        logging.getLogger("MultiGPU").mgpu_mm_log = Mock()  # type: ignore[attr-defined]
        self.placement = {"unet_device": "cuda:1", "clip_device": "cuda:1", "vae_device": "cuda:1"}

    @staticmethod
    def make_patcher(model, **kwargs):
        return SimpleNamespace(model=model, is_dynamic=lambda: True, **kwargs)

    def test_quantized_config_device_and_disk_backing_reach_core_apis(self):
        model, clip, vae = self.loader.load_checkpoint_with_placement("checkpoint", self.placement)
        self.management.unet_manual_cast.assert_called_once_with(
            None, torch.device("cuda:1"), [torch.float16]
        )
        self.config.set_inference_dtype.assert_called_once_with(
            torch.float16, torch.float16, device=torch.device("cuda:1")
        )
        self.assertEqual(model.load_device, torch.device("cuda:1"))
        self.assertTrue(model.fast_disk)
        self.vae.assert_called_once_with(
            sd=self.weights, metadata={"quantization": "test"}, device=torch.device("cuda:1")
        )
        self.assertIsNotNone(clip.patcher.cached_patcher_init)
        self.assertIsNotNone(vae.patcher.cached_patcher_init)
        self.utils.load_torch_file.assert_called_once_with("checkpoint", return_metadata=True)
        self.assertEqual((self.state.device, self.state.clip), ("cuda:0", "cuda:0"))

    def test_reload_factory_preserves_placement_and_loads_only_requested_component(self):
        model, _, _ = self.loader.load_checkpoint_with_placement("checkpoint", self.placement)
        self.placement["unet_device"] = "cuda:0"
        self.clip.reset_mock()
        self.vae.reset_mock()
        factory, args = model.cached_patcher_init
        reloaded = factory(*args, disable_dynamic=True)
        self.assertEqual(reloaded.load_device, torch.device("cuda:1"))
        self.clip.assert_not_called()
        self.vae.assert_not_called()
        self.assertEqual((self.state.device, self.state.clip), ("cuda:0", "cuda:0"))

    def test_failure_restores_device_state(self):
        self.vae.side_effect = ValueError("VAE failure")
        with self.assertRaisesRegex(ValueError, "VAE failure"):
            self.loader.load_checkpoint_with_placement("checkpoint", self.placement)
        self.assertEqual((self.state.device, self.state.clip), ("cuda:0", "cuda:0"))

    def test_explicit_offload_and_direct_gpu_load_are_preserved(self):
        self.management.unet_inital_load_device.return_value = torch.device("cuda:1")
        out = self.loader.patched_load_state_dict_guess_config(
            self.weights,
            output_clip=False,
            output_vae=False,
            model_options={"offload_device": torch.device("cuda:0")},
            device_config=self.placement,
        )
        self.assertEqual(out[0].offload_device, torch.device("cuda:0"))
        self.management.load_models_gpu.assert_called_once_with([out[0]], force_full_load=True)

    def test_checkpoint_ejection_toggle_reaches_distributed_metadata(self):
        for eject in (False, True):
            model, _, _ = self.loader.load_checkpoint_with_placement(
                "checkpoint",
                self.placement,
                {"unet_allocation": "#cuda:1;4;cpu", "eject_models": eject},
            )
            self.assertEqual(model.model._distorch_v2_meta["eject_models"], eject)

    def test_empty_clip_allocation_preserves_dynamic_encoder_loading(self):
        self.loader.load_checkpoint_with_placement(
            "checkpoint",
            self.placement,
            {"unet_allocation": "#cuda:1;4;cpu", "clip_allocation": ""},
        )
        self.assertFalse(self.clip.call_args.kwargs["disable_dynamic"])

    def test_checkpoint_without_vae_preserves_cores_invalid_vae_placeholder(self):
        self.vae.side_effect = lambda **kwargs: SimpleNamespace(first_stage_model=None)
        model, clip, vae = self.loader.load_checkpoint_with_placement("checkpoint", self.placement)
        self.assertIsNotNone(model.cached_patcher_init)
        self.assertIsNotNone(clip.patcher.cached_patcher_init)
        self.assertIsNone(vae.first_stage_model)
