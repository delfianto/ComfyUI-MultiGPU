"""CPU regressions; import individual modules without starting ComfyUI."""

import importlib.util
import os
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"


def load_module(name):
    if "mgpu_regression" not in sys.modules:
        package: Any = ModuleType("mgpu_regression")
        package.__path__ = [str(SOURCE)]
        sys.modules["mgpu_regression"] = package
    spec = importlib.util.spec_from_file_location(f"mgpu_regression.{name}", SOURCE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EntryPointTests(unittest.TestCase):
    def test_root_exports_nodes_frontend_and_historical_helpers(self):
        spec = importlib.util.spec_from_file_location(
            "mgpu_entry_regression", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        implementation: Any = ModuleType("mgpu_entry_regression.src")
        implementation.NODE_CLASS_MAPPINGS = {"ExampleNode": object()}
        implementation.get_current_device = lambda: "cuda:1"
        with patch.dict(
            sys.modules,
            {"mgpu_entry_regression": module, "mgpu_entry_regression.src": implementation},
        ):
            spec.loader.exec_module(module)
        self.assertIs(module.NODE_CLASS_MAPPINGS, implementation.NODE_CLASS_MAPPINGS)
        self.assertEqual(module.get_current_device(), "cuda:1")
        self.assertEqual(module.WEB_DIRECTORY, "./web")
        self.assertTrue((ROOT / module.WEB_DIRECTORY).is_dir())

    def test_prestartup_executes_only_independent_initializer(self):
        spec = importlib.util.spec_from_file_location(
            "mgpu_prestart_regression", ROOT / "prestartup_script.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with patch("runpy.run_path") as execute:
            spec.loader.exec_module(module)
        execute.assert_called_once_with(str(SOURCE / "prestartup_script.py"))


class PeerTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("p2p_registry")
        self.registry = self.module.MultiGPUP2PRegistry()

    def test_same_device_needs_no_probe(self):
        with patch.object(self.module, "_get_libcudart", side_effect=AssertionError):
            self.assertTrue(self.registry.can_access_peer(1, 1))

    def test_torch_api_and_cache(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "can_device_access_peer", return_value=False) as probe,
        ):
            self.assertFalse(self.registry.can_access_peer(0, 1))
            self.assertFalse(self.registry.can_access_peer(0, 1))
            self.assertEqual(probe.call_count, 1)
            self.registry.clear_cache()
            self.registry.can_access_peer(0, 1)
            self.assertEqual(probe.call_count, 2)

    def test_failed_runtime_probe_is_nonfatal(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "can_device_access_peer", side_effect=RuntimeError("probe")),
            patch.object(self.module, "_get_libcudart", side_effect=OSError("missing library")),
        ):
            self.assertFalse(self.registry.can_access_peer(0, 1))

    def test_ctypes_success_and_cuda_error(self):
        def probe(result, _src, _dst):
            result._obj.value = 1
            return 0

        library = SimpleNamespace(cudaDeviceCanAccessPeer=Mock(side_effect=probe))
        with (
            patch.object(torch.cuda, "can_device_access_peer", None),
            patch.object(self.module, "_get_libcudart", return_value=library),
        ):
            self.assertTrue(self.registry._raw_can_access_peer(0, 1))
            library.cudaDeviceCanAccessPeer.side_effect = None
            library.cudaDeviceCanAccessPeer.return_value = 7
            self.assertFalse(self.registry._raw_can_access_peer(0, 1))

    def test_versioned_linux_runtime(self):
        library = SimpleNamespace()
        with (
            patch.object(self.module.ctypes.util, "find_library", return_value="libcudart.so.13"),
            patch.object(self.module.ctypes, "CDLL", return_value=library) as loader,
        ):
            self.assertIs(self.module._get_libcudart(), library)
            loader.assert_called_once_with("libcudart.so.13")

    def test_windows_runtime_candidates(self):
        host_platform = os.name
        windows = SimpleNamespace(name="nt", path=os.path)
        with (
            patch.object(self.module, "os", windows),
            patch.object(self.module.ctypes.util, "find_library", return_value=None),
            patch.object(self.module.glob, "glob", return_value=["torch/lib/cudart64_13.dll"]),
        ):
            self.assertEqual(self.module._cudart_candidates()[0], "torch/lib/cudart64_13.dll")
            self.assertEqual(os.name, host_platform)


class MemoryTests(unittest.TestCase):
    def test_weight_norm_storage(self):
        module = load_module("memory_utils")
        model = torch.nn.utils.parametrizations.weight_norm(torch.nn.Linear(3, 2))
        expected = (6 + 2 + 2) * 4  # direction, magnitude, bias
        self.assertEqual(module.model_parameter_bytes(model), expected)

    def test_tied_parameters_are_counted_once(self):
        module = load_module("memory_utils")
        model = torch.nn.Module()
        model.left = torch.nn.Linear(3, 2, bias=False)
        model.right = torch.nn.Linear(3, 2, bias=False)
        model.right.weight = model.left.weight
        self.assertEqual(module.model_parameter_bytes(model), 6 * 4)

    def test_quantized_storage_uses_physical_bytes(self):
        module = load_module("memory_utils")
        model = Mock(spec=torch.nn.Module)
        model.parameters.return_value = [
            SimpleNamespace(nbytes=8, numel=lambda: 8, element_size=lambda: 2)
        ]
        self.assertEqual(module.model_parameter_bytes(model), 8)


class ClipTests(unittest.TestCase):
    def setUp(self):
        self.device = "cuda:0"
        test = self

        class Clip:
            def __init__(self):
                self.patcher = SimpleNamespace(load_device="cuda:1")

            def encode_from_tokens(self, tokens):
                if tokens == "error":
                    raise ValueError("encoding failed")
                return test.device

            def encode_from_tokens_scheduled(self, tokens):
                return self.encode_from_tokens(tokens)

        class VAE:
            device = "cuda:1"

            def encode(self, value):
                return self.decode(value)

            def decode(self, value):
                if value == "error":
                    raise ValueError("VAE failed")
                return test.device

            encode_tiled = encode
            decode_tiled = decode

        self.vae_type = VAE
        comfy: Any = ModuleType("comfy")
        comfy.sd = ModuleType("comfy.sd")
        setattr(comfy.sd, "CLIP", Clip)
        setattr(comfy.sd, "VAE", VAE)
        self.clip_type = Clip
        with patch.dict(sys.modules, {"comfy": comfy, "comfy.sd": comfy.sd}):
            self.module = load_module("clip_runtime_device_guard")

    @contextmanager
    def guard(self, device, reason):
        _ = reason
        previous = self.device
        self.device = device
        try:
            yield
        finally:
            self.device = previous

    def test_direct_and_scheduled_encoder_device(self):
        self.module.install_clip_runtime_device_guard(self.guard)
        clip = self.clip_type()
        self.assertEqual(clip.encode_from_tokens("ok"), "cuda:1")
        self.assertEqual(clip.encode_from_tokens_scheduled("ok"), "cuda:1")
        self.assertEqual(self.device, "cuda:0")

    def test_exception_restores_device_and_patch_is_idempotent(self):
        self.module.install_clip_runtime_device_guard(self.guard)
        original = self.clip_type.encode_from_tokens
        self.module.install_clip_runtime_device_guard(self.guard)
        self.assertIs(original, self.clip_type.encode_from_tokens)
        with self.assertRaises(ValueError):
            self.clip_type().encode_from_tokens_scheduled("error")
        self.assertEqual(self.device, "cuda:0")

    def test_vae_standard_tiled_and_exception_restoration(self):
        self.module.install_vae_runtime_device_guard(self.guard)
        vae = self.vae_type()
        for method in (vae.encode, vae.decode, vae.encode_tiled, vae.decode_tiled):
            self.assertEqual(method("ok"), "cuda:1")
            self.assertEqual(self.device, "cuda:0")
        with self.assertRaises(ValueError):
            vae.encode_tiled("error")
        self.assertEqual(self.device, "cuda:0")


class CompatibilityTests(unittest.TestCase):
    def test_native_kitchen_versions_are_not_patched(self):
        module = load_module("compatibility")
        self.assertTrue(module.needs_legacy_dlpack_guard("0.2.34"))
        for version in ("0.2.35", "0.2.37", "1.0.0", "unknown"):
            self.assertFalse(module.needs_legacy_dlpack_guard(version))

    def test_distorch_patcher_restores_alias_on_exception(self):
        module = load_module("compatibility")
        dynamic, legacy = object(), object()
        patcher = SimpleNamespace(CoreModelPatcher=dynamic, ModelPatcher=legacy)
        with self.assertRaises(ValueError):
            with module.legacy_patcher_for_distorch(patcher, True):
                self.assertIs(patcher.CoreModelPatcher, legacy)
                raise ValueError("loader failure")
        self.assertIs(patcher.CoreModelPatcher, dynamic)
        with module.legacy_patcher_for_distorch(patcher, False):
            self.assertIs(patcher.CoreModelPatcher, dynamic)


class AimdoTests(unittest.TestCase):
    def setUp(self):
        fake: Any = ModuleType("comfy_aimdo")
        fake.control = SimpleNamespace(init_device=Mock(), devctxs=[])
        with patch.dict(sys.modules, {"comfy_aimdo": fake}):
            self.module = load_module("prestartup_script")
        self.original = Mock(return_value=False)
        self.original._multigpu_all_devices = False
        self.control = SimpleNamespace(
            init_device=self.original, init_devices=Mock(return_value=True), devctxs=[]
        )
        self.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)

    def test_all_devices_and_headroom_are_preserved(self):
        self.module.install_aimdo_device_init_guard(self.control, self.cuda)
        self.assertTrue(self.control.init_device(1, extra_vram_headroom=123))
        self.control.init_devices.assert_called_once_with([0, (1, 123)])
        self.original.assert_not_called()
        self.assertFalse(self.module.install_aimdo_device_init_guard(self.control, self.cuda))

    def test_existing_contexts_are_not_reinitialized(self):
        self.control.devctxs = [object()]
        self.module.install_aimdo_device_init_guard(self.control, self.cuda)
        self.assertFalse(self.control.init_device(1))
        self.control.init_devices.assert_not_called()
        self.original.assert_called_once()

    def test_single_device_and_cpu_delegate(self):
        self.cuda.device_count = lambda: 1
        self.module.install_aimdo_device_init_guard(self.control, self.cuda)
        self.control.init_device(0, 42)
        self.original.assert_called_once_with(0, extra_vram_headroom=42)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.registry = {}
        folder = ModuleType("folder_paths")
        nodes: Any = ModuleType("nodes")
        nodes.NODE_CLASS_MAPPINGS = self.registry
        device: Any = ModuleType("mgpu_regression.device_utils")
        device.get_device_list = lambda: ["cpu", "cuda:0", "cuda:1"]
        with patch.dict(
            sys.modules,
            {"folder_paths": folder, "nodes": nodes, "mgpu_regression.device_utils": device},
        ):
            self.nodes = load_module("nodes")
            self.wrappers = load_module("wrappers")

    def test_florence_tracks_schema_and_keyword_arguments(self):
        for name in ("Florence2ModelLoader", "DownloadAndLoadFlorence2Model"):
            schema = {"required": {"model": (["new-model"],), "precision": (["bf16"],)}}
            upstream = Mock()
            upstream.INPUT_TYPES.return_value = schema
            self.registry[name] = upstream
            node = getattr(self.nodes, name)
            self.assertIs(node.INPUT_TYPES(), schema)
            node().loadmodel(model="new-model", precision="bf16", future_option=True)
            upstream.return_value.loadmodel.assert_called_once_with(
                model="new-model", precision="bf16", future_option=True
            )

    def test_ltx_passes_unet_and_vae_config(self):
        upstream = Mock()
        self.registry["LTXVLoader"] = upstream
        config = {"test_setting": 42}
        loader = self.nodes.LTXVLoader()
        loader._load_unet("cuda:0", "cpu", {}, 16, torch.float16, config=config)
        upstream.return_value._load_unet.assert_called_once_with(
            "cuda:0", "cpu", {}, 16, torch.float16, config=config
        )
        loader._load_vae({}, config=config)
        upstream.return_value._load_vae.assert_called_once_with({}, config=config)

    def test_schema_copy_does_not_mutate_or_deepcopy_upstream(self):
        class NonCopyable:
            def __deepcopy__(self, memo):
                raise AssertionError("Schema metadata must not be deep-copied")

        schema = {"required": {"model": ("MODEL",)}, "optional": {"metadata": NonCopyable()}}
        base = type("Base", (), {"INPUT_TYPES": classmethod(lambda cls: schema)})
        package = SimpleNamespace(
            cuda_device_guard=Mock(), get_current_device=Mock(), set_current_device=Mock()
        )
        with patch.dict(sys.modules, {"mgpu_regression": package}):
            wrapped = self.wrappers.override_class(base)
        actual = wrapped.INPUT_TYPES()
        self.assertNotIn("device", schema["optional"])
        self.assertIn("device", actual["optional"])
        self.assertIs(actual["optional"]["metadata"], schema["optional"]["metadata"])


if __name__ == "__main__":
    unittest.main()
