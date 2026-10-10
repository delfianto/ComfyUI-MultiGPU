"""Exercise the stack-overflow fallback without needing CUDA or a ComfyUI server."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch

SOURCE = Path(__file__).resolve().parents[1] / "src/clip_dynamic_load_list_guard.py"


class DynamicLoadingTests(unittest.TestCase):
    def setUp(self):
        self.native = Mock(return_value="native")

        def native(*args, **kwargs):
            return self.native(*args, **kwargs)

        self.patcher: Any = type("Dynamic", (), {"_load_list": native})
        model_patcher = SimpleNamespace(
            ModelPatcherDynamic=self.patcher,
            QuantizedTensor=type("QuantizedTensor", (), {}),
            get_key_weight=Mock(return_value=(None, None, None)),
            low_vram_patch_estimate_vram=Mock(return_value=0),
        )
        self.management = SimpleNamespace(
            module_size=lambda module: sum(t.nbytes for t in module.state_dict().values())
        )
        comfy: Any = ModuleType("comfy")
        comfy.model_patcher = model_patcher
        comfy.model_management = self.management
        modules = {
            "comfy": comfy,
            "comfy.model_patcher": model_patcher,
            "comfy.model_management": self.management,
        }
        self.imports = patch.dict(sys.modules, modules)
        self.imports.start()
        self.addCleanup(self.imports.stop)
        spec = importlib.util.spec_from_file_location("dynamic_loading_test", SOURCE)
        assert spec is not None and spec.loader is not None
        self.guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.guard)

    @staticmethod
    def deep_model():
        root = torch.nn.Module()
        current = root
        for _ in range(sys.getrecursionlimit() + 20):
            child = torch.nn.Module()
            current.child = child
            current = child
        current.weight = torch.nn.Parameter(torch.ones(1))
        return root, current

    def test_deep_graph_bypasses_recursive_native_loader(self):
        model, leaf = self.deep_model()
        with self.assertRaises(RecursionError):
            list(model.named_parameters())
        self.guard.register_clip_dynamic_load_list_guard()
        result = self.patcher._load_list(SimpleNamespace(model=model, patches={}), for_dynamic=True)
        self.native.assert_not_called()
        self.assertEqual(len(result), 1)
        self.assertIs(result[0][-2], leaf)
        self.assertEqual(result[0][:3], (False, -4, 4))

    def test_parent_without_direct_parameters_is_not_a_loading_leaf(self):
        model: Any = torch.nn.Module()
        model.comfy_cast_weights = True
        model.child = torch.nn.Linear(2, 1)
        result = self.guard._safe_dynamic_load_list(SimpleNamespace(model=model, patches={}))
        self.assertEqual([row[-3] for row in result], ["child"])

    def test_default_parent_parameters_move_with_requested_dtype(self):
        model: Any = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.ones(1))
        model.weight_comfy_model_dtype = torch.float64
        model.child = torch.nn.Linear(2, 1)
        result = self.guard._safe_dynamic_load_list(
            SimpleNamespace(model=model, patches={}), default_device="cpu"
        )
        self.assertEqual(model.weight.dtype, torch.float64)
        self.assertEqual([row[-3] for row in result], ["child"])

    def test_shared_modules_and_tied_parameters_match_torch(self):
        model = torch.nn.Module()
        model.first = torch.nn.Linear(2, 1)
        model.shared = model.first
        model.other = torch.nn.Linear(2, 1)
        model.other.weight = model.first.weight
        self.assertEqual(
            list(self.guard._iter_named_modules_nonrecursive(model)), list(model.named_modules())
        )
        expected = [(name, id(param)) for name, param in model.named_parameters()]
        actual = [
            (name, id(param))
            for name, param in self.guard._iter_named_parameters_nonrecursive(model)
        ]
        self.assertEqual(actual, expected)

    def test_parameter_walker_terminates_on_a_back_reference(self):
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.ones(1))
        model.child = torch.nn.Module()
        model.child.parent = model
        self.assertEqual(
            [name for name, _ in self.guard._iter_named_parameters_nonrecursive(model)],
            ["weight"],
        )

    def test_registration_is_idempotent_and_preserves_ordinary_and_legacy_calls(self):
        self.assertTrue(self.guard.register_clip_dynamic_load_list_guard())
        installed = self.patcher._load_list
        self.assertFalse(self.guard.register_clip_dynamic_load_list_guard())
        self.assertIs(self.patcher._load_list, installed)
        ordinary = SimpleNamespace(model=torch.nn.Linear(2, 1))
        self.assertEqual(installed(ordinary, for_dynamic=True, default_device="cpu"), "native")
        deep, _ = self.deep_model()
        legacy = SimpleNamespace(model=deep)
        self.assertEqual(installed(legacy, for_dynamic=False), "native")
        self.assertEqual(self.native.call_count, 2)
