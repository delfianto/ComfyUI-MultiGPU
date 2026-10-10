"""Exercise DisTorch's copied core scheduling without allocating model weights."""

import ast
import logging
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


class CoreLoadingContractTests(unittest.TestCase):
    def test_dependency_order_deduplication_pin_budget_and_protected_models(self):
        source = Path(__file__).resolve().parents[1] / "src/distorch_2.py"
        tree = ast.parse(source.read_text())
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "patched_load_models_gpu"
        )
        order = []

        class Model:
            def __init__(self, name, attachments=()):
                self.name = name
                self.attachments = attachments
                self.model = torch.nn.Module()
                self.load_device = torch.device("cuda:0")
                self.model_patches_to = Mock(side_effect=AssertionError("not a model list"))

            def model_patches_models(self):
                return self.attachments

            def is_dynamic(self):
                return False

            def is_clone(self, _other):
                return False

        def loaded(model):
            return SimpleNamespace(
                model=model,
                device=model.load_device,
                model_memory_required=lambda device: 13,
                model_memory=lambda: 13,
                model_load=lambda *a, **kw: order.append(model.name),
            )

        mm = SimpleNamespace(
            MIN_WEIGHT_MEMORY_RATIO=0.4,
            VRAMState=SimpleNamespace(DISABLED=0, LOW_VRAM=1, NORMAL_VRAM=2, NO_VRAM=3),
            cleanup_models_gc=Mock(),
            current_loaded_models=[],
            extra_reserved_memory=lambda: 0,
            free_memory=Mock(),
            get_free_memory=lambda device: 1000,
            is_device_cpu=lambda device: device.type == "cpu",
            lowvram_available=True,
            minimum_inference_memory=lambda: 10,
            vram_state=2,
            LoadedModel=loaded,
        )
        namespace = {
            "mm": mm,
            "torch": torch,
            "logging": logging,
            "logger": Mock(),
            "multigpu_memory_log": Mock(),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        attached = Model("attached")
        first = Model("first", [attached])
        second = Model("second")
        with patch.dict(sys.modules, {"comfy.model_management": mm}):
            namespace["patched_load_models_gpu"]([first, second, first], force_full_load=True)
        self.assertEqual(order, ["second", "attached", "first"])
        self.assertEqual(len(mm.current_loaded_models), 3)
        self.assertEqual(mm.free_memory.call_args.kwargs["pins_required"], 39)
        self.assertEqual(len(mm.free_memory.call_args.kwargs["keep_loaded"]), 3)
        for model in (first, second, attached):
            model.model_patches_to.assert_not_called()
