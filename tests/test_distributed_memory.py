"""Mixed-storage reservation and donor-GPU eviction regressions."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from test_regressions import load_module


class DistributedMemoryTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module("distributed_memory")

    def tensor(self, device, size, pointer):
        return SimpleNamespace(
            device=device,
            untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: pointer, nbytes=lambda: size),
        )

    def model(self, parameters, buffers=()):
        return SimpleNamespace(parameters=lambda: iter(parameters), buffers=lambda: iter(buffers))

    def test_real_tied_parameters_and_buffer_views_count_one_storage(self):
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.ones(8))
        model.register_buffer("alias", model.weight.detach()[:2])
        model.register_buffer("extra", torch.ones(3, dtype=torch.uint8))
        self.assertEqual(self.module.resident_storage_bytes(model), {torch.device("cpu"): 35})

    def test_packed_weights_include_scale_storage_without_dequantization(self):
        data = self.tensor("cuda:1", 8, 100)
        scale = self.tensor("cuda:1", 4, 200)
        quantized = SimpleNamespace(
            __tensor_flatten__=lambda: (["packed", "scale"], None),
            packed=data,
            scale=scale,
            numel=Mock(side_effect=AssertionError("must not evaluate logical weights")),
        )
        model = self.model([quantized, quantized], [scale, self.tensor("cpu", 2, 300)])
        self.assertEqual(
            self.module.resident_storage_bytes(model),
            {
                torch.device("cuda:1"): 12,
                torch.device("cpu"): 2,
            },
        )

    def test_plan_reserves_compute_and_donor_but_not_already_resident_storage(self):
        pending = self.model([self.tensor("cpu", 20, 1)])
        partial = self.model([self.tensor("cuda:1", 6, 2)], [self.tensor("cpu", 4, 3)])
        resident = self.model([self.tensor("cuda:0", 30, 4)])
        plan = {
            "device_assignments": {
                "cuda:0": [("a", pending, "Linear", 20), ("b", resident, "Linear", 30)],
                "cuda:1": [("c", partial, "Linear", 10)],
                "cpu": [],
            }
        }
        self.assertEqual(
            self.module.allocation_memory_required(plan),
            {
                torch.device("cuda:0"): 20,
                torch.device("cuda:1"): 4,
            },
        )

    def loaded(self, device, resident, freed, metadata=True):
        inner = self.model([self.tensor(resident, freed, id(freed))])
        if metadata:
            inner._distorch_v2_meta = {"full_allocation": "#cuda:0;6;cuda:1"}
        loaded = SimpleNamespace(
            device=device, model=SimpleNamespace(model=inner), is_dead=lambda: False
        )

        def unload():
            self.free += freed
            return True

        loaded.model_unload = Mock(side_effect=unload)
        return loaded

    def management(self, loaded):
        self.free = 0
        return SimpleNamespace(
            current_loaded_models=loaded,
            get_free_memory=lambda _device: self.free,
            soft_empty_cache=Mock(),
        )

    def test_donor_residents_evict_after_core_and_stop_when_enough_memory_is_free(self):
        first = self.loaded("cuda:0", "cuda:1", 20)
        second = self.loaded("cuda:0", "cuda:1", 30)
        mm = self.management([first, second])
        core = Mock(return_value=[])
        free = self.module.preserve_distributed_eviction(core, mm)
        self.assertEqual(free(25, "cuda:1", for_dynamic=True, ram_required=99), [second])
        core.assert_called_once_with(25, "cuda:1", [], for_dynamic=True, ram_required=99)
        second.model_unload.assert_called_once_with()
        first.model_unload.assert_not_called()
        self.assertEqual(mm.current_loaded_models, [first])
        mm.soft_empty_cache.assert_called_once_with()

    def test_requested_models_and_unrelated_devices_are_never_evicted(self):
        protected = self.loaded("cuda:0", "cuda:1", 20)
        ordinary = self.loaded("cuda:0", "cuda:1", 30, metadata=False)
        other = self.loaded("cuda:0", "cpu", 40)
        native = self.loaded("cuda:1", "cuda:1", 50)
        dead = self.loaded("cuda:0", "cuda:1", 60)
        dead.is_dead = lambda: True
        mm = self.management([protected, ordinary, other, native, dead])
        free = self.module.preserve_distributed_eviction(Mock(return_value=[]), mm)
        self.assertEqual(free(100, "cuda:1", keep_loaded=iter([protected])), [])
        for loaded in mm.current_loaded_models:
            loaded.model_unload.assert_not_called()

    def test_core_results_arguments_and_non_gpu_requests_are_preserved(self):
        loaded = self.loaded("cuda:0", "cuda:1", 20)
        mm = self.management([loaded])
        sentinel = object()
        core = Mock(return_value=[sentinel])
        free = self.module.preserve_distributed_eviction(core, mm)
        for device in (None, "cpu"):
            self.assertEqual(free(100, device, [loaded], True, 10, 20), [sentinel])
            core.assert_called_with(100, device, [loaded], True, 10, 20)
        loaded.model_unload.assert_not_called()

    def test_sufficient_free_memory_and_failed_detach_keep_models_registered(self):
        loaded = self.loaded("cuda:0", "cuda:1", 20)
        mm = self.management([loaded])
        free = self.module.preserve_distributed_eviction(Mock(return_value=[]), mm)
        self.free = 100
        self.assertEqual(free(50, "cuda:1"), [])
        loaded.model_unload.assert_not_called()
        self.free = 0
        loaded.model_unload.return_value = False
        loaded.model_unload.side_effect = None
        self.assertEqual(free(50, "cuda:1"), [])
        self.assertEqual(mm.current_loaded_models, [loaded])


class QuantizedAllocationTests(unittest.TestCase):
    def test_only_cross_cuda_allocations_pause_allocation_recording(self):
        from contextlib import contextmanager

        module = load_module("distributed_memory")
        active = True
        pauses = []

        @contextmanager
        def pause():
            nonlocal active
            previous = active
            active = False
            pauses.append(True)
            try:
                yield
            finally:
                active = previous

        def allocate(weight, args, kwargs):
            return (active, weight, args, kwargs)

        guarded = module.guard_quantized_empty_like(allocate, pause)
        for source, target, should_pause in [
            ("cuda:1", "cuda:0", True),
            ("cuda:0", "cuda:1", True),
            ("cuda:1", "cuda:1", False),
            ("cpu", "cuda:0", False),
            ("cuda:1", "cpu", False),
            ("cuda:0", None, False),
        ]:
            with self.subTest(source=source, target=target):
                weight = SimpleNamespace(device=source)
                options = {"device": target, "dtype": torch.bfloat16, "memory_format": "format"}
                result = guarded(weight, (weight,), options)
                self.assertEqual(result, (not should_pause, weight, (weight,), options))
                self.assertTrue(active)
        self.assertEqual(len(pauses), 2)

    def test_recording_resumes_after_allocation_failure(self):
        from contextlib import contextmanager

        module = load_module("distributed_memory")
        events = []

        @contextmanager
        def pause():
            events.append("pause")
            try:
                yield
            finally:
                events.append("resume")

        guarded = module.guard_quantized_empty_like(
            Mock(side_effect=ValueError("allocation failure")), pause
        )
        with self.assertRaisesRegex(ValueError, "allocation failure"):
            guarded(SimpleNamespace(device="cuda:1"), (), {"device": torch.device("cuda:0")})
        self.assertEqual(events, ["pause", "resume"])

    def test_registration_is_optional_idempotent_and_only_changes_empty_like(self):
        module = load_module("distributed_memory")
        pause = Mock()
        operation = torch.ops.aten.empty_like.default
        other = object()
        handler = lambda *_args: None
        base = SimpleNamespace(_DISPATCH_TABLE={operation: handler, "other": other})
        self.assertTrue(module.register_quantized_allocation_guard(base, pause))
        guarded = base._DISPATCH_TABLE[operation]
        self.assertIsNot(guarded, handler)
        self.assertTrue(module.register_quantized_allocation_guard(base, pause))
        self.assertIs(base._DISPATCH_TABLE[operation], guarded)
        self.assertIs(base._DISPATCH_TABLE["other"], other)
        self.assertFalse(module.register_quantized_allocation_guard(SimpleNamespace(), pause))
        self.assertFalse(
            module.register_quantized_allocation_guard(SimpleNamespace(_DISPATCH_TABLE={}), pause)
        )
