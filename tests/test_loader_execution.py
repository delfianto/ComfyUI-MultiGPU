"""Loader contract, failures, cancellation and concurrent device isolation."""

import asyncio
import inspect
import sys
import unittest
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch
from test_regressions import SOURCE, load_module


class LoaderExecutionTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "device": "cuda:0",
            "text_encoder_device": "cuda:0",
            "unet_offload_device": "cpu",
        }
        self.package: Any = ModuleType("mgpu_regression")
        self.package.__path__ = [str(SOURCE)]
        for name in self.state:
            setattr(self.package, f"get_current_{name}", lambda name=name: self.state[name])
            setattr(
                self.package,
                f"set_current_{name}",
                lambda value, name=name: self.state.__setitem__(name, value),
            )
        self.cuda = 0

        @contextmanager
        def cuda_guard(device, **_kwargs):
            previous = self.cuda
            self.cuda = int(device.split(":")[1])
            try:
                yield
            finally:
                self.cuda = previous

        self.package.cuda_device_guard = cuda_guard
        devices: Any = ModuleType("mgpu_regression.device_utils")
        devices.get_device_list = lambda: ["cpu", "cuda:0", "cuda:1"]
        distorch: Any = ModuleType("mgpu_regression.distorch_2")
        self.registration = Mock()
        distorch.register_patched_safetensor_modelpatcher = self.registration
        self.core = object()
        self.legacy = object()
        self.patcher = SimpleNamespace(CoreModelPatcher=self.core, ModelPatcher=self.legacy)
        comfy: Any = ModuleType("comfy")
        comfy.model_patcher = self.patcher
        comfy.model_management = SimpleNamespace(current_loaded_models=[])
        self.modules = patch.dict(
            sys.modules,
            {
                "mgpu_regression": self.package,
                "mgpu_regression.device_utils": devices,
                "mgpu_regression.distorch_2": distorch,
                "comfy": comfy,
                "comfy.model_patcher": self.patcher,
                "comfy.model_management": comfy.model_management,
            },
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.wrappers: Any = load_module("wrappers")
        self.wrappers.logger = SimpleNamespace(info=Mock(), mgpu_mm_log=Mock())
        self.factories = [
            ("override_class", "device", "device", False),
            ("override_class_offload", "device", "device", False),
            ("override_class_clip", "device", "text_encoder_device", False),
            ("override_class_clip_no_device", "device", "text_encoder_device", False),
            ("override_class_with_distorch_gguf", "device", "device", True),
            ("override_class_with_distorch_gguf_v2", "compute_device", "device", True),
            ("override_class_with_distorch_clip", "device", "text_encoder_device", True),
            ("override_class_with_distorch_clip_no_device", "device", "text_encoder_device", True),
            ("override_class_with_distorch_safetensor_v2", "compute_device", "device", True),
            (
                "override_class_with_distorch_safetensor_v2_clip",
                "device",
                "text_encoder_device",
                True,
            ),
            (
                "override_class_with_distorch_safetensor_v2_clip_no_device",
                "device",
                "text_encoder_device",
                True,
            ),
        ]

    def node(self, factory, load):
        provider = type("Provider", (), {"FUNCTION": "load", "load": load})
        return getattr(self.wrappers, factory)(provider)()

    def assert_restored(self):
        self.assertEqual(
            self.state,
            {"device": "cuda:0", "text_encoder_device": "cuda:0", "unet_offload_device": "cpu"},
        )
        self.assertIs(self.patcher.CoreModelPatcher, self.core)
        self.assertEqual(self.cuda, 0)

    def test_florence_selects_text_encoder_and_restores_both_device_scopes(self):
        for asynchronous in (False, True):
            for fail in (False, True):
                with self.subTest(asynchronous=asynchronous, fail=fail):

                    def check(fail=fail):
                        self.assertEqual(self.state["device"], "cuda:1")
                        self.assertEqual(self.state["text_encoder_device"], "cuda:1")
                        self.assertEqual(self.state["unet_offload_device"], "cuda:0")
                        self.assertEqual(self.cuda, 1)
                        if fail:
                            raise ValueError("Florence provider failed")
                        return ({"patcher": object()},)

                    def sync_load(_self, **kwargs):
                        self.assertEqual(kwargs, {"model": "Florence-2-base", "precision": "fp16"})
                        return check()

                    async def async_load(_self, **kwargs):
                        await asyncio.sleep(0)
                        return sync_load(_self, **kwargs)

                    provider = type(
                        "FlorenceProvider",
                        (),
                        {
                            "FUNCTION": "loadmodel",
                            "loadmodel": async_load if asynchronous else sync_load,
                        },
                    )
                    node = self.wrappers.override_class_offload(provider, text_encoder=True)()

                    def execute(node=node, asynchronous=asynchronous):
                        result = node.override(
                            model="Florence-2-base",
                            precision="fp16",
                            device="cuda:1",
                            offload_device="cuda:0",
                        )
                        return asyncio.run(result) if asynchronous else result

                    if fail:
                        with self.assertRaisesRegex(ValueError, "Florence provider failed"):
                            execute()
                    else:
                        self.assertIn("patcher", execute()[0])
                    self.assert_restored()

    def test_sync_results_failures_and_metadata(self):
        for factory, parameter, state_key, distributed in self.factories:
            for failure in (None, "provider", "output", "registration"):
                with self.subTest(factory=factory, failure=failure):

                    def load(_self, state_key=state_key, failure=failure, **kwargs):
                        self.assertEqual(self.state[state_key], "cuda:1")
                        self.assertTrue(kwargs["future_option"])
                        if failure == "provider":
                            raise ValueError("provider failure")
                        return (
                            ()
                            if failure == "output"
                            else (SimpleNamespace(model=SimpleNamespace()),)
                        )

                    node = self.node(factory, load)
                    self.assertFalse(inspect.iscoroutinefunction(node.override))
                    kwargs = {parameter: "cuda:1", "future_option": True}
                    if distributed:
                        kwargs["virtual_vram_gb"] = 1.0
                    self.registration.side_effect = (
                        ValueError("registration failure") if failure == "registration" else None
                    )
                    if failure == "provider" or (
                        distributed and failure in ("output", "registration")
                    ):
                        with self.assertRaises((ValueError, IndexError)):
                            node.override(**kwargs)
                    else:
                        result = node.override(**kwargs)
                        if distributed:
                            self.assertIn(
                                "cuda:1;1.0;cpu",
                                result[0].model._distorch_v2_meta["full_allocation"],
                            )
                    self.assert_restored()

    def test_async_success_failure_and_cancellation_all_wrappers(self):
        async def exercise():
            for factory, parameter, state_key, distributed in self.factories:
                for failure in (None, "provider", "output", "cancel"):
                    with self.subTest(factory=factory, failure=failure):
                        entered = asyncio.Event()
                        resume = asyncio.Event()

                        def verify(state_key=state_key, factory=factory, distributed=distributed):
                            self.assertEqual(self.state[state_key], "cuda:1")
                            if "safetensor_v2" in factory and distributed:
                                self.assertIs(self.patcher.CoreModelPatcher, self.legacy)

                        async def load(
                            _self,
                            verify=verify,
                            entered=entered,
                            resume=resume,
                            failure=failure,
                            **kwargs,
                        ):
                            self.assertTrue(kwargs["future_option"])
                            verify()
                            entered.set()
                            try:
                                await resume.wait()
                                verify()
                                await asyncio.sleep(0)
                                verify()
                                if failure == "provider":
                                    raise ValueError("async provider failure")
                                return (
                                    ()
                                    if failure == "output"
                                    else (
                                        SimpleNamespace(
                                            patcher=SimpleNamespace(model=SimpleNamespace())
                                        ),
                                    )
                                )
                            finally:
                                verify()

                        node = self.node(factory, load)
                        self.assertTrue(inspect.iscoroutinefunction(node.override))
                        kwargs = {parameter: "cuda:1", "future_option": True}
                        if distributed:
                            kwargs["virtual_vram_gb"] = 1.0
                        if factory == "override_class_offload":
                            kwargs["offload_device"] = "cuda:1"
                        task = asyncio.create_task(node.override(**kwargs))
                        await entered.wait()
                        self.assert_restored()
                        if failure == "cancel":
                            task.cancel()
                        else:
                            resume.set()
                        if failure == "cancel":
                            with self.assertRaises(asyncio.CancelledError):
                                await task
                        elif failure == "provider" or (distributed and failure == "output"):
                            with self.assertRaises((ValueError, IndexError)):
                                await task
                        else:
                            result = await task
                            if distributed:
                                self.assertIn(
                                    "cuda:1;1.0;cpu",
                                    result[0].patcher.model._distorch_v2_meta["full_allocation"],
                                )
                        self.assert_restored()

        with (
            patch.object(torch.cuda, "is_initialized", return_value=True),
            patch.object(torch.cuda, "current_device", side_effect=lambda: self.cuda),
            patch.object(
                torch.cuda, "set_device", side_effect=lambda device: setattr(self, "cuda", device)
            ),
        ):
            asyncio.run(exercise())

    def test_async_setup_failure_and_empty_allocation_restore_core_alias(self):
        async def exercise():
            async def load(_self, **_kwargs):
                self.assertIs(self.patcher.CoreModelPatcher, self.core)
                await asyncio.sleep(0)
                self.assertIs(self.patcher.CoreModelPatcher, self.core)
                return (SimpleNamespace(model=SimpleNamespace()),)

            node = self.node("override_class_with_distorch_safetensor_v2", load)
            result = await node.override(compute_device="cuda:1", virtual_vram_gb=0.0)
            self.assertFalse(hasattr(result[0].model, "_distorch_v2_meta"))
            self.assert_restored()
            self.registration.side_effect = ValueError("setup failure")
            with self.assertRaisesRegex(ValueError, "setup failure"):
                await node.override(compute_device="cuda:1", virtual_vram_gb=1.0)
            self.assert_restored()

        asyncio.run(exercise())

    def test_concurrent_loaders_keep_independent_device_and_cuda_contexts(self):
        async def exercise():
            events = [asyncio.Event(), asyncio.Event()]

            async def load(_self, selected, **_kwargs):
                expected = f"cuda:{selected}"
                self.assertEqual(self.state["device"], expected)
                self.assertEqual(self.cuda, selected)
                events[selected].set()
                await events[1 - selected].wait()
                for _ in range(3):
                    await asyncio.sleep(0)
                    self.assertEqual(self.state["device"], expected)
                    self.assertEqual(self.cuda, selected)
                return (expected,)

            node = self.node("override_class", load)
            result = await asyncio.gather(
                *(node.override(device=f"cuda:{index}", selected=index) for index in range(2))
            )
            self.assertEqual(result, [("cuda:0",), ("cuda:1",)])
            self.assert_restored()

        with (
            patch.object(torch.cuda, "is_initialized", return_value=True),
            patch.object(torch.cuda, "current_device", side_effect=lambda: self.cuda),
            patch.object(
                torch.cuda, "set_device", side_effect=lambda device: setattr(self, "cuda", device)
            ),
        ):
            asyncio.run(exercise())
