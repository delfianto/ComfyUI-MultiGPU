"""Small compatibility decisions kept separate from ComfyUI startup patches."""

from contextlib import contextmanager

from packaging.version import InvalidVersion, Version


def needs_legacy_dlpack_guard(version: str) -> bool:
    try:
        return Version(version) < Version("0.2.35")
    except InvalidVersion:
        # An unknown/new backend should keep its own export implementation.
        return False


@contextmanager
def legacy_patcher_for_distorch(model_patcher, enabled: bool):
    original = model_patcher.CoreModelPatcher
    if enabled:
        model_patcher.CoreModelPatcher = model_patcher.ModelPatcher
    try:
        yield
    finally:
        model_patcher.CoreModelPatcher = original
