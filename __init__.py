"""ComfyUI entry point; the implementation lives in src/."""

from . import src as _implementation

NODE_CLASS_MAPPINGS = _implementation.NODE_CLASS_MAPPINGS
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "WEB_DIRECTORY"]


def __getattr__(name):
    """Preserve access to the historical module's helpers for integrations."""
    return getattr(_implementation, name)
