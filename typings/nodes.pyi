from typing import Any

NODE_CLASS_MAPPINGS: dict[str, Any]
CheckpointLoaderSimple: Any
CLIPLoader: Any
DualCLIPLoader: Any

async def init_extra_nodes(init_custom_nodes: bool = True, init_api_nodes: bool = True) -> None: ...
