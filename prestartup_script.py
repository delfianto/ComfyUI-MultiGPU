"""ComfyUI prestartup entry point, before importing the runtime package."""

import runpy
from pathlib import Path

# Importing src as a package here would load ComfyUI/Torch before core has
# configured CUDA visibility. Execute just the independent aimdo initializer.
runpy.run_path(str(Path(__file__).parent / "src" / "prestartup_script.py"))
