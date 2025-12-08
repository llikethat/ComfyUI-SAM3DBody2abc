"""
SAM3DBody2abc - Video to Animated FBX Export Extension

This extension works with SAM3DBody's Process Image node to:
1. Accumulate frames with temporal smoothing
2. Export to JSON (intermediate format)
3. Convert to animated FBX

Workflow:
    Load Video → SAM3DBody Process Image (frame by frame) → Frame Accumulator → Apply Smoothing → Export FBX

Fixed settings:
- Scale: 1.0
- Up axis: Y

Version: 2.6.0
"""

__version__ = "2.6.0"

import os
import sys
import importlib.util

# Track loaded modules
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _load_module(name: str, path: str):
    """Dynamically load a module from path."""
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
    except Exception as e:
        print(f"[SAM3DBody2abc] Error loading {name}: {e}")
    return None


# Get base path
_base_path = os.path.dirname(os.path.abspath(__file__))
_nodes_path = os.path.join(_base_path, "nodes")

# Load modules
_frame_accumulator = _load_module(
    "sam3dbody2abc_frame_accumulator",
    os.path.join(_nodes_path, "frame_accumulator.py")
)

_fbx_export = _load_module(
    "sam3dbody2abc_fbx_export",
    os.path.join(_nodes_path, "fbx_export.py")
)

_batch_process = _load_module(
    "sam3dbody2abc_batch_process",
    os.path.join(_nodes_path, "batch_process.py")
)

# Register Frame Accumulator nodes
if _frame_accumulator:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_FrameAccumulator"] = _frame_accumulator.FrameAccumulator
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_ApplySmoothing"] = _frame_accumulator.ApplySmoothing
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_ExportJSON"] = _frame_accumulator.ExportSequenceJSON
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_ClearSequences"] = _frame_accumulator.ClearSequences
    
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_FrameAccumulator"] = "📋 Frame Accumulator"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_ApplySmoothing"] = "〰️ Apply Smoothing"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_ExportJSON"] = "💾 Export JSON"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_ClearSequences"] = "🗑️ Clear Sequences"

# Register FBX Export nodes
if _fbx_export:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_ExportFBX"] = _fbx_export.ExportFBX
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_ExportFBXDirect"] = _fbx_export.ExportFBXDirect
    
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_ExportFBX"] = "📦 Export FBX (from JSON)"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_ExportFBXDirect"] = "📦 Export FBX Direct"

# Register Batch Process node
if _batch_process:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_BatchProcess"] = _batch_process.BatchProcess
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_BatchProcess"] = "🎬 Batch Process"

# Print loaded nodes
print(f"[SAM3DBody2abc] v{__version__} loaded {len(NODE_CLASS_MAPPINGS)} nodes:")
for name in NODE_CLASS_MAPPINGS:
    display = NODE_DISPLAY_NAME_MAPPINGS.get(name, name)
    print(f"  - {display}")

# ComfyUI requires these
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
