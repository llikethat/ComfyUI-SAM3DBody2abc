"""
SAM3DBody2abc - Video to Animated FBX Export

Extends SAM3DBody with video processing and animated FBX export.

Workflow:
    Option 1 (Batch): Load Video → 🎬 Video Batch Processor → 📦 Export Animated FBX
    Option 2 (Manual): SAM3DBody Process → 📋 Skeleton Accumulator → 📦 Export Animated FBX

Fixed settings:
- Scale: 1.0
- Up axis: Y

Version: 3.0.0
"""

__version__ = "3.0.0"

import os
import sys
import importlib.util

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _load_module(name: str, path: str):
    """Load module from path."""
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


# Paths
_base = os.path.dirname(os.path.abspath(__file__))
_nodes = os.path.join(_base, "nodes")

# Load modules
_skeleton_acc = _load_module("sam3d2abc_skeleton_acc", os.path.join(_nodes, "skeleton_accumulator.py"))
_fbx_export = _load_module("sam3d2abc_fbx_export", os.path.join(_nodes, "fbx_export.py"))
_video_proc = _load_module("sam3d2abc_video_proc", os.path.join(_nodes, "video_processor.py"))

# Register skeleton accumulator
if _skeleton_acc:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_SkeletonAccumulator"] = _skeleton_acc.SkeletonAccumulator
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_ExportSkeletonJSON"] = _skeleton_acc.ExportSkeletonSequenceJSON
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_ClearAccumulator"] = _skeleton_acc.ClearAccumulator
    
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_SkeletonAccumulator"] = "📋 Skeleton Accumulator"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_ExportSkeletonJSON"] = "💾 Export Skeleton JSON"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_ClearAccumulator"] = "🗑️ Clear Accumulator"

# Register FBX export
if _fbx_export:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_ExportAnimatedFBX"] = _fbx_export.ExportAnimatedFBX
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_ExportFBXFromJSON"] = _fbx_export.ExportAnimatedFBXFromJSON
    
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_ExportAnimatedFBX"] = "📦 Export Animated FBX"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_ExportFBXFromJSON"] = "📦 Export FBX from JSON"

# Register video processor
if _video_proc:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_VideoBatchProcessor"] = _video_proc.VideoBatchProcessor
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_VideoBatchProcessor"] = "🎬 Video Batch Processor"

# Print loaded nodes
print(f"[SAM3DBody2abc] v{__version__} loaded {len(NODE_CLASS_MAPPINGS)} nodes:")
for name in NODE_CLASS_MAPPINGS:
    print(f"  - {NODE_DISPLAY_NAME_MAPPINGS.get(name, name)}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
