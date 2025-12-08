"""
SAM3DBody2abc
Extension for ComfyUI-SAM3DBody that adds video batch processing
and animated export capabilities.

This extension works WITH the existing ComfyUI-SAM3DBody node, extending it with:
- Video/image sequence batch processing
- Animated Alembic geometry export (full timeline in single file)
- Animated FBX export with rigged skeleton (requires Blender)
- BVH skeleton animation export (universal mocap format)
- Real-time mesh overlay visualization on video frames

Prerequisites:
- ComfyUI-SAM3DBody (https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody)
- ComfyUI-VideoHelperSuite [optional]
- Blender [for FBX and Alembic export]

Author: Custom Extension
License: MIT
Version: 2.5.0
"""

import os
import sys
import importlib.util

__version__ = "2.5.0"
__author__ = "Custom Extension"

_current_dir = os.path.dirname(os.path.abspath(__file__))
_nodes_dir = os.path.join(_current_dir, "nodes")

def _load_module(name, filepath):
    """Load a module from file path."""
    if not os.path.exists(filepath):
        print(f"[SAM3DBody2abc] Warning: Module not found: {filepath}")
        return None
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"[SAM3DBody2abc] Error loading {name}: {e}")
        return None

# Load core modules
_video_batch = _load_module("sam3dbody2abc_video_batch", os.path.join(_nodes_dir, "video_batch_processor.py"))
_animated_export = _load_module("sam3dbody2abc_animated_export", os.path.join(_nodes_dir, "animated_export.py"))
_mesh_accumulator = _load_module("sam3dbody2abc_mesh_accumulator", os.path.join(_nodes_dir, "mesh_accumulator.py"))
_overlay_renderer = _load_module("sam3dbody2abc_overlay_renderer", os.path.join(_nodes_dir, "overlay_renderer.py"))
_bvh_export = _load_module("sam3dbody2abc_bvh_export", os.path.join(_nodes_dir, "bvh_export.py"))
_fbx_export = _load_module("sam3dbody2abc_fbx_export", os.path.join(_nodes_dir, "fbx_export.py"))

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Video Batch Processing
if _video_batch:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_BatchProcessor"] = _video_batch.SAM3DBodyBatchProcessor
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_SequenceProcess"] = _video_batch.SAM3DBodySequenceProcess
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_BatchProcessor"] = "🎬 Batch Processor"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_SequenceProcess"] = "📹 Sequence Process"

# Export Nodes
if _animated_export:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_ExportAlembic"] = _animated_export.ExportAnimatedAlembic
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_ExportAlembic"] = "📦 Export Alembic (.abc)"

if _bvh_export:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_ExportBVH"] = _bvh_export.ExportBVH
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_ExportBVH"] = "📦 Export BVH Skeleton"

if _fbx_export:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_ExportAnimatedFBX"] = _fbx_export.ExportAnimatedFBX
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_ExportAnimatedFBX"] = "📦 Export Animated FBX"

# Mesh Sequence Tools
if _mesh_accumulator:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_Accumulator"] = _mesh_accumulator.MeshSequenceAccumulator
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_MeshToSequence"] = _mesh_accumulator.MeshSequenceFromSAM3DBody
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_Preview"] = _mesh_accumulator.MeshSequencePreview
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_Smooth"] = _mesh_accumulator.MeshSequenceSmooth
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_Clear"] = _mesh_accumulator.ClearMeshSequence
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_MayaCamera"] = _mesh_accumulator.MayaCameraScript
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_MultiToSequence"] = _mesh_accumulator.MultiOutputToMeshSequence
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_MultiAccumulator"] = _mesh_accumulator.MultiOutputBatchToSequence
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_Accumulator"] = "📋 Accumulator"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_MeshToSequence"] = "🔄 Mesh → Sequence"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_Preview"] = "👁️ Preview"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_Smooth"] = "〰️ Smooth"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_Clear"] = "🗑️ Clear"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_MayaCamera"] = "🎥 Maya Camera Script"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_MultiToSequence"] = "👥 Multi Output → Sequence"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_MultiAccumulator"] = "👥 Multi Output Accumulator"

# Overlay Rendering
if _overlay_renderer:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_Overlay"] = _overlay_renderer.SAM3DBody2abcOverlay
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_OverlayBatch"] = _overlay_renderer.SAM3DBody2abcOverlayBatch
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_Overlay"] = "🎨 Overlay"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_OverlayBatch"] = "🎨 Overlay Batch"

print(f"[SAM3DBody2abc] v{__version__} loaded {len(NODE_CLASS_MAPPINGS)} nodes")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
