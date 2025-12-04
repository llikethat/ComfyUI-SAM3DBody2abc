"""
SAM3DBody2abc
Extension for ComfyUI-SAM3DBody that adds video batch processing
and animated export to Alembic (.abc) and FBX formats.

This extension works WITH the existing ComfyUI-SAM3DBody node, extending it with:
- Video/image sequence batch processing
- Animated Alembic geometry export (full timeline, not per-frame)
- Animated FBX skeleton export (full timeline, not per-frame)
- Real-time mesh overlay visualization on video frames

Prerequisites:
- ComfyUI-SAM3DBody (https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody)
- ComfyUI-VideoHelperSuite (https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) [optional]

Author: Custom Extension
License: MIT
Version: 2.0.0
"""

import os
import sys
import importlib.util

# Get the directory where this __init__.py is located
_current_dir = os.path.dirname(os.path.abspath(__file__))
_nodes_dir = os.path.join(_current_dir, "nodes")

def _load_module(name, filepath):
    """Load a module from file path (works regardless of folder name)."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load node modules using direct file paths
_video_batch = _load_module(
    "sam3dbody2abc_video_batch", 
    os.path.join(_nodes_dir, "video_batch_processor.py")
)
_animated_export = _load_module(
    "sam3dbody2abc_animated_export",
    os.path.join(_nodes_dir, "animated_export.py")
)
_mesh_accumulator = _load_module(
    "sam3dbody2abc_mesh_accumulator",
    os.path.join(_nodes_dir, "mesh_accumulator.py")
)
_overlay_renderer = _load_module(
    "sam3dbody2abc_overlay_renderer",
    os.path.join(_nodes_dir, "overlay_renderer.py")
)

# Import classes from loaded modules
SAM3DBodyBatchProcessor = _video_batch.SAM3DBodyBatchProcessor
SAM3DBodySequenceProcess = _video_batch.SAM3DBodySequenceProcess

ExportAnimatedAlembic = _animated_export.ExportAnimatedAlembic
ExportAnimatedFBX = _animated_export.ExportAnimatedFBX
ExportAnimatedMesh = _animated_export.ExportAnimatedMesh
ExportOBJSequence = _animated_export.ExportOBJSequence

MeshSequenceAccumulator = _mesh_accumulator.MeshSequenceAccumulator
MeshSequenceFromSAM3DBody = _mesh_accumulator.MeshSequenceFromSAM3DBody
MeshSequencePreview = _mesh_accumulator.MeshSequencePreview
MeshSequenceSmooth = _mesh_accumulator.MeshSequenceSmooth
ClearMeshSequence = _mesh_accumulator.ClearMeshSequence

RenderMeshOverlay = _overlay_renderer.SAM3DBody2abcOverlay
RenderMeshOverlayBatch = _overlay_renderer.SAM3DBody2abcOverlayBatch

NODE_CLASS_MAPPINGS = {
    # Video/Batch Processing
    "SAM3DBody2abc_BatchProcessor": SAM3DBodyBatchProcessor,
    "SAM3DBody2abc_SequenceProcess": SAM3DBodySequenceProcess,
    
    # Animated Export
    "SAM3DBody2abc_ExportAlembic": ExportAnimatedAlembic,
    "SAM3DBody2abc_ExportFBX": ExportAnimatedFBX,
    "SAM3DBody2abc_ExportMesh": ExportAnimatedMesh,
    "SAM3DBody2abc_ExportOBJSequence": ExportOBJSequence,
    
    # Mesh Sequence Management
    "SAM3DBody2abc_Accumulator": MeshSequenceAccumulator,
    "SAM3DBody2abc_MeshToSequence": MeshSequenceFromSAM3DBody,
    "SAM3DBody2abc_Preview": MeshSequencePreview,
    "SAM3DBody2abc_Smooth": MeshSequenceSmooth,
    "SAM3DBody2abc_Clear": ClearMeshSequence,
    
    # Overlay Rendering
    "SAM3DBody2abc_Overlay": RenderMeshOverlay,
    "SAM3DBody2abc_OverlayBatch": RenderMeshOverlayBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Video/Batch Processing
    "SAM3DBody2abc_BatchProcessor": "🎬 SAM3DBody2abc Batch Processor",
    "SAM3DBody2abc_SequenceProcess": "📹 SAM3DBody2abc Sequence Process",
    
    # Animated Export
    "SAM3DBody2abc_ExportAlembic": "📦 SAM3DBody2abc Export Alembic (.abc)",
    "SAM3DBody2abc_ExportFBX": "🦴 SAM3DBody2abc Export FBX Skeleton",
    "SAM3DBody2abc_ExportMesh": "💾 SAM3DBody2abc Export All Formats",
    "SAM3DBody2abc_ExportOBJSequence": "📁 SAM3DBody2abc Export OBJ Sequence",
    
    # Mesh Sequence Management
    "SAM3DBody2abc_Accumulator": "📋 SAM3DBody2abc Accumulator",
    "SAM3DBody2abc_MeshToSequence": "🔄 SAM3DBody2abc Mesh → Sequence",
    "SAM3DBody2abc_Preview": "👁️ SAM3DBody2abc Preview",
    "SAM3DBody2abc_Smooth": "〰️ SAM3DBody2abc Smooth",
    "SAM3DBody2abc_Clear": "🗑️ SAM3DBody2abc Clear",
    
    # Overlay Rendering
    "SAM3DBody2abc_Overlay": "🎨 SAM3DBody2abc Overlay",
    "SAM3DBody2abc_OverlayBatch": "🎨 SAM3DBody2abc Overlay Batch",
}

# Custom type for mesh sequences
# Note: Uses SAM3D_MODEL and SAM3D_MESH types from ComfyUI-SAM3DBody
MESH_SEQUENCE_TYPE = "MESH_SEQUENCE"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

__version__ = "2.2.2"
__author__ = "Custom Extension"
