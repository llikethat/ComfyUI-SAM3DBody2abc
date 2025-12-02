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

from .nodes.video_batch_processor import (
    SAM3DBodyBatchProcessor,
    SAM3DBodySequenceProcess,
    SAM3DBodyModelDebug
)
from .nodes.animated_export import (
    ExportAnimatedAlembic,
    ExportAnimatedFBX,
    ExportAnimatedMesh
)
from .nodes.mesh_accumulator import (
    MeshSequenceAccumulator,
    MeshSequenceFromSAM3DBody,
    MeshSequencePreview,
    MeshSequenceSmooth,
    ClearMeshSequence
)
from .nodes.overlay_renderer import (
    RenderMeshOverlay,
    RenderMeshOverlayBatch
)

NODE_CLASS_MAPPINGS = {
    # Video/Batch Processing
    "SAM3DBody2abc_BatchProcessor": SAM3DBodyBatchProcessor,
    "SAM3DBody2abc_SequenceProcess": SAM3DBodySequenceProcess,
    "SAM3DBody2abc_ModelDebug": SAM3DBodyModelDebug,
    
    # Animated Export
    "SAM3DBody2abc_ExportAlembic": ExportAnimatedAlembic,
    "SAM3DBody2abc_ExportFBX": ExportAnimatedFBX,
    "SAM3DBody2abc_ExportMesh": ExportAnimatedMesh,
    
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
    "SAM3DBody2abc_ModelDebug": "🔍 SAM3DBody2abc Model Debug",
    
    # Animated Export
    "SAM3DBody2abc_ExportAlembic": "📦 SAM3DBody2abc Export Alembic (.abc)",
    "SAM3DBody2abc_ExportFBX": "🦴 SAM3DBody2abc Export FBX Skeleton",
    "SAM3DBody2abc_ExportMesh": "💾 SAM3DBody2abc Export All Formats",
    
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

__version__ = "2.0.0"
__author__ = "Custom Extension"
