"""
ComfyUI-SAM3DBody-Video-Alembic
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
    SAM3DBodySequenceProcess
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
    "SAM3DBodyBatchProcessor": SAM3DBodyBatchProcessor,
    "SAM3DBodySequenceProcess": SAM3DBodySequenceProcess,
    
    # Animated Export
    "ExportAnimatedAlembic": ExportAnimatedAlembic,
    "ExportAnimatedFBX": ExportAnimatedFBX,
    "ExportAnimatedMesh": ExportAnimatedMesh,
    
    # Mesh Sequence Management
    "MeshSequenceAccumulator": MeshSequenceAccumulator,
    "MeshSequenceFromSAM3DBody": MeshSequenceFromSAM3DBody,
    "MeshSequencePreview": MeshSequencePreview,
    "MeshSequenceSmooth": MeshSequenceSmooth,
    "ClearMeshSequence": ClearMeshSequence,
    
    # Overlay Rendering
    "RenderMeshOverlay": RenderMeshOverlay,
    "RenderMeshOverlayBatch": RenderMeshOverlayBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Video/Batch Processing
    "SAM3DBodyBatchProcessor": "🎬 SAM3DBody Batch Processor",
    "SAM3DBodySequenceProcess": "📹 Process Image Sequence → SAM3DBody",
    
    # Animated Export
    "ExportAnimatedAlembic": "📦 Export Animated Alembic (.abc)",
    "ExportAnimatedFBX": "🦴 Export Animated Skeleton FBX",
    "ExportAnimatedMesh": "💾 Export Animated Mesh (All Formats)",
    
    # Mesh Sequence Management
    "MeshSequenceAccumulator": "📋 Mesh Sequence Accumulator",
    "MeshSequenceFromSAM3DBody": "🔄 Convert SAM3DBody Mesh → Sequence",
    "MeshSequencePreview": "👁️ Preview Mesh Sequence",
    "MeshSequenceSmooth": "〰️ Smooth Mesh Sequence",
    "ClearMeshSequence": "🗑️ Clear Mesh Sequence",
    
    # Overlay Rendering
    "RenderMeshOverlay": "🎨 Render Mesh Overlay",
    "RenderMeshOverlayBatch": "🎨 Render Mesh Overlay (Batch)",
}

# Custom type for mesh sequences
MESH_SEQUENCE_TYPE = "MESH_SEQUENCE"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

__version__ = "2.0.0"
__author__ = "Custom Extension"
