"""
SAM3DBody2abc - Video to Animated FBX Export

v5.0 Architecture: Stabilization-First Pipeline
================================================

The v5.0 pipeline removes camera motion BEFORE pose estimation:

    Video Input
        │
        ├──► SAM3 Segmentation ──► Foreground Mask
        │
        ├──► Intrinsics Estimator ──► Camera K
        │                                │
        └──► Camera Solver V2 ◄──────────┘
                    │
                    ├── TAPIR background tracking
                    ├── Shot classification (rotation/translation/mixed)
                    ├── Camera solve
                    │
                    ▼
             Video Stabilizer ──► Stabilized Frames
                    │
                    ▼
             SAM3DBody ◄── External intrinsics
                    │
                    ▼
             FBX Export ◄── Restore camera motion

Key v5.0 Features:
- TAPIR-based temporal point tracking (not frame-pair matching)
- Automatic shot type classification
- Video stabilization before pose estimation
- Rainbow trail debug visualization
- Unified intrinsics source

Legacy v4.x nodes are still available with "(Legacy v4)" suffix.

Version: 5.0.0
"""

__version__ = "5.0.24"

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


_base = os.path.dirname(os.path.abspath(__file__))
_nodes = os.path.join(_base, "nodes")

# Load modules
_accumulator = _load_module("sam3d2abc_accumulator", os.path.join(_nodes, "accumulator.py"))
_fbx_export = _load_module("sam3d2abc_fbx_export", os.path.join(_nodes, "fbx_export.py"))
_video_proc = _load_module("sam3d2abc_video_proc", os.path.join(_nodes, "video_processor.py"))
_fbx_viewer = _load_module("sam3d2abc_fbx_viewer", os.path.join(_nodes, "fbx_viewer.py"))
_verify_overlay = _load_module("sam3d2abc_verify_overlay", os.path.join(_nodes, "verify_overlay.py"))
_camera_solver = _load_module("sam3d2abc_camera_solver", os.path.join(_nodes, "camera_solver.py"))
_moge_intrinsics = _load_module("sam3d2abc_moge_intrinsics", os.path.join(_nodes, "moge_intrinsics.py"))
_colmap_bridge = _load_module("sam3d2abc_colmap_bridge", os.path.join(_nodes, "colmap_bridge.py"))
_motion_analyzer = _load_module("sam3d2abc_motion_analyzer", os.path.join(_nodes, "motion_analyzer.py"))

# v5.0 new modules
_intrinsics_estimator = _load_module("sam3d2abc_intrinsics_estimator", os.path.join(_nodes, "intrinsics_estimator.py"))
_intrinsics_from_sam3dbody = _load_module("sam3d2abc_intrinsics_from_sam3dbody", os.path.join(_nodes, "intrinsics_from_sam3dbody.py"))
_camera_solver_v2 = _load_module("sam3d2abc_camera_solver_v2", os.path.join(_nodes, "camera_solver_v2.py"))

# Register accumulator nodes
if _accumulator:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_MeshAccumulator"] = _accumulator.MeshDataAccumulator
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_ExportJSON"] = _accumulator.ExportMeshSequenceJSON
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_Clear"] = _accumulator.ClearAccumulator
    
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_MeshAccumulator"] = "📋 Mesh Data Accumulator"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_ExportJSON"] = "💾 Export Sequence JSON"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_Clear"] = "🗑️ Clear Accumulator"

# Register FBX export nodes
if _fbx_export:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_ExportAnimatedFBX"] = _fbx_export.ExportAnimatedFBX
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_ExportFBXFromJSON"] = _fbx_export.ExportFBXFromJSON
    
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_ExportAnimatedFBX"] = "📦 Export Animated FBX"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_ExportFBXFromJSON"] = "📦 Export FBX from JSON"

# Register video processor
if _video_proc:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_VideoBatchProcessor"] = _video_proc.VideoBatchProcessor
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_VideoBatchProcessor"] = "🎬 Video Batch Processor"

# Register FBX viewer
if _fbx_viewer:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_FBXAnimationViewer"] = _fbx_viewer.FBXAnimationViewer
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_FBXAnimationViewer"] = "🎥 FBX Animation Viewer"

# Register verification overlay nodes
if _verify_overlay:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_VerifyOverlay"] = _verify_overlay.VerifyOverlay
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_VerifyOverlayBatch"] = _verify_overlay.VerifyOverlayBatch
    
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_VerifyOverlay"] = "🔍 Verify Overlay"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_VerifyOverlayBatch"] = "🔍 Verify Overlay (Sequence)"

# Register camera solver nodes
if _camera_solver:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_CameraSolver"] = _camera_solver.CameraSolver
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_CameraSolverLegacy"] = _camera_solver.CameraSolver  # Alias for backwards compat
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_CameraDataFromJSON"] = _camera_solver.CameraDataFromJSON
    
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_CameraSolver"] = "📷 Camera Solver (Legacy v4)"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_CameraSolverLegacy"] = "📷 Camera Solver (Legacy v4)"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_CameraDataFromJSON"] = "📷 Camera Extrinsics from JSON"

# Register v5.0 Camera Solver V2 (TAPIR-based)
if _camera_solver_v2:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_CameraSolverV2"] = _camera_solver_v2.CameraSolverV2
    
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_CameraSolverV2"] = "📷 Camera Solver V2 (TAPIR)"

# Register v5.0 Video Stabilizer (Phase 2)
_video_stabilizer = _load_module("sam3d2abc_video_stabilizer", os.path.join(_nodes, "video_stabilizer.py"))
if _video_stabilizer:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_VideoStabilizer"] = _video_stabilizer.VideoStabilizer
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_StabilizationInfo"] = _video_stabilizer.StabilizationInfo
    
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_VideoStabilizer"] = "🎬 Video Stabilizer"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_StabilizationInfo"] = "🎬 Stabilization Info"

# Register COLMAP bridge node
if _colmap_bridge:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_COLMAPBridge"] = _colmap_bridge.COLMAPToExtrinsicsBridge
    
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_COLMAPBridge"] = "🔄 COLMAP → Extrinsics Bridge"

# Register MoGe intrinsics nodes
if _moge_intrinsics:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_MoGe2Intrinsics"] = _moge_intrinsics.MoGe2IntrinsicsEstimator
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_ApplyIntrinsicsToMesh"] = _moge_intrinsics.ApplyIntrinsicsToMeshSequence
    
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_MoGe2Intrinsics"] = "📷 MoGe2 Intrinsics Estimator"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_ApplyIntrinsicsToMesh"] = "📷 Apply Intrinsics to Mesh"

# Register motion analyzer nodes
if _motion_analyzer:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_MotionAnalyzer"] = _motion_analyzer.MotionAnalyzer
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_ScaleInfoDisplay"] = _motion_analyzer.ScaleInfoDisplay
    
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_MotionAnalyzer"] = "📊 Motion Analyzer"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_ScaleInfoDisplay"] = "📏 Scale Info Display"

# Register v5.0 intrinsics estimator nodes
if _intrinsics_estimator:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_IntrinsicsEstimator"] = _intrinsics_estimator.IntrinsicsEstimator
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_IntrinsicsInfo"] = _intrinsics_estimator.IntrinsicsInfo
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_IntrinsicsFromJSON"] = _intrinsics_estimator.IntrinsicsFromJSON
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_IntrinsicsToJSON"] = _intrinsics_estimator.IntrinsicsToJSON
    
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_IntrinsicsEstimator"] = "📷 Intrinsics Estimator (v5.0)"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_IntrinsicsInfo"] = "📷 Intrinsics Info"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_IntrinsicsFromJSON"] = "📷 Intrinsics from JSON"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_IntrinsicsToJSON"] = "📷 Intrinsics to JSON"

# Register v5.0 intrinsics from SAM3DBody (single frame)
if _intrinsics_from_sam3dbody:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_IntrinsicsFromSAM3DBody"] = _intrinsics_from_sam3dbody.IntrinsicsFromSAM3DBody
    
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_IntrinsicsFromSAM3DBody"] = "📷 Intrinsics from SAM3DBody"

# Register v5.0 SAM3DBody processor (float32 wrapper)
_sam3dbody_process = _load_module("sam3d2abc_sam3dbody_process", os.path.join(_nodes, "sam3dbody_process.py"))
if _sam3dbody_process:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_SAM3DBodyContext"] = _sam3dbody_process.SAM3DBodyProcess
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_SAM3DBodyFloat32Patch"] = _sam3dbody_process.SAM3DBodyFloat32Patch
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_SAM3DBodyConfigHelper"] = _sam3dbody_process.SAM3DBodyConfigHelper
    
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_SAM3DBodyContext"] = "🦴 SAM3DBody Context (Float32)"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_SAM3DBodyFloat32Patch"] = "🦴 SAM3DBody Float32 Patch"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_SAM3DBodyConfigHelper"] = "🦴 SAM3DBody BFloat16 Fix Help"

# Print loaded nodes
print(f"[SAM3DBody2abc] v{__version__} loaded {len(NODE_CLASS_MAPPINGS)} nodes:")
for name in NODE_CLASS_MAPPINGS:
    print(f"  - {NODE_DISPLAY_NAME_MAPPINGS.get(name, name)}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Web extension directory
WEB_DIRECTORY = "./web"
