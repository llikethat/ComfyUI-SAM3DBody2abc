"""
SAM3DBody2abc - Video to Animated FBX Export

Extends SAM3DBody with video processing and animated FBX export.

Workflow:
    Load Video → SAM3 Segmentation → SAM3 Extract Masks
                                            ↓
    Load SAM3DBody → 🎬 Video Batch Processor ←──┘
                              ↓
                   📦 Export Animated FBX

Outputs match SAM3DBody Process:
- mesh_data (SAM3D_OUTPUT) → vertices, faces, joint_coords
- Uses SAM3DBodyExportFBX format for single frames
- Animated FBX has shape keys + skeleton keyframes

Version: 5.1.0
- NEW: 📐 Body Shape Lock node
  - Locks SMPL beta parameters across all frames
  - Eliminates body size/proportion flickering
  - Methods: Median, Mean, Trimmed Mean, First Frame, Best Frame, Weighted Mean
  - Outlier rejection for robust shape estimation
- NEW: 🔄 Pose Smoothing node
  - Quaternion-based joint rotation smoothing
  - Methods: Gaussian, Savitzky-Golay, Kalman, SLERP, Moving Average
  - Eliminates jitter while preserving motion dynamics
  - Optional trajectory (pred_cam_t) smoothing
- NEW: 🦶 Foot Contact Enforcer node
  - Automatic foot contact detection (height + velocity based)
  - Ground plane estimation
  - Foot pinning via root translation adjustment
  - Smooth transitions in/out of contacts
  - Reduces foot skating in motion capture
- NEW: 📹 SLAM Camera Solver node
  - Visual SLAM for world-coordinate camera poses
  - DPVO backend support (recommended)
  - Feature-based fallback when DPVO unavailable
  - Automatic scale estimation using person height
  - Compatible with FBX Export and Motion Analyzer
"""

__version__ = "5.1.0"

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
        import traceback
        print(f"[SAM3DBody2abc] Error loading {name}: {e}")
        traceback.print_exc()
    return None


_base = os.path.dirname(os.path.abspath(__file__))
_nodes = os.path.join(_base, "nodes")

# Load modules
_fbx_export = _load_module("sam3d2abc_fbx_export", os.path.join(_nodes, "fbx_export.py"))
_video_proc = _load_module("sam3d2abc_video_proc", os.path.join(_nodes, "video_processor.py"))
_fbx_viewer = _load_module("sam3d2abc_fbx_viewer", os.path.join(_nodes, "fbx_viewer.py"))
_verify_overlay = _load_module("sam3d2abc_verify_overlay", os.path.join(_nodes, "verify_overlay.py"))
_camera_solver = _load_module("sam3d2abc_camera_solver", os.path.join(_nodes, "camera_solver.py"))
_motion_analyzer = _load_module("sam3d2abc_motion_analyzer", os.path.join(_nodes, "motion_analyzer.py"))
_character_trajectory = _load_module("sam3d2abc_character_trajectory", os.path.join(_nodes, "character_trajectory.py"))
_temporal_smoothing = _load_module("sam3d2abc_temporal_smoothing", os.path.join(_nodes, "temporal_smoothing.py"))

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
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_CameraDataFromJSON"] = _camera_solver.CameraDataFromJSON
    
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_CameraSolver"] = "📷 Camera Solver"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_CameraDataFromJSON"] = "📷 Camera Extrinsics from JSON"

# Register motion analyzer nodes
if _motion_analyzer:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_MotionAnalyzer"] = _motion_analyzer.MotionAnalyzer
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_ScaleInfoDisplay"] = _motion_analyzer.ScaleInfoDisplay
    
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_MotionAnalyzer"] = "📊 Motion Analyzer"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_ScaleInfoDisplay"] = "📏 Scale Info Display"

# Register character trajectory tracker
if _character_trajectory:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_CharacterTrajectoryTracker"] = _character_trajectory.CharacterTrajectoryTracker
    
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_CharacterTrajectoryTracker"] = "🏃 Character Trajectory Tracker"

# Register temporal smoothing
if _temporal_smoothing:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_TemporalSmoothing"] = _temporal_smoothing.TemporalSmoothing
    
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_TemporalSmoothing"] = "🔄 Temporal Smoothing"

# Load and register trajectory smoother
_trajectory_smoother = _load_module("sam3d2abc_trajectory_smoother", os.path.join(_nodes, "trajectory_smoother.py"))
if _trajectory_smoother:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_TrajectorySmoother"] = _trajectory_smoother.TrajectorySmoother
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_TrajectorySmoother"] = "📈 Trajectory Smoother"

# Load and register body shape lock
_body_shape_lock = _load_module("sam3d2abc_body_shape_lock", os.path.join(_nodes, "body_shape_lock.py"))
if _body_shape_lock:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_BodyShapeLock"] = _body_shape_lock.BodyShapeLockNode
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_BodyShapeLock"] = "📐 Body Shape Lock"

# Load and register pose smoothing
_pose_smoothing = _load_module("sam3d2abc_pose_smoothing", os.path.join(_nodes, "pose_smoothing.py"))
if _pose_smoothing:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_PoseSmoothing"] = _pose_smoothing.PoseSmoothingNode
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_PoseSmoothing"] = "🔄 Pose Smoothing"

# Load and register foot contact enforcer
_ground_contact = _load_module("sam3d2abc_ground_contact", os.path.join(_nodes, "ground_contact.py"))
if _ground_contact:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_FootContactEnforcer"] = _ground_contact.FootContactNode
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_FootContactEnforcer"] = "🦶 Foot Contact Enforcer"

# Load and register SLAM camera solver
_slam_path = os.path.join(_nodes, "slam")
if os.path.isdir(_slam_path):
    try:
        _slam_solver = _load_module(
            "sam3d2abc_slam_solver",
            os.path.join(_slam_path, "slam_camera_solver.py")
        )
        if _slam_solver:
            NODE_CLASS_MAPPINGS["SAM3DBody2abc_SLAMCameraSolver"] = _slam_solver.SLAMCameraSolver
            NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_SLAMCameraSolver"] = "📹 SLAM Camera Solver"
    except Exception as e:
        print(f"[SAM3DBody2abc] Error loading SLAM solver: {e}")

# Load and register multicamera nodes
_multicamera_path = os.path.join(_nodes, "multicamera")
if os.path.isdir(_multicamera_path):
    try:
        # Load auto calibrator
        _auto_calibrator = _load_module(
            "sam3d2abc_auto_calibrator",
            os.path.join(_multicamera_path, "auto_calibrator.py")
        )
        if _auto_calibrator:
            NODE_CLASS_MAPPINGS["SAM3DBody2abc_CameraAutoCalibrator"] = _auto_calibrator.CameraAutoCalibrator
            NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_CameraAutoCalibrator"] = "🎯 Camera Auto-Calibrator"
        
        # Load triangulator
        _triangulator = _load_module(
            "sam3d2abc_triangulator",
            os.path.join(_multicamera_path, "triangulator.py")
        )
        if _triangulator:
            NODE_CLASS_MAPPINGS["SAM3DBody2abc_MultiCameraTriangulator"] = _triangulator.MultiCameraTriangulator
            NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_MultiCameraTriangulator"] = "🔺 Multi-Camera Triangulator"
    except Exception as e:
        print(f"[SAM3DBody2abc] Error loading multicamera nodes: {e}")

# Print loaded nodes
print(f"[SAM3DBody2abc] v{__version__} loaded {len(NODE_CLASS_MAPPINGS)} nodes:")
for name in NODE_CLASS_MAPPINGS:
    print(f"  - {NODE_DISPLAY_NAME_MAPPINGS.get(name, name)}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Web extension directory
WEB_DIRECTORY = "./web"
