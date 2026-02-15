"""
SAM3DBody2abc - Video to Animated FBX Export

Extends SAM3DBody with video processing and animated FBX export.

Workflow (STANDALONE - No third-party wrapper needed!):
    Load Video → SAM3 Segmentation → SAM3 Extract Masks
                                            ↓
    🔧 Load SAM3DBody Model (Direct) → 🎬 Video Batch Processor ←──┘
                                              ↓
                                    📦 Export Animated FBX

Outputs match SAM3DBody Process:
- mesh_data (SAM3D_OUTPUT) → vertices, faces, joint_coords
- Uses SAM3DBodyExportFBX format for single frames
- Animated FBX has shape keys + skeleton keyframes

Version: 5.9.3
- FIXED: Kinematic Contact Detector - stationary pose detection
  - Added depth_scale_limit parameter (default 0.25) to prevent false negatives
  - Removed pelvis_moving_away requirement (failed for standing poses)
  - Gradual confidence falloff instead of hard 0.00 cutoff
  - New double support detection for both-feet-on-ground poses
  - Increased default geometry_threshold from 0.03m to 0.05m
- UPDATED: Video Batch Processor batch_size
  - Increased max from 50 to 500
  - Changed default from 10 to 50
  - Clarified tooltip: controls progress update interval, not GPU batching
- REMOVED: Unused nodes cleaned up
  - SLAM Camera Solver (nodes/slam/)
  - MegaSAM Camera Solver (megasam_solver.py)
  - Foot Contact Test (foot_contact_test.py)

Version: 5.9.2
- CHANGED: SAM3 nodes moved to separate package (ComfyUI-SAM3-Chunked)
  - Install ComfyUI-SAM3-Chunked alongside this package
  - Memory-efficient SAM3 processing for 1000+ frame videos
  - Drop-in replacement for PozzettiAndrea/ComfyUI-SAM3
  - Same node names: LoadSAM3Model, SAM3VideoSegmentation, SAM3Propagate, etc.
- NEW: 3D Joint Smoothing in Kinematic Contact Detector
  - Savitzky-Golay filter (preserves peaks/edges)
  - Butterworth filter (clean frequency cutoff, biomechanics standard)
  - One-Euro filter (adaptive, responsive to fast motion)
  - Apply before detection, after stabilization, or both
  - Chunked processing for long videos
- FIXED: Confidence-based pinning with proper formula
  - conf = 1.0 - (geom_diff / threshold) when below threshold
  - Pin at high-confidence frames (conf >= 0.95)
  - Ease-out release over N frames (quadratic curve)
- FIXED: Triangulator NameError (undefined `cameras` variable)

Version: 5.9.1
- NEW: 🦶📐 Kinematic Contact Detector node - PURE GEOMETRY APPROACH
  - Detects foot contacts using biomechanical principles, NO ML models
  - Contact = foot flat (ankle, toe, heel at same height)
  - Contact = pelvis moving away from planted foot
  - Works universally for walking, running, sprinting - any gait
  - Reprojection error provides confidence score (3D→2D consistency)
  - Configurable joint indices for different skeleton formats
  - Debug overlay shows feet, hips, torso with trajectories
  - Motion plot data output for analyzing joint movement
  - No TAPNet, no GroundLink, no velocity thresholds - just physics!

Version: 5.9.0
- NEW: 🎭 Monocular Silhouette Refiner node
  - Refine SAM3DBody pose using SAM3 segmentation masks
  - Compares rendered silhouette vs mask, optimizes to match
  - Uses skeleton hull rendering (no PyTorch3D required)
  - Improves pose accuracy by enforcing silhouette boundary
  - Works with single-camera (monocular) video
- NEW: 🦶🦿 Smart GroundLink (Video + Pinning) node - UNIFIED SOLUTION
  - Combines video-based detection + physics-based pinning
  - Uses TAPNet to track feet in video (ground truth)
  - Uses GroundLink's proven pinning to lock feet during contacts
  - No foot sliding during contact frames
  - Verification step compares pinned result vs video
  - Single node does detection + pinning together
- NEW: 🦶✨ Smart Foot Contact (Visual Feedback) node
  - Unified foot contact detection using video as ground truth
  - TAPNet 2D tracking validates and corrects SAM3DBody output
  - Reprojection error feedback loop for accurate correction
  - Contact segments detected from 2D foot velocity (not 3D guesswork)
  - Multiple temporal filters: EMA, Gaussian, Savitzky-Golay, Bidirectional EMA
  - Smooth blend in/out at contact boundaries (no snapping)
  - Memory-efficient chunked processing for long videos
  - Comprehensive logging system for debugging
- NEW: ContactSegmentDetector class for reusable segment detection
- NEW: TemporalFilter class with multiple filter options

Version: 5.8.0
- NEW: Camera Accumulator now accepts per-camera intrinsics
  - focal_length_mm input (0 = auto-detect from mesh_sequence)
  - sensor_width_mm input (default: 36mm full-frame)
- UPDATED: Auto-Calibrator uses intrinsics from CAMERA_LIST
  - Intrinsics override inputs now default to 0 (use from camera)
- UPDATED: Comprehensive documentation overhaul
  - Full version history in README
  - External dependencies and licenses (docs/Dependencies.md)
  - Updated node reference

Version: 5.7.0
- NEW: 🎭 Silhouette Refiner node
  - Refine triangulated 3D trajectory using silhouette consistency
  - Uses differentiable rendering to match projected body to SAM3 masks
  - Supports SMPL body model (PyTorch3D) or skeleton hull fallback
  - Multi-camera silhouette constraints for improved accuracy
  - Temporal smoothness and bone length preservation
  - 50-100 optimization iterations per frame for precise boundary alignment

Version: 5.6.0
- NEW: 📷 Camera Accumulator node (Option B - serial chaining)
  - Build CAMERA_LIST by chaining cameras serially
  - Supports 2 to unlimited cameras
  - Uniform interface for auto-calibration and triangulation
- UPDATED: 🎯 Camera Auto-Calibrator now accepts CAMERA_LIST
  - Pairwise calibration against reference camera
  - Supports N cameras (3, 4, 5+)
- UPDATED: 🔺 Multi-Camera Triangulator now accepts CAMERA_LIST
  - N-camera triangulation using weighted least squares
  - Better accuracy with more camera views
  - Backward-compatible quality metrics

Version: 5.5.0
- NEW: ⚡ GroundLink Physics-Based Foot Contact (PRIMARY solver)
  - Neural network predicts Ground Reaction Forces (GRF) from poses
  - More accurate than heuristics for complex movements
  - MIT License (commercial use OK)
  - Pretrained checkpoints included
  - Falls back to heuristic if needed
- NEW: 📊 GroundLink Contact Visualizer
  - Debug/visualize contact predictions

Version: 5.4.0
- NEW: 🔧 Direct SAM3DBody Integration (MAJOR CHANGE!)
  - No longer requires third-party ComfyUI-SAM3DBody wrapper
  - Load Meta's SAM-3D-Body model directly using new "Load SAM3DBody Model (Direct)" node
  - Just clone https://github.com/facebookresearch/sam-3d-body and download weights
  - Full control over model loading and coordinate system
- NEW: 🦶 Foot Tracker (TAPNet) - Visual foot tracking
- NEW: Coordinate system documentation
- FIX: Mesh-to-skeleton alignment for new SAM3DBody

Version: 5.2.0
- NEW: Added flip_ty option to FBX Export for newer SAM3DBody versions
- FIX: Improved Blender auto-detection with more search paths
- FIX: Support path-based SAM3D_MODEL format from newer ComfyUI-SAM3DBody
- NEW: 📐 Body Shape Lock node
- NEW: 🔄 Pose Smoothing node  
- NEW: 🦶 Foot Contact Enforcer node
- NEW: 📐 Body Shape Lock node
"""

__version__ = "5.9.3"

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
_load_model = _load_module("sam3d2abc_load_model", os.path.join(_nodes, "load_model.py"))
_fbx_export = _load_module("sam3d2abc_fbx_export", os.path.join(_nodes, "fbx_export.py"))
_video_proc = _load_module("sam3d2abc_video_proc", os.path.join(_nodes, "video_processor.py"))
_fbx_viewer = _load_module("sam3d2abc_fbx_viewer", os.path.join(_nodes, "fbx_viewer.py"))
_verify_overlay = _load_module("sam3d2abc_verify_overlay", os.path.join(_nodes, "verify_overlay.py"))
_camera_solver = _load_module("sam3d2abc_camera_solver", os.path.join(_nodes, "camera_solver.py"))
_motion_analyzer = _load_module("sam3d2abc_motion_analyzer", os.path.join(_nodes, "motion_analyzer.py"))
_character_trajectory = _load_module("sam3d2abc_character_trajectory", os.path.join(_nodes, "character_trajectory.py"))
_temporal_smoothing = _load_module("sam3d2abc_temporal_smoothing", os.path.join(_nodes, "temporal_smoothing.py"))

# Register model loader (NEW in v5.4.0 - Direct integration, no third-party wrapper needed!)
if _load_model:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_LoadModel"] = _load_model.LoadSAM3DBodyModel
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_LoadModel"] = "🔧 Load SAM3DBody Model (Direct)"

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

# Register camera solver v2 (TAPIR-based)
_camera_solver_v2 = _load_module("sam3d2abc_camera_solver_v2", os.path.join(_nodes, "camera_solver_v2.py"))
if _camera_solver_v2:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_CameraSolverV2"] = _camera_solver_v2.CameraSolverV2
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_CameraSolverV2"] = "📷 Camera Solver V2 (TAPIR)"

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

# Load and register multicamera nodes
_multicamera_path = os.path.join(_nodes, "multicamera")
if os.path.isdir(_multicamera_path):
    try:
        # Load camera accumulator (NEW in v5.6.0)
        _camera_accumulator = _load_module(
            "sam3d2abc_camera_accumulator",
            os.path.join(_multicamera_path, "camera_accumulator.py")
        )
        if _camera_accumulator:
            NODE_CLASS_MAPPINGS["SAM3DBody2abc_CameraAccumulator"] = _camera_accumulator.CameraAccumulator
            NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_CameraAccumulator"] = "📷 Camera Accumulator"
        
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
        
        # Load calibration loader
        _calibration_loader = _load_module(
            "sam3d2abc_calibration_loader",
            os.path.join(_multicamera_path, "calibration_loader.py")
        )
        if _calibration_loader:
            NODE_CLASS_MAPPINGS["SAM3DBody2abc_CameraCalibrationLoader"] = _calibration_loader.CameraCalibrationLoader
            NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_CameraCalibrationLoader"] = "📷 Camera Calibration Loader"
        
        # Load silhouette refiner (NEW in v5.7.0)
        _silhouette_refiner = _load_module(
            "sam3d2abc_silhouette_refiner",
            os.path.join(_multicamera_path, "silhouette_refiner.py")
        )
        if _silhouette_refiner:
            NODE_CLASS_MAPPINGS["SAM3DBody2abc_SilhouetteRefiner"] = _silhouette_refiner.SilhouetteRefiner
            NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_SilhouetteRefiner"] = "🎭 Silhouette Refiner"
    except Exception as e:
        print(f"[SAM3DBody2abc] Error loading multicamera nodes: {e}")

# Load and register BVH export
_bvh_export = _load_module("sam3d2abc_bvh_export", os.path.join(_nodes, "bvh_export.py"))
if _bvh_export:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_ExportBVH"] = _bvh_export.ExportBVH
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_ExportBVH"] = "📄 Export BVH"

# Load and register COLMAP bridge
_colmap_bridge = _load_module("sam3d2abc_colmap_bridge", os.path.join(_nodes, "colmap_bridge.py"))
if _colmap_bridge:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_COLMAPBridge"] = _colmap_bridge.COLMAPToExtrinsicsBridge
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_COLMAPBridge"] = "🔗 COLMAP to Extrinsics Bridge"

# Load and register video stabilizer
_video_stabilizer = _load_module("sam3d2abc_video_stabilizer", os.path.join(_nodes, "video_stabilizer.py"))
if _video_stabilizer:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_VideoStabilizer"] = _video_stabilizer.VideoStabilizer
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_VideoStabilizer"] = "📹 Video Stabilizer"

# Load and register intrinsics estimator
_intrinsics_estimator = _load_module("sam3d2abc_intrinsics_estimator", os.path.join(_nodes, "intrinsics_estimator.py"))
if _intrinsics_estimator:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_IntrinsicsEstimator"] = _intrinsics_estimator.IntrinsicsEstimator
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_IntrinsicsFromJSON"] = _intrinsics_estimator.IntrinsicsFromJSON
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_IntrinsicsToJSON"] = _intrinsics_estimator.IntrinsicsToJSON
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_IntrinsicsEstimator"] = "📐 Intrinsics Estimator"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_IntrinsicsFromJSON"] = "📐 Intrinsics from JSON"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_IntrinsicsToJSON"] = "📐 Intrinsics to JSON"

# Load and register intrinsics from SAM3DBody
_intrinsics_from_sam3dbody = _load_module("sam3d2abc_intrinsics_from_sam3dbody", os.path.join(_nodes, "intrinsics_from_sam3dbody.py"))
if _intrinsics_from_sam3dbody:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_IntrinsicsFromSAM3DBody"] = _intrinsics_from_sam3dbody.IntrinsicsFromSAM3DBody
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_IntrinsicsFromSAM3DBody"] = "📐 Intrinsics from SAM3DBody"

# Load and register MoGe2 intrinsics estimator
_moge_intrinsics = _load_module("sam3d2abc_moge_intrinsics", os.path.join(_nodes, "moge_intrinsics.py"))
if _moge_intrinsics:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_MoGe2Intrinsics"] = _moge_intrinsics.MoGe2IntrinsicsEstimator
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_ApplyIntrinsicsToMesh"] = _moge_intrinsics.ApplyIntrinsicsToMeshSequence
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_MoGe2Intrinsics"] = "📐 MoGe2 Intrinsics Estimator"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_ApplyIntrinsicsToMesh"] = "📐 Apply Intrinsics to Mesh"

# Load and register foot tracker (TAPNet-based)
_foot_tracker = _load_module("sam3d2abc_foot_tracker", os.path.join(_nodes, "foot_tracker.py"))
if _foot_tracker:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_FootTracker"] = _foot_tracker.FootTracker
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_FootTracker"] = "🦶 Foot Tracker (TAPNet)"

# Load and register GroundLink physics-based foot contact solver (PRIMARY)
_groundlink_solver = _load_module("sam3d2abc_groundlink_solver", os.path.join(_nodes, "groundlink_solver.py"))
if _groundlink_solver:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_GroundLinkSolver"] = _groundlink_solver.GroundLinkSolverNode
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_GroundLinkVisualizer"] = _groundlink_solver.GroundLinkContactVisualizer
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_GroundLinkSolver"] = "⚡ GroundLink Foot Contact (Physics)"

# Load and register Smart Foot Contact (Visual Feedback Loop) - RECOMMENDED
_smart_foot_contact = _load_module("sam3d2abc_smart_foot_contact", os.path.join(_nodes, "smart_foot_contact.py"))
if _smart_foot_contact:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_SmartFootContact"] = _smart_foot_contact.SmartFootContactNode
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_SmartFootContact"] = "🦶✨ Smart Foot Contact (Visual Feedback)"
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_GroundLinkVisualizer"] = "📊 GroundLink Contact Visualizer"

# Load and register Smart GroundLink (Unified Video + Pinning) - NEW in v5.9.0
_smart_groundlink = _load_module("sam3d2abc_smart_groundlink", os.path.join(_nodes, "smart_groundlink.py"))
if _smart_groundlink:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_SmartGroundLink"] = _smart_groundlink.SmartGroundLinkNode
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_SmartGroundLink"] = "🦶🦿 Smart GroundLink (Video + Pinning)"

# Load and register Monocular Silhouette Refiner - NEW in v5.9.0
_mono_silhouette = _load_module("sam3d2abc_mono_silhouette", os.path.join(_nodes, "monocular_silhouette_refiner.py"))
if _mono_silhouette:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_MonocularSilhouetteRefiner"] = _mono_silhouette.MonocularSilhouetteRefinerNode
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_MonocularSilhouetteRefiner"] = "🎭 Monocular Silhouette Refiner"

# Load and register Kinematic Contact Detector - NEW in v5.9.1
_kinematic_contact = _load_module("sam3d2abc_kinematic_contact", os.path.join(_nodes, "kinematic_contact.py"))
if _kinematic_contact:
    NODE_CLASS_MAPPINGS["SAM3DBody2abc_KinematicContact"] = _kinematic_contact.KinematicContactNode
    NODE_DISPLAY_NAME_MAPPINGS["SAM3DBody2abc_KinematicContact"] = "🦶📐 Kinematic Contact Detector"

# Print loaded nodes
print(f"[SAM3DBody2abc] v{__version__} loaded {len(NODE_CLASS_MAPPINGS)} nodes:")
for name in NODE_CLASS_MAPPINGS:
    print(f"  - {NODE_DISPLAY_NAME_MAPPINGS.get(name, name)}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Web extension directory
WEB_DIRECTORY = "./web"
