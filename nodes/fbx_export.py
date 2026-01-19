"""
Animated FBX Export for SAM3DBody2abc
Exports MESH_SEQUENCE to animated FBX using Blender.

The exported FBX contains:
- Mesh with shape keys (vertex animation per frame)
- Armature with keyframed bone rotations (from MHR) or positions

Settings:
- skeleton_mode: "Rotations" uses true joint rotations from MHR model
                 "Positions" uses joint positions (legacy)

Export Methods:
- Primary: Direct bpy module (no external Blender needed) - requires Python 3.11 + bpy
- Fallback: Blender subprocess (requires blender installation)
"""

import os
import json
import subprocess
import shutil
import glob
import tempfile
import sys
import numpy as np
import torch
from typing import Dict, Tuple, Any, Optional
import folder_paths

# Try to import bpy exporter module
BPY_AVAILABLE = False
BPY_INSTALL_HELP = """
To enable direct bpy export (no Blender subprocess needed):

1. Check your Python version: python --version
   - bpy 4.1+ requires Python 3.11.x EXACTLY
   - bpy 4.0.0 requires Python 3.10.x EXACTLY

2. Install bpy for your Python version:
   
   For Python 3.11:
     pip install bpy
   
   For Python 3.10:
     pip install bpy==4.0.0

3. Alternative: Use Blender subprocess (current method)
   Install Blender 4.2:
     cd /workspace
     wget https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz
     tar -xf blender-4.2.0-linux-x64.tar.xz
     ln -sf /workspace/blender-4.2.0-linux-x64/blender /usr/local/bin/blender
"""

try:
    from ..lib.bpy_exporter import export_animated_fbx, is_bpy_available
    BPY_AVAILABLE = is_bpy_available()
    if BPY_AVAILABLE:
        print("[FBX Export] ✓ bpy module available - direct export enabled (no Blender subprocess needed)")
    else:
        print(f"[FBX Export] bpy module imported but not functional (Python {sys.version_info.major}.{sys.version_info.minor})")
        print("[FBX Export] Will use Blender subprocess for export")
except ImportError as e:
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"[FBX Export] bpy module not available (Python {py_ver})")
    if sys.version_info >= (3, 12):
        print("[FBX Export] ⚠️  Python 3.12+ not supported by bpy - must use Blender subprocess")
    elif sys.version_info < (3, 10):
        print("[FBX Export] ⚠️  Python < 3.10 not supported by bpy - must use Blender subprocess")
    else:
        print(f"[FBX Export] Install bpy with: pip install bpy" + ("==4.0.0" if sys.version_info.minor == 10 else ""))
    print("[FBX Export] Will use Blender subprocess for export")

# Import version and logger from package
try:
    from .. import __version__
    from ..lib.logger import log, set_module, LogLevel, set_verbosity_from_string, LOG_LEVEL_CHOICES
    set_module("FBX Export")
except ImportError:
    __version__ = "unknown"
    LOG_LEVEL_CHOICES = ["Normal (Info)", "Silent", "Errors Only", "Warnings", "Verbose (Status)", "Debug (All)"]
    # Fallback logger
    class _FallbackLog:
        def info(self, msg): print(f"[FBX Export] {msg}")
        def debug(self, msg): pass
        def warn(self, msg): print(f"[FBX Export] WARN: {msg}")
        def error(self, msg): print(f"[FBX Export] ERROR: {msg}")
        def progress(self, c, t, task="", interval=10): 
            if c == 0 or c == t-1 or (c+1) % interval == 0: print(f"[FBX Export] {task}: {c+1}/{t}")
    log = _FallbackLog()
    def set_verbosity_from_string(s): pass


def to_list(obj):
    """Convert to JSON-serializable list."""
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_list(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_list(v) for v in obj]
    return obj


def get_incremental_filename(output_dir: str, filename: str, ext: str) -> str:
    """
    Get an incremental filename to avoid overwriting existing files.
    
    Args:
        output_dir: Directory for output
        filename: Base filename (without extension)
        ext: File extension (including dot, e.g., ".fbx")
    
    Returns:
        Full path with incremental number if needed
        e.g., animation.fbx, animation_0001.fbx, animation_0002.fbx
    """
    base_path = os.path.join(output_dir, f"{filename}{ext}")
    
    # If base file doesn't exist, use it
    if not os.path.exists(base_path):
        return base_path
    
    # Find next available number
    counter = 1
    while True:
        incremental_path = os.path.join(output_dir, f"{filename}_{counter:04d}{ext}")
        if not os.path.exists(incremental_path):
            return incremental_path
        counter += 1
        # Safety limit
        if counter > 9999:
            return os.path.join(output_dir, f"{filename}_{counter}{ext}")


BLENDER_TIMEOUT = 600

_current_dir = os.path.dirname(os.path.abspath(__file__))
_lib_dir = os.path.join(os.path.dirname(_current_dir), "lib")
BLENDER_SCRIPT = os.path.join(_lib_dir, "blender_animated_fbx.py")

_BLENDER_PATH = None
BLENDER_VERSION = "4.2.0"  # LTS version
BLENDER_DOWNLOAD_URL = f"https://download.blender.org/release/Blender4.2/blender-{BLENDER_VERSION}-linux-x64.tar.xz"


def find_blender() -> Optional[str]:
    """Find Blender executable."""
    global _BLENDER_PATH
    
    if _BLENDER_PATH is not None:
        return _BLENDER_PATH
    
    # Get custom_nodes directory
    try:
        custom_nodes = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    except Exception:
        custom_nodes = None
    
    locations = [
        shutil.which("blender"),
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "/opt/blender/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ]
    
    # Check SAM3DBody bundled Blender (old location)
    if custom_nodes:
        patterns = [
            os.path.join(custom_nodes, "ComfyUI-SAM3DBody", "lib", "blender", "blender-*-linux-x64", "blender"),
            os.path.join(custom_nodes, "ComfyUI-SAM3DBody", "lib", "blender", "*", "blender"),
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            locations.extend(matches)
    
    # Check SAM3DBody2abc bundled Blender (our own location)
    if custom_nodes:
        our_blender_dir = os.path.join(custom_nodes, "ComfyUI-SAM3DBody2abc", "blender")
        patterns = [
            os.path.join(our_blender_dir, "blender-*-linux-x64", "blender"),
            os.path.join(our_blender_dir, "blender"),
            os.path.join(our_blender_dir, "*", "blender"),
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            locations.extend(matches)
    
    # Check workspace locations (RunPod)
    locations.extend([
        "/workspace/blender/blender",
        "/workspace/blender-4.2.0-linux-x64/blender",
        os.path.expanduser("~/blender/blender"),
    ])
    
    # Windows
    for ver in ["4.2", "4.1", "4.0", "3.6"]:
        locations.append(f"C:\\Program Files\\Blender Foundation\\Blender {ver}\\blender.exe")
    
    for loc in locations:
        if loc and os.path.exists(loc):
            _BLENDER_PATH = loc
            log.info(f"Found Blender: {loc}")
            return loc
    
    log.info("Blender not found in standard locations")
    log.info("To install Blender, run in terminal:")
    log.info("  cd /workspace && wget https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz")
    log.info("  tar -xf blender-4.2.0-linux-x64.tar.xz && ln -s /workspace/blender-4.2.0-linux-x64/blender /usr/local/bin/blender")
    
    return None


class ExportAnimatedFBX:
    """
    Export MESH_SEQUENCE to animated FBX or Alembic.
    
    Creates export with:
    - Mesh + shape keys (FBX) or vertex cache (ABC)
    - Skeleton with rotation animation (from MHR) or position animation
    - Camera with estimated focal length
    
    Output Formats:
    - FBX: Uses blend shapes for mesh animation (may show hidden per-frame geometry in Maya)
    - ABC: Uses vertex cache for mesh animation (better for Maya, cleaner playback)
    
    Skeleton Modes:
    - Rotations (Recommended): Uses true joint rotation matrices from MHR model.
      This produces proper bone rotations for retargeting and animation editing.
    - Positions (Legacy): Uses joint positions only. Bones animate via location offset.
      Limited for retargeting but shows exact joint positions.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE",),
                "filename": ("STRING", {
                    "default": "animation",
                }),
            },
            "optional": {
                # Camera data - separate intrinsics and extrinsics
                "camera_extrinsics": ("CAMERA_EXTRINSICS", {
                    "tooltip": "Camera extrinsics (rotation + translation per frame) from Camera Solver or COLMAP Bridge."
                }),
                "camera_intrinsics": ("CAMERA_INTRINSICS", {
                    "tooltip": "Camera intrinsics (focal length) from MoGe2 Intrinsics. Takes precedence over manual sensor_width."
                }),
                
                # Timing
                "fps": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 120.0,
                    "tooltip": "FPS for animation (0 = use source fps from mesh_sequence)"
                }),
                "frame_offset": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 1000,
                    "tooltip": "Start frame in output (1 = start from frame 1 for Maya, 0 = start from frame 0)"
                }),
                
                # Format
                "output_format": (["FBX", "ABC (Alembic)"], {
                    "default": "FBX",
                    "tooltip": "FBX: blend shapes, ABC: vertex cache (better for Maya)"
                }),
                "up_axis": (["Y", "Z", "-Y", "-Z"], {
                    "default": "Y",
                    "tooltip": "Which axis points up in the output"
                }),
                
                # Skeleton
                "skeleton_mode": (["Rotations (Recommended)", "Positions (Legacy)"], {
                    "default": "Rotations (Recommended)",
                    "tooltip": "Rotations: proper bone rotations for retargeting. Positions: exact joint locations."
                }),
                
                # World translation / camera compensation
                "world_translation": ([
                    "None (Body at Origin)",
                    "Baked into Mesh/Joints",
                    "Baked into Camera",
                    "Root Locator + Camera Compensation",
                    "Root Locator",
                    "Root Locator + Animated Camera",
                    "Separate Track"
                ], {
                    "default": "None (Body at Origin)",
                    "tooltip": "How to handle world translation. 'Root Locator + Camera Compensation' bakes inverse camera extrinsics into root for stable exports."
                }),
                
                # Mesh/camera options
                "flip_x": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Mirror/flip the animation on X axis"
                }),
                "include_mesh": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include mesh with animation"
                }),
                "include_skeleton": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include skeleton/armature. Disable for camera-only export."
                }),
                "include_camera": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include camera in export"
                }),
                
                # Camera motion style (for animated camera modes)
                "camera_motion": (["Translation (Default)", "Rotation (Pan/Tilt)", "Static"], {
                    "default": "Translation (Default)",
                    "tooltip": "How camera follows character when using animated camera modes."
                }),
                
                # Smoothing (extrinsics only)
                "extrinsics_smoothing": (["Kalman Filter", "Spline Fitting", "Gaussian", "None"], {
                    "default": "Kalman Filter",
                    "tooltip": "Smoothing method for camera extrinsics. Kalman: optimal for sequential data. Spline: smooth curves."
                }),
                "smoothing_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Smoothing strength (0=minimal, 1=maximum). Only applies to extrinsics, not intrinsics."
                }),
                
                # Fallback sensor width (used if no camera_intrinsics provided)
                "sensor_width": ("FLOAT", {
                    "default": 36.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Camera sensor width in mm. Only used if camera_intrinsics not provided."
                }),
                
                # Metadata inputs for embedding in FBX
                "subject_motion": ("SUBJECT_MOTION", {
                    "tooltip": "Motion analysis data from Motion Analyzer node (for metadata embedding)"
                }),
                "scale_info": ("SCALE_INFO", {
                    "tooltip": "Scale information from Motion Analyzer node (for metadata embedding)"
                }),
                
                # Depth handling (v4.6.9)
                "use_depth_positioning": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable depth-based positioning. Required for videos where character moves toward/away from camera."
                }),
                "depth_mode": (["Position (Recommended)", "Scale Only", "Both (Position + Scale)", "Off (Legacy)"], {
                    "default": "Position (Recommended)",
                    "tooltip": "Position: character moves in Z axis (correct for 3D lighting/shadows). Scale: mesh scales with depth (2D compositing only). Both: combined."
                }),
                
                # SAM3DBody mesh alignment (v5.2.0)
                "align_mesh_to_skeleton": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Align mesh vertices to skeleton origin (pelvis). Required for new SAM3DBody where mesh uses ground-centered coordinates."
                }),
                
                # Video metadata (direct inputs - auto-filled from mesh_sequence if available)
                "source_video_fps": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 120.0,
                    "tooltip": "Source video FPS (for metadata). 0 = use fps from mesh_sequence."
                }),
                "skip_first_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "tooltip": "Number of frames skipped from start of video (for metadata)."
                }),
                
                "output_dir": ("STRING", {
                    "default": "",
                }),
                
                # Blender path (if auto-detection fails)
                "blender_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to Blender executable. Leave empty for auto-detection."
                }),
                
                # Logging verbosity
                "log_level": (["Normal (Info)", "Silent", "Errors Only", "Warnings", "Verbose (Status)", "Debug (All)"], {
                    "default": "Normal (Info)",
                    "tooltip": "Console output verbosity. Normal shows key status messages. Debug shows all diagnostics."
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("output_path", "status", "frame_count", "fps")
    FUNCTION = "export_fbx"
    CATEGORY = "SAM3DBody2abc/Export"
    OUTPUT_NODE = True
    
    def _build_metadata(
        self,
        world_translation: str,
        camera_motion: str,
        subject_motion: Optional[Dict],
        scale_info: Optional[Dict],
        source_video_fps: float,
        skip_first_frames: int,
        fps: float,
        frame_count: int,
        camera_extrinsics: Optional[Dict] = None,
        # New in v4.8.8 - export settings
        extrinsics_smoothing: str = "",
        smoothing_strength: float = 0.0,
        use_depth_positioning: bool = True,
        depth_mode: str = "",
        skeleton_mode: str = "",
        up_axis: str = "Y",
        flip_x: bool = False,
        include_mesh: bool = True,
        include_skeleton: bool = True,
        include_camera: bool = True,
        # Filename info
        filename: str = "",
        output_path: str = "",
    ) -> Dict:
        """
        Build metadata dict to be embedded in FBX as custom properties.
        
        This creates a SAM3DBody_Metadata locator in the FBX with all
        analysis data accessible as Extra Attributes in Maya.
        """
        from datetime import datetime, timezone, timedelta
        
        # Get version - use module-level import first, then file read as fallback
        # Note: module-level __version__ is imported at top of file
        version_str = __version__  # From module-level import
        if version_str == "unknown":
            try:
                import os
                init_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "__init__.py")
                if os.path.exists(init_path):
                    with open(init_path, "r") as f:
                        for line in f:
                            if line.strip().startswith("__version__"):
                                # Parse: __version__ = "4.5.8"
                                version_str = line.split("=")[1].strip().strip('"\'').strip()
                                break
            except Exception as e:
                log.info(f"Warning: Could not read version from file: {e}")
        
        # Get timestamp in IST
        ist = timezone(timedelta(hours=5, minutes=30))
        timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
        
        metadata = {
            # Build info
            "sam3dbody2abc_version": version_str,
            "export_timestamp": timestamp,
            # File info
            "filename": filename,
            "output_path": output_path,
            # Export settings
            "world_translation": world_translation,
            "camera_motion": camera_motion,
            "export_fps": fps,
            "frame_count": frame_count,
            # Additional export settings (v4.8.8)
            "extrinsics_smoothing": extrinsics_smoothing,
            "extrinsics_smoothing_strength": smoothing_strength,
            "depth_positioning_enabled": str(use_depth_positioning),
            "depth_mode": depth_mode,
            "skeleton_export_mode": skeleton_mode,
            "up_axis": up_axis,
            "flip_x": str(flip_x),
            "include_mesh": str(include_mesh),
            "include_skeleton": str(include_skeleton),
            "include_camera": str(include_camera),
        }
        
        # Video info - use source_video_fps if provided, otherwise use export fps
        if source_video_fps > 0:
            metadata["video_fps"] = source_video_fps
        else:
            metadata["video_fps"] = fps  # Fall back to export fps
        
        if skip_first_frames > 0:
            metadata["skip_first_frames"] = skip_first_frames
        
        # Scale info (from Motion Analyzer)
        if scale_info:
            metadata["subject_height_m"] = scale_info.get("actual_height_m", 0)  # Fixed key name
            metadata["mesh_height_units"] = scale_info.get("mesh_height", 0)
            metadata["estimated_height_units"] = scale_info.get("estimated_height", 0)
            metadata["scale_factor"] = scale_info.get("scale_factor", 1.0)
            metadata["leg_length_units"] = scale_info.get("leg_length", 0)
            metadata["torso_head_units"] = scale_info.get("torso_head_length", 0)
            metadata["height_source"] = scale_info.get("height_source", "auto")
            metadata["skeleton_mode"] = scale_info.get("skeleton_mode", "")
            metadata["keypoint_source"] = scale_info.get("keypoint_source", "")
            metadata["reference_frame"] = scale_info.get("reference_frame", 0)
        
        # Motion analysis results (from Motion Analyzer)
        if subject_motion:
            metadata["motion_num_frames"] = subject_motion.get("num_frames", 0)
            metadata["motion_scale_factor"] = subject_motion.get("scale_factor", 1.0)
            metadata["motion_skeleton_mode"] = subject_motion.get("skeleton_mode", "")
            metadata["motion_keypoint_source"] = subject_motion.get("keypoint_source", "")
            
            # Foot contact statistics
            foot_contact = subject_motion.get("foot_contact", [])
            if foot_contact:
                grounded = sum(1 for fc in foot_contact if fc in ["both", "left", "right"])
                metadata["grounded_frames"] = grounded
                metadata["airborne_frames"] = len(foot_contact) - grounded
                metadata["grounded_percent"] = round(100.0 * grounded / len(foot_contact), 1)
            
            # Depth range
            depth_estimates = subject_motion.get("depth_estimate", [])
            if depth_estimates:
                metadata["depth_min_m"] = round(min(depth_estimates), 3)
                metadata["depth_max_m"] = round(max(depth_estimates), 3)
            
            # Body world trajectory (always available - computed from pred_cam_t)
            body_world_3d = subject_motion.get("body_world_3d", [])
            if body_world_3d:
                import numpy as np
                bw_arr = np.array(body_world_3d)
                displacement = bw_arr[-1] - bw_arr[0]
                metadata["body_world_disp_x"] = round(float(displacement[0]), 4)
                metadata["body_world_disp_y"] = round(float(displacement[1]), 4)
                metadata["body_world_disp_z"] = round(float(displacement[2]), 4)
                
                # Total distance
                if len(bw_arr) > 1:
                    velocities = np.diff(bw_arr, axis=0)
                    total_dist = np.sum(np.linalg.norm(velocities, axis=1))
                    metadata["body_world_total_distance"] = round(float(total_dist), 4)
                    
                    # Average speed (per frame, then per second)
                    avg_speed_per_frame = total_dist / (len(bw_arr) - 1)
                    metadata["body_world_avg_speed_per_frame"] = round(float(avg_speed_per_frame), 4)
            
            # New trajectory metrics (RAW - includes camera effects)
            metadata["total_distance_m_raw"] = round(subject_motion.get("total_distance_m", 0), 4)
            metadata["avg_speed_ms_raw"] = round(subject_motion.get("avg_speed_ms", 0), 4)
            metadata["avg_speed_kmh_raw"] = round(subject_motion.get("avg_speed_ms", 0) * 3.6, 2)
            metadata["direction_angle_raw"] = round(subject_motion.get("direction_angle", 0), 1)
            metadata["direction_desc_raw"] = subject_motion.get("direction_desc", "Unknown")
            metadata["duration_sec"] = round(subject_motion.get("duration_sec", 0), 2)
            
            # Direction vector RAW (normalized, Y-up coordinate system)
            direction_vector = subject_motion.get("direction_vector", [0, 0, 0])
            if direction_vector:
                metadata["direction_vector_x_raw"] = round(direction_vector[0], 4)
                metadata["direction_vector_y_raw"] = round(direction_vector[1], 4)
                metadata["direction_vector_z_raw"] = round(direction_vector[2], 4)
            
            # Compensated trajectory metrics (camera effects removed)
            metadata["total_distance_m_comp"] = round(subject_motion.get("total_distance_m_compensated", 0), 4)
            metadata["avg_speed_ms_comp"] = round(subject_motion.get("avg_speed_ms_compensated", 0), 4)
            metadata["avg_speed_kmh_comp"] = round(subject_motion.get("avg_speed_ms_compensated", 0) * 3.6, 2)
            metadata["direction_angle_comp"] = round(subject_motion.get("direction_angle_compensated", 0), 1)
            metadata["direction_desc_comp"] = subject_motion.get("direction_desc_compensated", "Unknown")
            
            # Direction vector COMPENSATED (normalized, Y-up coordinate system)
            direction_vector_comp = subject_motion.get("direction_vector_compensated", [0, 0, 0])
            if direction_vector_comp:
                metadata["direction_vector_x_comp"] = round(direction_vector_comp[0], 4)
                metadata["direction_vector_y_comp"] = round(direction_vector_comp[1], 4)
                metadata["direction_vector_z_comp"] = round(direction_vector_comp[2], 4)
            
            # Focal length info used for compensation
            metadata["focal_length_ref_px"] = round(subject_motion.get("focal_length_ref_px", 0), 1)
            metadata["focal_length_ref_mm"] = round(subject_motion.get("focal_length_ref_mm", 0), 1)
            metadata["focal_length_min_mm"] = round(subject_motion.get("focal_length_min_mm", 0), 1)
            metadata["focal_length_max_mm"] = round(subject_motion.get("focal_length_max_mm", 0), 1)
            metadata["focal_variation_percent"] = round(subject_motion.get("focal_variation_percent", 0), 1)
            metadata["sensor_width_mm"] = round(subject_motion.get("sensor_width_mm", 36.0), 1)
            metadata["has_extrinsics_compensation"] = str(subject_motion.get("has_extrinsics_compensation", False))
            
            # === Motion Analyzer settings ===
            metadata["depth_source"] = subject_motion.get("depth_source", "Auto")
            
            # === Trajectory Smoother settings (if applied) ===
            smoothing_info = subject_motion.get("smoothing_applied", {})
            if smoothing_info:
                metadata["trajectory_smoothing_method"] = smoothing_info.get("method", "None")
                metadata["trajectory_smoothing_strength"] = str(smoothing_info.get("strength", 0))
                metadata["trajectory_jitter_reduction_pct"] = str(round(smoothing_info.get("jitter_reduction_pct", 0), 1))
                metadata["trajectory_reference_joint"] = smoothing_info.get("reference_joint_name", "")
                metadata["trajectory_skeleton_format"] = smoothing_info.get("skeleton_format", "")
            
            # === Character Trajectory settings (if available) ===
            char_traj_info = subject_motion.get("character_trajectory_settings", {})
            if char_traj_info:
                metadata["char_traj_tracking_mode"] = char_traj_info.get("tracking_mode", "")
                metadata["char_traj_smoothing_method"] = char_traj_info.get("smoothing_method", "")
                metadata["char_traj_smoothing_window"] = str(char_traj_info.get("smoothing_window", 0))
            
            # === Camera Solver settings (if available from extrinsics) ===
            if camera_extrinsics:
                metadata["camera_solving_method"] = camera_extrinsics.get("solving_method", "")
                metadata["camera_translational_solver"] = camera_extrinsics.get("translational_solver", "")
                metadata["camera_coordinate_system"] = camera_extrinsics.get("coordinate_system", "")
        
        # Add detailed joint indices reference for Maya users
        # Full Skeleton (127 joints) - SMPL-X based
        full_skeleton_joints = (
            "0=Pelvis, 1=L_Hip, 2=R_Hip, 3=Spine1, 4=L_Knee, 5=R_Knee, "
            "6=Spine2, 7=L_Ankle, 8=R_Ankle, 9=Spine3, 10=L_Foot, 11=R_Foot, "
            "12=Neck, 13=L_Collar, 14=R_Collar, 15=Head, 16=L_Shoulder, 17=R_Shoulder, "
            "18=L_Elbow, 19=R_Elbow, 20=L_Wrist, 21=R_Wrist, "
            "22-36=L_Hand (15 joints), 37-51=R_Hand (15 joints), "
            "52-67=Face/Jaw, 68-126=Additional body"
        )
        simple_skeleton_joints = (
            "0=Nose, 1=L_Eye, 2=R_Eye, 3=L_Ear, 4=R_Ear, "
            "5=L_Shoulder, 6=R_Shoulder, 7=L_Elbow, 8=R_Elbow, "
            "9=L_Wrist, 10=R_Wrist, 11=L_Hip, 12=R_Hip, "
            "13=L_Knee, 14=R_Knee, 15=L_Ankle, 16=R_Ankle"
        )
        metadata["full_skeleton_joints"] = full_skeleton_joints
        metadata["simple_skeleton_joints"] = simple_skeleton_joints
        
        # Add trajectory interpretation note
        metadata["trajectory_note"] = (
            "RAW = motion relative to camera (includes camera pan/zoom effects). "
            "COMP = estimated actual world motion (camera effects removed). "
            "Use COMP for real subject velocity/direction."
        )
        
        # Convert ALL values to strings to avoid sliders in Maya
        string_metadata = {}
        for key, value in metadata.items():
            string_metadata[key] = str(value)
        
        return string_metadata
    
    def export_fbx(
        self,
        mesh_sequence: Dict,
        filename: str = "animation",
        camera_extrinsics: Optional[Dict] = None,
        camera_intrinsics: Optional[Dict] = None,
        fps: float = 0.0,
        frame_offset: int = 1,
        output_format: str = "FBX",
        up_axis: str = "Y",
        skeleton_mode: str = "Rotations (Recommended)",
        world_translation: str = "None (Body at Origin)",
        flip_x: bool = False,
        include_mesh: bool = True,
        include_skeleton: bool = True,
        include_camera: bool = True,
        camera_motion: str = "Translation (Default)",
        extrinsics_smoothing: str = "Kalman Filter",
        smoothing_strength: float = 0.5,
        sensor_width: float = 36.0,
        subject_motion: Optional[Dict] = None,
        scale_info: Optional[Dict] = None,
        use_depth_positioning: bool = True,
        depth_mode: str = "Position (Z-axis)",
        align_mesh_to_skeleton: bool = True,
        source_video_fps: float = 0.0,
        skip_first_frames: int = 0,
        output_dir: str = "",
        blender_path: str = "",
        log_level: str = "Normal (Info)",
    ) -> Tuple[str, str, int, float]:
        """Export to animated FBX or Alembic."""
        
        # EARLY DIAGNOSTIC - print immediately
        print(f"[FBX Export] ============ STARTING EXPORT ============")
        print(f"[FBX Export] mesh_sequence type: {type(mesh_sequence)}")
        if mesh_sequence is None:
            print(f"[FBX Export] ERROR: mesh_sequence is None!")
            return ("", "Error: mesh_sequence is None", 0, 24.0)
        if isinstance(mesh_sequence, dict):
            print(f"[FBX Export] mesh_sequence keys: {list(mesh_sequence.keys())}")
            frames = mesh_sequence.get("frames", {})
            print(f"[FBX Export] frames type: {type(frames)}, count: {len(frames) if frames else 0}")
        else:
            print(f"[FBX Export] ERROR: mesh_sequence is not a dict!")
            return ("", f"Error: mesh_sequence is {type(mesh_sequence)}, expected dict", 0, 24.0)
        
        # Set logging verbosity from node parameter
        set_verbosity_from_string(log_level)
        
        # Log version at start of export
        log.info(f"SAM3DBody2abc version: {__version__}")
        
        # Get fps from mesh_sequence if not specified (0 means use source)
        if fps <= 0:
            fps = mesh_sequence.get("fps", 24.0)
            log.info(f"Using fps from source: {fps}")
        
        # Get intrinsics from camera_intrinsics if provided (MoGe2 takes precedence)
        if camera_intrinsics is not None:
            intrinsics_focal_px = camera_intrinsics.get("focal_length_px", None)
            intrinsics_sensor_mm = camera_intrinsics.get("sensor_width_mm", sensor_width)
            intrinsics_source = camera_intrinsics.get("source", "unknown")
            intrinsics_cx = camera_intrinsics.get("principal_point_x", None)
            intrinsics_cy = camera_intrinsics.get("principal_point_y", None)
            intrinsics_w = camera_intrinsics.get("image_width", None)
            intrinsics_h = camera_intrinsics.get("image_height", None)
            
            if intrinsics_focal_px:
                log.info(f"Using intrinsics from {intrinsics_source}: focal={intrinsics_focal_px:.1f}px")
            
            # Log principal point (cx, cy)
            if intrinsics_cx is not None and intrinsics_cy is not None:
                log.info(f"Principal point: cx={intrinsics_cx:.2f}px, cy={intrinsics_cy:.2f}px")
                if intrinsics_w and intrinsics_h:
                    # Calculate offset from image center
                    center_x = intrinsics_w / 2.0
                    center_y = intrinsics_h / 2.0
                    offset_x = intrinsics_cx - center_x
                    offset_y = intrinsics_cy - center_y
                    log.info(f"Image center: ({center_x:.1f}, {center_y:.1f})")
                    log.info(f"Principal point offset: dx={offset_x:.2f}px, dy={offset_y:.2f}px")
                    
                    # Calculate film offset (normalized)
                    # Film offset = offset from center / image dimension
                    film_offset_x = offset_x / intrinsics_w
                    film_offset_y = offset_y / intrinsics_h
                    log.info(f"Film offset (normalized): X={film_offset_x:.4f}, Y={film_offset_y:.4f}")
        else:
            intrinsics_focal_px = None
            intrinsics_sensor_mm = sensor_width
            intrinsics_source = "manual"
        
        # Check if we have camera extrinsics
        has_extrinsics = camera_extrinsics is not None and "rotations" in camera_extrinsics
        if has_extrinsics:
            extrinsics_source = camera_extrinsics.get("source", "unknown")
            log.info(f"Using camera extrinsics from {extrinsics_source} ({len(camera_extrinsics['rotations'])} frames)")
        
        # Determine camera behavior based on world_translation mode
        use_camera_rotation = ("Rotation" in camera_motion)
        camera_static = (camera_motion == "Static")
        animate_camera = (world_translation == "Baked into Camera")
        camera_follow_root = ("Root Locator + Animated Camera" in world_translation)
        camera_compensation = ("Camera Compensation" in world_translation)
        
        # Camera compensation mode: bake inverse extrinsics into root locator
        if camera_compensation and not has_extrinsics:
            log.info(" Warning: 'Root Locator + Camera Compensation' requires camera_extrinsics input. Falling back to 'Root Locator'.")
            world_translation = "Root Locator"
            camera_compensation = False
        
        if include_camera:
            if camera_compensation:
                log.info(f"Camera Compensation mode: inverse extrinsics baked into root, static camera exported")
            elif animate_camera:
                mode_str = "rotation (pan/tilt)" if use_camera_rotation else "translation"
                log.info(f"Camera animated with {mode_str}")
            elif camera_follow_root:
                mode_str = "rotation (pan/tilt)" if use_camera_rotation else "translation"
                log.info(f"Camera follows root with {mode_str}")
            elif not camera_compensation:
                log.info(f"Camera will be static")
        
        frames = mesh_sequence.get("frames", {})
        log.info(f"mesh_sequence keys: {list(mesh_sequence.keys())}")
        log.info(f"frames type: {type(frames)}, count: {len(frames) if frames else 0}")
        
        if not frames:
            log.info("ERROR: No frames found in mesh_sequence!")
            return ("", "Error: No frames in mesh_sequence", 0, fps)
        
        # Find Blender - use custom path if provided
        if blender_path and os.path.exists(blender_path):
            blender_exe = blender_path
            log.info(f"Using custom Blender path: {blender_exe}")
        else:
            blender_exe = find_blender()
        
        log.info(f"Blender path: {blender_exe}")
        if not blender_exe and not BPY_AVAILABLE:
            python_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            
            # Python 3.12+ specific message
            if sys.version_info >= (3, 12):
                bpy_note = f"""
  ⚠️  Your Python {python_ver} is NOT supported by bpy module.
  bpy only works with Python 3.10.x or 3.11.x (exact version match required).
  
  → You MUST use Blender subprocess method (Option 2 below)."""
            else:
                bpy_note = f"""
  For Python 3.11.x: pip install bpy
  For Python 3.10.x: pip install bpy==4.0.0
  
  Note: bpy requires EXACT Python version match."""
            
            error_msg = f"""Error: No export method available!

OPTION 1: Install bpy Python module (faster, no Blender needed)
  Your Python: {python_ver}
  {bpy_note}

OPTION 2: Install Blender (works with any Python version):
  cd /workspace
  wget https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz
  tar -xf blender-4.2.0-linux-x64.tar.xz
  ln -sf /workspace/blender-4.2.0-linux-x64/blender /usr/local/bin/blender

OPTION 3: Specify Blender path in the blender_path input."""
            log.info(error_msg)
            return ("", "Error: No export method available. See console for installation instructions.", 0, fps)
        
        # If bpy is not available but blender_exe is also not found, we already returned above
        # If blender_exe is None but BPY_AVAILABLE is True, we'll use bpy (handled later)
        
        if not os.path.exists(BLENDER_SCRIPT) and not BPY_AVAILABLE:
            log.info(f"Script not found: {BLENDER_SCRIPT}")
            return ("", f"Error: Script not found: {BLENDER_SCRIPT}", 0, fps)
        
        if not output_dir:
            output_dir = folder_paths.get_output_directory()
        log.info(f"Output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        sorted_indices = sorted(frames.keys())
        
        # Determine output extension
        use_alembic = "ABC" in output_format
        ext = ".abc" if use_alembic else ".fbx"
        
        # Map world_translation option to mode
        translation_mode = "none"
        if "Camera Compensation" in world_translation:
            translation_mode = "root_camera_compensation"
        elif "Baked into Geometry" in world_translation:
            translation_mode = "bake_to_geometry"
        elif "Baked into Mesh" in world_translation:
            translation_mode = "baked"
        elif "Baked into Camera" in world_translation:
            translation_mode = "camera"
        elif "Root" in world_translation:
            translation_mode = "root"
        elif "Separate" in world_translation:
            translation_mode = "separate"
        
        # Map skeleton_mode option
        use_rotations = "Rotations" in skeleton_mode
        
        # Check if rotation data is available (handle numpy arrays properly)
        first_frame = frames[sorted_indices[0]]
        joint_rots = first_frame.get("joint_rotations")
        
        # Debug: print what data we have
        log.info(f"First frame keys: {list(first_frame.keys())}")
        log.info(f"joint_rotations type: {type(joint_rots)}")
        if joint_rots is not None:
            if isinstance(joint_rots, np.ndarray):
                log.info(f"joint_rotations shape: {joint_rots.shape}, size: {joint_rots.size}")
            elif isinstance(joint_rots, list):
                log.info(f"joint_rotations is list with {len(joint_rots)} elements")
        
        has_rotations = joint_rots is not None
        if has_rotations and isinstance(joint_rots, np.ndarray):
            has_rotations = joint_rots.size > 0
        elif has_rotations and isinstance(joint_rots, list):
            has_rotations = len(joint_rots) > 0
        
        if use_rotations and not has_rotations:
            log.info(" Warning: Rotation mode requested but no rotation data available. Falling back to positions.")
            log.info(" Note: Make sure you're using a recent version of ComfyUI-SAM3DBody that outputs joint_rotations.")
            use_rotations = False
        else:
            log.info(f"Rotation data available: {has_rotations}, using rotations: {use_rotations}")
        
        # Build JSON for Blender
        joint_parents = mesh_sequence.get("joint_parents")
        log.info(f"joint_parents in mesh_sequence: {joint_parents is not None}")
        if joint_parents is not None:
            if hasattr(joint_parents, 'shape'):
                log.info(f"joint_parents shape: {joint_parents.shape}")
            elif isinstance(joint_parents, list):
                log.info(f"joint_parents length: {len(joint_parents)}")
        
        log.info(f"Camera motion mode: {'Static' if camera_static else ('Rotation (Pan/Tilt)' if use_camera_rotation else 'Translation')}")
        
        # Prepare camera extrinsics for Blender
        solved_rotations = None
        if has_extrinsics:
            solved_rotations = []
            has_translation = camera_extrinsics.get("has_translation", False)
            for rot_data in camera_extrinsics["rotations"]:
                rot_entry = {
                    "frame": rot_data.get("frame", 0),
                    "pan": rot_data.get("pan", 0.0),
                    "tilt": rot_data.get("tilt", 0.0),
                    "roll": rot_data.get("roll", 0.0),
                    "tx": rot_data.get("tx", 0.0),
                    "ty": rot_data.get("ty", 0.0),
                    "tz": rot_data.get("tz", 0.0),
                }
                solved_rotations.append(rot_entry)
            
            final_rot = solved_rotations[-1]
            log.info(f"Camera extrinsics: {len(solved_rotations)} frames")
            log.info(f"  Final: pan={np.degrees(final_rot['pan']):.2f}°, tilt={np.degrees(final_rot['tilt']):.2f}°")
            if has_translation:
                log.info(f"  Translation present: tx={final_rot['tx']:.4f}, ty={final_rot['ty']:.4f}, tz={final_rot['tz']:.4f}")
        
        # Map smoothing method to internal name
        smoothing_method_map = {
            "Kalman Filter": "kalman",
            "Spline Fitting": "spline",
            "Gaussian": "gaussian",
            "None": "none",
        }
        smoothing_method = smoothing_method_map.get(extrinsics_smoothing, "kalman")
        
        # Prepare intrinsics data for export
        intrinsics_export = None
        if camera_intrinsics is not None:
            intrinsics_export = {
                "focal_length_px": camera_intrinsics.get("focal_length_px"),
                "focal_length_mm": camera_intrinsics.get("focal_length_mm"),
                "sensor_width_mm": camera_intrinsics.get("sensor_width_mm", sensor_width),
                "principal_point_x": camera_intrinsics.get("principal_point_x"),
                "principal_point_y": camera_intrinsics.get("principal_point_y"),
                "image_width": camera_intrinsics.get("image_width"),
                "image_height": camera_intrinsics.get("image_height"),
                "source": camera_intrinsics.get("source", "MoGe2"),
            }
        
        # Determine output path early so we can include it in metadata
        output_path = get_incremental_filename(output_dir, filename, ext)
        
        export_data = {
            "fps": fps,
            "frame_count": len(sorted_indices),
            "frame_offset": frame_offset,
            "faces": to_list(mesh_sequence.get("faces")),
            "joint_parents": to_list(joint_parents),
            "sensor_width": sensor_width,
            "world_translation_mode": translation_mode,
            "skeleton_mode": "rotations" if use_rotations else "positions",
            "flip_x": flip_x,
            "align_mesh_to_skeleton": align_mesh_to_skeleton,  # v5.2.0: Align mesh to skeleton origin
            "include_skeleton": include_skeleton,  # v4.6.10: Option to exclude skeleton
            "animate_camera": animate_camera,
            "camera_follow_root": camera_follow_root,
            "camera_use_rotation": use_camera_rotation,
            "camera_static": camera_static,
            "camera_compensation": camera_compensation,  # NEW: bake inverse extrinsics to root
            "camera_extrinsics": solved_rotations,  # From Camera Solver / COLMAP Bridge / JSON
            "camera_intrinsics": intrinsics_export,  # From MoGe2 Intrinsics
            "extrinsics_smoothing_method": smoothing_method,
            "extrinsics_smoothing_strength": smoothing_strength,
            # Depth positioning (v4.6.9 fix, v4.8.8 Position is now default)
            "use_depth_positioning": use_depth_positioning,
            "depth_mode": {
                "Position (Recommended)": "position",
                "Scale Only": "scale",
                "Both (Position + Scale)": "both",
                "Off (Legacy)": "off",
                # Backwards compatibility with old option names
                "Scale (Recommended)": "scale",
                "Position (Z Movement)": "position",
                "Both (Scale + Z)": "both",
            }.get(depth_mode, "position"),  # Default to position now
            # Scale factor for consistent world coordinates (v4.8.8)
            "scale_factor": scale_info.get("scale_factor", 1.0) if scale_info else (subject_motion.get("scale_factor", 1.0) if subject_motion else 1.0),
            "frames": [],
            # Body world trajectory for animated locator (COMPENSATED - camera effects removed)
            # Uses body_world_3d_compensated if available, falls back to raw body_world_3d
            "body_world_trajectory": (
                subject_motion.get("body_world_3d_compensated") or 
                subject_motion.get("body_world_3d", [])
            ) if subject_motion else [],
            # Explicitly pass compensated trajectory for WorldPosition locator (v4.6.7)
            "body_world_trajectory_compensated": subject_motion.get("body_world_3d_compensated", []) if subject_motion else [],
            # Also include raw trajectory for reference
            "body_world_trajectory_raw": subject_motion.get("body_world_3d_raw", []) if subject_motion else [],
            # Metadata for embedding in FBX
            "metadata": self._build_metadata(
                world_translation=world_translation,
                camera_motion=camera_motion,
                subject_motion=subject_motion,
                scale_info=scale_info,
                source_video_fps=source_video_fps,
                skip_first_frames=skip_first_frames,
                fps=fps,
                frame_count=len(sorted_indices),
                camera_extrinsics=camera_extrinsics,
                # New in v4.8.8 - export settings
                extrinsics_smoothing=extrinsics_smoothing,
                smoothing_strength=smoothing_strength,
                use_depth_positioning=use_depth_positioning,
                depth_mode=depth_mode,
                skeleton_mode=skeleton_mode,
                up_axis=up_axis,
                flip_x=flip_x,
                include_mesh=include_mesh,
                include_skeleton=include_skeleton,
                include_camera=include_camera,
                # Filename info
                filename=filename,
                output_path=output_path,
            ),
        }
        
        # DEBUG: Show root_locator and body_offset calculation for frame 0
        first_frame = frames[sorted_indices[0]]
        first_cam_t = first_frame.get("pred_cam_t")
        first_focal = first_frame.get("focal_length")
        first_image_size = first_frame.get("image_size")
        if first_cam_t is not None:
            first_cam_t = to_list(first_cam_t)
            if len(first_cam_t) >= 3:
                tx, ty, tz = first_cam_t[0], first_cam_t[1], first_cam_t[2]
                
                # What screen position does this correspond to?
                if first_focal and first_image_size:
                    focal = float(first_focal) if not isinstance(first_focal, (list, tuple)) else float(first_focal[0])
                    img_w, img_h = first_image_size[0], first_image_size[1]
                    cx, cy = img_w / 2, img_h / 2
                    # Screen position from pred_cam_t
                    screen_x = focal * tx / tz + cx
                    screen_y = focal * ty / tz + cy  # NO negation - SAM3DBody coords are image-aligned
        
        for idx in sorted_indices:
            frame = frames[idx]
            
            # Handle pred_cam_t vs camera field (can't use 'or' with numpy arrays)
            pred_cam_t = frame.get("pred_cam_t")
            if pred_cam_t is None:
                pred_cam_t = frame.get("camera")
            
            frame_data = {
                "frame_index": idx,
                "joint_coords": to_list(frame.get("joint_coords")),
                "pred_cam_t": to_list(pred_cam_t),
                "focal_length": frame.get("focal_length"),
                "bbox": to_list(frame.get("bbox")),  # For camera alignment
                "image_size": frame.get("image_size"),  # (width, height)
                "keypoints_2d": to_list(frame.get("keypoints_2d")),  # For apparent height
                "keypoints_3d": to_list(frame.get("keypoints_3d")),  # 18-joint 3D keypoints
                # Tracked depth from CharacterTrajectoryTracker (v4.6.9)
                "tracked_depth": frame.get("tracked_depth"),
                "tracked_position_2d": to_list(frame.get("tracked_position_2d")),
                "tracked_position_3d": to_list(frame.get("tracked_position_3d")),
            }
            if include_mesh:
                frame_data["vertices"] = to_list(frame.get("vertices"))
            if use_rotations:
                frame_data["joint_rotations"] = to_list(frame.get("joint_rotations"))
            export_data["frames"].append(frame_data)
        
        format_name = "Alembic" if use_alembic else "FBX"
        skel_mode_str = "rotations" if use_rotations else "positions"
        log.info(f"Exporting {len(sorted_indices)} frames as {format_name}")
        log.info(f"Settings: up={up_axis}, translation={translation_mode}, skeleton={skel_mode_str}, camera={include_camera}")
        log.info(f"Output path: {output_path}")
        
        # =====================================================================
        # EXPORT METHOD SELECTION
        # Try bpy module first (no external Blender needed), fall back to subprocess
        # =====================================================================
        
        if BPY_AVAILABLE:
            # PRIMARY METHOD: Direct bpy export (faster, no subprocess overhead)
            log.info("Using direct bpy export (no Blender subprocess needed)")
            try:
                result = export_animated_fbx(
                    export_data=export_data,
                    output_path=output_path,
                    up_axis=up_axis,
                    include_mesh=include_mesh,
                    include_camera=include_camera
                )
                
                if result.get("status") == "success":
                    file_size_mb = result.get("file_size_mb", 0)
                    
                    status = f"Exported {len(sorted_indices)} frames as {format_name} (up={up_axis}, skeleton={skel_mode_str})"
                    if not include_mesh:
                        status += " skeleton only"
                    if include_camera:
                        status += " +camera"
                    status += " [bpy direct]"
                    
                    log.info("=" * 60)
                    log.info(f"SUCCESS!")
                    log.info(f"{status}")
                    log.info(f"File: {output_path}")
                    log.info(f"Size: {file_size_mb:.2f} MB")
                    log.info("=" * 60)
                    
                    return (output_path, status, len(sorted_indices), fps)
                else:
                    error_msg = result.get("message", "Unknown error")
                    log.warn(f"bpy export failed: {error_msg}")
                    log.info("Falling back to Blender subprocess...")
                    # Fall through to subprocess method
                    
            except Exception as e:
                log.warn(f"bpy export exception: {str(e)}")
                log.info("Falling back to Blender subprocess...")
                # Fall through to subprocess method
        
        # FALLBACK METHOD: Blender subprocess (requires external Blender installation)
        log.info("Using Blender subprocess export")
        
        # Write temp JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(export_data, f)
            json_path = f.name
        
        try:
            cmd = [
                blender_exe,
                "--background",
                "--python", BLENDER_SCRIPT,
                "--",
                json_path,
                output_path,
                up_axis,
                "1" if include_mesh else "0",
                "1" if include_camera else "0",
            ]
            
            # Map log_level to environment variable for Blender subprocess
            level_to_env = {
                "Silent": "SILENT",
                "Errors Only": "ERROR",
                "Warnings": "WARN",
                "Normal (Info)": "INFO",
                "Verbose (Status)": "STATUS",
                "Debug (All)": "DEBUG",
            }
            env = os.environ.copy()
            env["SAM3DBODY_LOG_LEVEL"] = level_to_env.get(log_level, "INFO")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=BLENDER_TIMEOUT,
                env=env,
            )
            
            # Log Blender output
            if result.stdout:
                lines = result.stdout.split('\n')
                
                # First, print all DEBUG lines (important for troubleshooting)
                debug_lines = [l for l in lines if 'DEBUG' in l and l.strip()]
                if debug_lines:
                    log.debug(f"=== DEBUG OUTPUT ({len(debug_lines)} lines) ===")
                    for line in debug_lines:
                        log.debug(f"{line}")
                    log.debug(f"=== END DEBUG ===")
                
                # Then print last 30 lines of regular output
                for line in lines[-30:]:
                    if line.strip() and 'DEBUG' not in line:
                        log.debug(f"{line}")
            
            if result.returncode != 0:
                error = result.stderr[:500] if result.stderr else "Unknown error"
                log.info(f"Blender error: {error}")
                return ("", f"Blender error: {error}", 0, fps)
            
            if not os.path.exists(output_path):
                log.info(f"ERROR: File not created at {output_path}")
                if result.stderr:
                    log.info(f"Blender stderr: {result.stderr[:500]}")
                return ("", f"Error: {format_name} not created at {output_path}", 0, fps)
            
            file_size = os.path.getsize(output_path)
            file_size_mb = file_size / (1024 * 1024)
            
            status = f"Exported {len(sorted_indices)} frames as {format_name} (up={up_axis}, skeleton={skel_mode_str})"
            if not include_mesh:
                status += " skeleton only"
            if include_camera:
                status += " +camera"
            status += " [subprocess]"
            
            log.info("=" * 60)
            log.info(f"SUCCESS!")
            log.info(f"{status}")
            log.info(f"File: {output_path}")
            log.info(f"Size: {file_size_mb:.2f} MB")
            log.info("=" * 60)
            
            return (output_path, status, len(sorted_indices), fps)
            
        except subprocess.TimeoutExpired:
            return ("", "Error: Blender timed out", 0, fps)
        except Exception as e:
            return ("", f"Error: {str(e)}", 0, fps)
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)


class ExportFBXFromJSON:
    """
    Export animated FBX or Alembic from saved JSON file.
    
    Note: This is a simplified export without camera_extrinsics/intrinsics inputs.
    For full camera support, use Export Animated FBX with MESH_SEQUENCE input.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "json_path": ("STRING", {"forceInput": True}),
                "output_format": (["FBX", "ABC (Alembic)"], {"default": "FBX"}),
                "up_axis": (["Y", "Z", "-Y", "-Z"], {"default": "Y"}),
                "skeleton_mode": (["Rotations (Recommended)", "Positions (Legacy)"], {
                    "default": "Rotations (Recommended)",
                }),
                "world_translation": ([
                    "None (Body at Origin)",
                    "Baked into Mesh/Joints",
                    "Root Locator",
                    "Separate Track"
                ], {
                    "default": "None (Body at Origin)",
                    "tooltip": "How to handle world translation. For camera modes, use Export Animated FBX with camera inputs."
                }),
            },
            "optional": {
                "filename": ("STRING", {"default": "animation"}),
                "include_mesh": ("BOOLEAN", {"default": True}),
                "include_camera": ("BOOLEAN", {"default": True}),
                "output_dir": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "status")
    FUNCTION = "export_fbx"
    CATEGORY = "SAM3DBody2abc/Export"
    OUTPUT_NODE = True
    
    def export_fbx(
        self,
        json_path: str,
        output_format: str = "FBX",
        up_axis: str = "Y",
        skeleton_mode: str = "Rotations (Recommended)",
        world_translation: str = "None (Body at Origin)",
        filename: str = "animation",
        include_mesh: bool = True,
        include_camera: bool = True,
        output_dir: str = "",
    ) -> Tuple[str, str]:
        """Convert JSON to FBX or Alembic."""
        
        if not os.path.exists(json_path):
            return ("", f"Error: JSON not found: {json_path}")
        
        blender_path = find_blender()
        if not blender_path:
            return ("", "Error: Blender not found")
        
        if not output_dir:
            output_dir = folder_paths.get_output_directory()
        
        # Determine output extension
        use_alembic = "ABC" in output_format
        ext = ".abc" if use_alembic else ".fbx"
        
        # Get incremental filename to avoid overwriting
        output_path = get_incremental_filename(output_dir, filename, ext)
        
        # Map world_translation option to mode
        translation_mode = "none"
        if "Baked into Mesh" in world_translation:
            translation_mode = "baked"
        elif "Root" in world_translation:
            translation_mode = "root"
        elif "Separate" in world_translation:
            translation_mode = "separate"
        
        # Map skeleton_mode option
        use_rotations = "Rotations" in skeleton_mode
        
        # Read JSON and add modes
        with open(json_path, 'r') as f:
            data = json.load(f)
        data["world_translation_mode"] = translation_mode
        data["skeleton_mode"] = "rotations" if use_rotations else "positions"
        
        # Write to temp file with updated modes
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_json = f.name
        
        try:
            cmd = [
                blender_path,
                "--background",
                "--python", BLENDER_SCRIPT,
                "--",
                temp_json,
                output_path,
                up_axis,
                "1" if include_mesh else "0",
                "1" if include_camera else "0",
            ]
            
            format_name = "Alembic" if use_alembic else "FBX"
            skel_mode_str = "rotations" if use_rotations else "positions"
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=BLENDER_TIMEOUT)
            
            if result.returncode != 0:
                return ("", f"Blender error: {result.stderr[:500]}")
            
            if not os.path.exists(output_path):
                return ("", f"Error: {format_name} not created")
            
            return (output_path, f"Exported to {filename}{ext} (up={up_axis}, skeleton={skel_mode_str}, translation={translation_mode})")
            
        except Exception as e:
            return ("", f"Error: {str(e)}")
        finally:
            if os.path.exists(temp_json):
                os.unlink(temp_json)
