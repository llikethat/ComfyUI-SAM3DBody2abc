"""
Animated FBX Export for SAM3DBody2abc
Exports MESH_SEQUENCE to animated FBX using Blender.

The exported FBX contains:
- Mesh with shape keys (vertex animation per frame)
- Armature with keyframed bone rotations (from MHR) or positions

Settings:
- skeleton_mode: "Rotations" uses true joint rotations from MHR model
                 "Positions" uses joint positions (legacy)
"""

import os
import json
import subprocess
import shutil
import glob
import tempfile
import numpy as np
import torch
from typing import Dict, Tuple, Any, Optional
import folder_paths

# Import version from package
try:
    from .. import __version__
except ImportError:
    __version__ = "unknown"


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


def find_blender() -> Optional[str]:
    """Find Blender executable."""
    global _BLENDER_PATH
    
    if _BLENDER_PATH is not None:
        return _BLENDER_PATH
    
    locations = [
        shutil.which("blender"),
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ]
    
    # Check SAM3DBody bundled Blender
    try:
        custom_nodes = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        patterns = [
            os.path.join(custom_nodes, "ComfyUI-SAM3DBody", "lib", "blender", "blender-*-linux-x64", "blender"),
            os.path.join(custom_nodes, "ComfyUI-SAM3DBody", "lib", "blender", "*", "blender"),
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            locations.extend(matches)
    except Exception:
        pass
    
    # Windows
    for ver in ["4.2", "4.1", "4.0", "3.6"]:
        locations.append(f"C:\\Program Files\\Blender Foundation\\Blender {ver}\\blender.exe")
    
    for loc in locations:
        if loc and os.path.exists(loc):
            _BLENDER_PATH = loc
            print(f"[FBX Export] Found Blender: {loc}")
            return loc
    
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
                # Camera data - unified input from CameraSolver V2 or Legacy
                "camera_matrices": ("CAMERA_MATRICES", {
                    "tooltip": "Camera matrices from Camera Solver (V2 or Legacy). Contains rotation/translation per frame + intrinsics."
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
                
                # World translation mode
                "world_translation": ([
                    "None (Body at Origin)",
                    "Baked into Mesh/Joints",
                    "Root Locator",
                    "Separate Track"
                ], {
                    "default": "None (Body at Origin)",
                    "tooltip": "How to handle world translation from pred_cam_t."
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
                "include_camera": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include animated camera from camera_matrices"
                }),
                
                # Fallback sensor width (used if no camera_matrices provided)
                "sensor_width": ("FLOAT", {
                    "default": 36.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Camera sensor width in mm. Only used if camera_matrices not provided."
                }),
                
                # Optional metadata inputs
                "subject_motion": ("SUBJECT_MOTION", {
                    "tooltip": "Motion analysis data from Motion Analyzer node (optional, for metadata)"
                }),
                "scale_info": ("SCALE_INFO", {
                    "tooltip": "Scale information from Motion Analyzer node (optional, for metadata)"
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
        subject_motion: Optional[Dict],
        scale_info: Optional[Dict],
        source_video_fps: float,
        skip_first_frames: int,
        fps: float,
        frame_count: int,
    ) -> Dict:
        """
        Build metadata dict to be embedded in FBX as custom properties.
        
        This creates a SAM3DBody_Metadata locator in the FBX with all
        analysis data accessible as Extra Attributes in Maya.
        """
        from datetime import datetime, timezone, timedelta
        
        # Get version from parent package - use direct file read since module name has dashes
        __version__ = "unknown"
        try:
            import os
            init_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "__init__.py")
            if os.path.exists(init_path):
                with open(init_path, "r") as f:
                    for line in f:
                        if line.strip().startswith("__version__"):
                            # Parse: __version__ = "4.5.8"
                            __version__ = line.split("=")[1].strip().strip('"\'').strip()
                            break
        except Exception as e:
            print(f"[Export] Warning: Could not read version: {e}")
            __version__ = "unknown"
        
        # Get timestamp in IST
        ist = timezone(timedelta(hours=5, minutes=30))
        timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
        
        metadata = {
            # Build info
            "sam3dbody2abc_version": __version__,
            "export_timestamp": timestamp,
            # Export settings
            "world_translation": world_translation,
            "export_fps": fps,
            "frame_count": frame_count,
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
        camera_matrices: Optional[Dict] = None,
        fps: float = 0.0,
        frame_offset: int = 1,
        output_format: str = "FBX",
        up_axis: str = "Y",
        skeleton_mode: str = "Rotations (Recommended)",
        world_translation: str = "None (Body at Origin)",
        flip_x: bool = False,
        include_mesh: bool = True,
        include_camera: bool = True,
        sensor_width: float = 36.0,
        subject_motion: Optional[Dict] = None,
        scale_info: Optional[Dict] = None,
        source_video_fps: float = 0.0,
        skip_first_frames: int = 0,
        output_dir: str = "",
    ) -> Tuple[str, str, int, float]:
        """Export to animated FBX or Alembic."""
        
        # Log version at start of export
        print(f"[Export] SAM3DBody2abc version: {__version__}")
        
        # Get fps from mesh_sequence if not specified (0 means use source)
        if fps <= 0:
            fps = mesh_sequence.get("fps", 24.0)
            print(f"[Export] Using fps from source: {fps}")
        
        # Extract intrinsics from camera_matrices if provided
        if camera_matrices is not None:
            cam_intrinsics = camera_matrices.get("intrinsics", {})
            intrinsics_focal_px = cam_intrinsics.get("focal_px", None)
            intrinsics_sensor_mm = cam_intrinsics.get("sensor_width_mm", sensor_width)
            intrinsics_source = cam_intrinsics.get("source", "camera_solver")
            intrinsics_cx = cam_intrinsics.get("cx", None)
            intrinsics_cy = cam_intrinsics.get("cy", None)
            intrinsics_w = cam_intrinsics.get("width", None)
            intrinsics_h = cam_intrinsics.get("height", None)
            
            if intrinsics_focal_px:
                print(f"[Export] Using intrinsics from {intrinsics_source}: focal={intrinsics_focal_px:.1f}px")
            
            # Log principal point (cx, cy)
            if intrinsics_cx is not None and intrinsics_cy is not None:
                print(f"[Export] Principal point: cx={intrinsics_cx:.2f}px, cy={intrinsics_cy:.2f}px")
        else:
            intrinsics_focal_px = None
            intrinsics_sensor_mm = sensor_width
            intrinsics_source = "manual"
            cam_intrinsics = {}
        
        # Check if we have camera matrices (rotation/translation data)
        has_camera_data = camera_matrices is not None and "matrices" in camera_matrices
        if has_camera_data:
            cam_source = camera_matrices.get("source", "CameraSolver")
            shot_type = camera_matrices.get("shot_type", "unknown")
            num_cam_frames = len(camera_matrices.get("matrices", []))
            print(f"[Export] Using camera data from {cam_source}: {num_cam_frames} frames, shot_type={shot_type}")
        
        # Simplified camera behavior
        # If camera_matrices provided and include_camera=True, export animated camera
        animate_camera = has_camera_data and include_camera
        
        if include_camera:
            if has_camera_data:
                print(f"[Export] Camera will be animated from camera_matrices ({shot_type} shot)")
            else:
                print(f"[Export] Camera will be static (no camera_matrices provided)")
        
        frames = mesh_sequence.get("frames", {})
        if not frames:
            return ("", "Error: No frames", 0, fps)
        
        blender_path = find_blender()
        if not blender_path:
            return ("", "Error: Blender not found", 0, fps)
        
        if not os.path.exists(BLENDER_SCRIPT):
            return ("", f"Error: Script not found: {BLENDER_SCRIPT}", 0, fps)
        
        if not output_dir:
            output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        sorted_indices = sorted(frames.keys())
        
        # Determine output extension
        use_alembic = "ABC" in output_format
        ext = ".abc" if use_alembic else ".fbx"
        
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
        
        # Check if rotation data is available (handle numpy arrays properly)
        first_frame = frames[sorted_indices[0]]
        joint_rots = first_frame.get("joint_rotations")
        
        # Debug: print what data we have
        print(f"[Export] First frame keys: {list(first_frame.keys())}")
        print(f"[Export] joint_rotations type: {type(joint_rots)}")
        if joint_rots is not None:
            if isinstance(joint_rots, np.ndarray):
                print(f"[Export] joint_rotations shape: {joint_rots.shape}, size: {joint_rots.size}")
            elif isinstance(joint_rots, list):
                print(f"[Export] joint_rotations is list with {len(joint_rots)} elements")
        
        has_rotations = joint_rots is not None
        if has_rotations and isinstance(joint_rots, np.ndarray):
            has_rotations = joint_rots.size > 0
        elif has_rotations and isinstance(joint_rots, list):
            has_rotations = len(joint_rots) > 0
        
        if use_rotations and not has_rotations:
            print("[Export] Warning: Rotation mode requested but no rotation data available. Falling back to positions.")
            print("[Export] Note: Make sure you're using a recent version of ComfyUI-SAM3DBody that outputs joint_rotations.")
            use_rotations = False
        else:
            print(f"[Export] Rotation data available: {has_rotations}, using rotations: {use_rotations}")
        
        # Build JSON for Blender
        joint_parents = mesh_sequence.get("joint_parents")
        print(f"[Export] joint_parents in mesh_sequence: {joint_parents is not None}")
        if joint_parents is not None:
            if hasattr(joint_parents, 'shape'):
                print(f"[Export] joint_parents shape: {joint_parents.shape}")
            elif isinstance(joint_parents, list):
                print(f"[Export] joint_parents length: {len(joint_parents)}")
        
        print(f"[Export] Camera animated: {animate_camera}")
        
        # Prepare camera extrinsics for Blender (from camera_matrices)
        solved_rotations = None
        if has_camera_data:
            solved_rotations = []
            matrices_list = camera_matrices.get("matrices", [])
            for rot_data in matrices_list:
                rot_entry = {
                    "frame": rot_data.get("frame", 0),
                    "pan": np.radians(rot_data.get("pan", 0.0)),  # Convert from degrees
                    "tilt": np.radians(rot_data.get("tilt", 0.0)),
                    "roll": np.radians(rot_data.get("roll", 0.0)),
                    "tx": 0.0,
                    "ty": 0.0,
                    "tz": 0.0,
                }
                solved_rotations.append(rot_entry)
            
            final_rot = solved_rotations[-1]
            print(f"[Export] Camera extrinsics: {len(solved_rotations)} frames")
            print(f"[Export]   Final: pan={np.degrees(final_rot['pan']):.2f}°, tilt={np.degrees(final_rot['tilt']):.2f}°")
        
        # Prepare intrinsics data for export (from camera_matrices)
        intrinsics_export = None
        if cam_intrinsics:
            intrinsics_export = {
                "focal_length_px": to_list(cam_intrinsics.get("focal_px")),
                "focal_length_mm": to_list(cam_intrinsics.get("focal_mm")),
                "sensor_width_mm": to_list(cam_intrinsics.get("sensor_width_mm", sensor_width)),
                "principal_point_x": to_list(cam_intrinsics.get("cx")),
                "principal_point_y": to_list(cam_intrinsics.get("cy")),
                "image_width": to_list(cam_intrinsics.get("width")),
                "image_height": to_list(cam_intrinsics.get("height")),
                "source": cam_intrinsics.get("source", "camera_solver"),
            }
        
        export_data = {
            "fps": float(fps),
            "frame_count": len(sorted_indices),
            "frame_offset": int(frame_offset),
            "faces": to_list(mesh_sequence.get("faces")),
            "joint_parents": to_list(joint_parents),
            "sensor_width": float(sensor_width),
            "world_translation_mode": translation_mode,
            "skeleton_mode": "rotations" if use_rotations else "positions",
            "flip_x": flip_x,
            "animate_camera": animate_camera,
            "camera_follow_root": animate_camera and len(solved_rotations) > 0,  # Enable when we have camera data
            "camera_use_rotation": True,   # Always use rotation for camera animation
            "camera_static": not animate_camera,
            "camera_compensation": False,  # Simplified - no longer using this mode
            "camera_extrinsics": to_list(solved_rotations),  # From Camera Solver
            "camera_intrinsics": intrinsics_export,
            "extrinsics_smoothing_method": "none",  # Smoothing already done in solver
            "extrinsics_smoothing_strength": 0.0,
            "frames": [],
            # Body world trajectory for animated locator
            "body_world_trajectory": to_list(subject_motion.get("body_world_3d", [])) if subject_motion else [],
            "body_world_trajectory_compensated": to_list(subject_motion.get("body_world_3d_compensated", [])) if subject_motion else [],
            "body_world_trajectory_raw": to_list(subject_motion.get("body_world_3d_raw", [])) if subject_motion else [],
            # Metadata for embedding in FBX
            "metadata": self._build_metadata(
                world_translation=world_translation,
                subject_motion=subject_motion,
                scale_info=scale_info,
                source_video_fps=source_video_fps,
                skip_first_frames=skip_first_frames,
                fps=fps,
                frame_count=len(sorted_indices),
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
                print(f"\n[Export DEBUG] ========== BODY ALIGNMENT (Frame 0) ==========")
                print(f"[Export DEBUG] pred_cam_t: tx={tx:.4f}, ty={ty:.4f}, tz={tz:.4f}")
                print(f"[Export DEBUG] focal_length: {first_focal}")
                print(f"[Export DEBUG] image_size: {first_image_size}")
                print(f"[Export DEBUG]")
                print(f"[Export DEBUG] NEW APPROACH (v3.5.7):")
                print(f"[Export DEBUG]   root_locator = (0, 0, 0)  ← Fixed at origin")
                print(f"[Export DEBUG]   body_offset in Blender = (tx, -ty, 0) = ({tx:.4f}, {-ty:.4f}, 0)")
                print(f"[Export DEBUG]   body_offset in Maya = (tx, 0, -ty) = ({tx:.4f}, 0, {-ty:.4f})")
                print(f"[Export DEBUG]   (ty negated due to camera rotation convention)")
                print(f"[Export DEBUG]")
                
                # What screen position does this correspond to?
                if first_focal and first_image_size:
                    focal = float(first_focal) if not isinstance(first_focal, (list, tuple)) else float(first_focal[0])
                    img_w, img_h = first_image_size[0], first_image_size[1]
                    cx, cy = img_w / 2, img_h / 2
                    # Screen position from pred_cam_t
                    screen_x = focal * tx / tz + cx
                    screen_y = focal * ty / tz + cy  # NO negation - SAM3DBody coords are image-aligned
                    print(f"[Export DEBUG] EXPECTED SCREEN POSITION:")
                    print(f"[Export DEBUG]   screen_x = {screen_x:.1f}px")
                    print(f"[Export DEBUG]   screen_y = {screen_y:.1f}px")
                    print(f"[Export DEBUG]   (Image center: {cx:.0f}, {cy:.0f})")
                print(f"[Export DEBUG] ================================================================\n")
        
        # ============================================================
        # DEPTH/SCALE ANALYSIS - Track subject distance across frames
        # ============================================================
        # This captures subject movement toward/away from camera
        # Even with nodal/rotation camera, subject can translate in depth
        # ============================================================
        
        print(f"\n[Export] ============== DEPTH/SCALE ANALYSIS ==============")
        
        tz_values = []
        tx_values = []
        ty_values = []
        focal_values = []
        
        for idx in sorted_indices:
            frame = frames[idx]
            pred_cam_t = frame.get("pred_cam_t")
            if pred_cam_t is None:
                pred_cam_t = frame.get("camera")
            
            if pred_cam_t is not None:
                cam_t = to_list(pred_cam_t)
                if len(cam_t) >= 3:
                    tx_values.append(cam_t[0])
                    ty_values.append(cam_t[1])
                    tz_values.append(cam_t[2])
            
            focal = frame.get("focal_length")
            if focal is not None:
                if hasattr(focal, 'item'):
                    focal = focal.item()
                focal_values.append(float(focal))
        
        if tz_values:
            tz_arr = np.array(tz_values)
            tx_arr = np.array(tx_values)
            ty_arr = np.array(ty_values)
            
            # Depth statistics
            tz_min, tz_max = tz_arr.min(), tz_arr.max()
            tz_range = tz_max - tz_min
            tz_first, tz_last = tz_arr[0], tz_arr[-1]
            tz_change = tz_last - tz_first
            
            # Scale change (inverse of depth - closer = larger)
            scale_first = 1.0 / tz_first if tz_first != 0 else 0
            scale_last = 1.0 / tz_last if tz_last != 0 else 0
            scale_change_pct = ((scale_last / scale_first) - 1) * 100 if scale_first != 0 else 0
            
            print(f"[Export] pred_cam_t depth (tz) analysis:")
            print(f"[Export]   Frame 0:   tz = {tz_first:.3f}")
            print(f"[Export]   Frame {len(tz_arr)-1}:  tz = {tz_last:.3f}")
            print(f"[Export]   Change:    Δtz = {tz_change:+.3f} ({'closer' if tz_change < 0 else 'farther' if tz_change > 0 else 'same'})")
            print(f"[Export]   Range:     {tz_min:.3f} to {tz_max:.3f} (spread: {tz_range:.3f})")
            print(f"[Export]   Scale change: {scale_change_pct:+.1f}% ({'larger' if scale_change_pct > 0 else 'smaller' if scale_change_pct < 0 else 'same'})")
            
            # Sample tz values at key frames
            sample_frames = [0, len(tz_arr)//4, len(tz_arr)//2, 3*len(tz_arr)//4, len(tz_arr)-1]
            sample_frames = sorted(set([min(f, len(tz_arr)-1) for f in sample_frames]))
            print(f"[Export]   Sample tz values:")
            for f in sample_frames:
                rel_scale = (1.0/tz_arr[f]) / scale_first * 100 if scale_first != 0 and tz_arr[f] != 0 else 100
                print(f"[Export]     Frame {f:3d}: tz={tz_arr[f]:.3f}, scale={rel_scale:.1f}%")
            
            # Horizontal/Vertical movement
            tx_change = tx_arr[-1] - tx_arr[0]
            ty_change = ty_arr[-1] - ty_arr[0]
            print(f"[Export]   Horizontal movement (tx): {tx_arr[0]:.3f} → {tx_arr[-1]:.3f} (Δ={tx_change:+.3f})")
            print(f"[Export]   Vertical movement (ty):   {ty_arr[0]:.3f} → {ty_arr[-1]:.3f} (Δ={ty_change:+.3f})")
            
            # Detect if this looks like subject approaching camera
            if abs(tz_change) > 0.5:
                if tz_change < 0:
                    print(f"[Export]   ⚠️  Subject appears to be APPROACHING camera (tz decreasing)")
                    print(f"[Export]      This creates scaling effect similar to zoom/dolly")
                else:
                    print(f"[Export]   ⚠️  Subject appears to be RECEDING from camera (tz increasing)")
            
            # Check focal length consistency
            if focal_values:
                focal_arr = np.array(focal_values)
                focal_std = focal_arr.std()
                if focal_std > 1.0:
                    print(f"[Export]   ⚠️  Focal length varies: {focal_arr.min():.1f} to {focal_arr.max():.1f} (std={focal_std:.2f})")
                else:
                    print(f"[Export]   Focal length: {focal_arr.mean():.1f}px (consistent)")
        
        print(f"[Export] =========================================================\n")
        
        for idx in sorted_indices:
            frame = frames[idx]
            
            # Handle pred_cam_t vs camera field (can't use 'or' with numpy arrays)
            pred_cam_t = frame.get("pred_cam_t")
            if pred_cam_t is None:
                pred_cam_t = frame.get("camera")
            
            # Get focal_length and convert to float if needed
            focal = frame.get("focal_length")
            if focal is not None:
                if hasattr(focal, 'item'):
                    focal = float(focal.item())
                elif isinstance(focal, np.ndarray):
                    focal = float(focal.flat[0])
                else:
                    focal = float(focal)
            
            # Get image_size and ensure it's a list
            img_size = frame.get("image_size")
            if img_size is not None:
                img_size = to_list(img_size)
            
            frame_data = {
                "frame_index": int(idx),
                "joint_coords": to_list(frame.get("joint_coords")),
                "pred_cam_t": to_list(pred_cam_t),
                "focal_length": focal,
                "bbox": to_list(frame.get("bbox")),  # For camera alignment
                "image_size": img_size,  # (width, height)
                "keypoints_2d": to_list(frame.get("keypoints_2d")),  # For apparent height
                "keypoints_3d": to_list(frame.get("keypoints_3d")),  # 18-joint 3D keypoints
            }
            if include_mesh:
                frame_data["vertices"] = to_list(frame.get("vertices"))
            if use_rotations:
                frame_data["joint_rotations"] = to_list(frame.get("joint_rotations"))
            export_data["frames"].append(frame_data)
        
        # Write temp JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(export_data, f)
            json_path = f.name
        
        # Get incremental filename to avoid overwriting
        output_path = get_incremental_filename(output_dir, filename, ext)
        
        try:
            cmd = [
                blender_path,
                "--background",
                "--python", BLENDER_SCRIPT,
                "--",
                json_path,
                output_path,
                up_axis,
                "1" if include_mesh else "0",
                "1" if include_camera else "0",
            ]
            
            format_name = "Alembic" if use_alembic else "FBX"
            skel_mode_str = "rotations" if use_rotations else "positions"
            print(f"[Export] Exporting {len(sorted_indices)} frames as {format_name}")
            print(f"[Export] Settings: up={up_axis}, translation={translation_mode}, skeleton={skel_mode_str}, camera={include_camera}")
            print(f"[Export] Output path: {output_path}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=BLENDER_TIMEOUT,
            )
            
            # Log Blender output
            if result.stdout:
                lines = result.stdout.split('\n')
                
                # First, print all DEBUG lines (important for troubleshooting)
                debug_lines = [l for l in lines if 'DEBUG' in l and l.strip()]
                if debug_lines:
                    print(f"[Blender] === DEBUG OUTPUT ({len(debug_lines)} lines) ===")
                    for line in debug_lines:
                        print(f"[Blender] {line}")
                    print(f"[Blender] === END DEBUG ===")
                
                # Then print last 30 lines of regular output
                for line in lines[-30:]:
                    if line.strip() and 'DEBUG' not in line:
                        print(f"[Blender] {line}")
            
            if result.returncode != 0:
                error = result.stderr[:500] if result.stderr else "Unknown error"
                print(f"[Export] Blender error: {error}")
                return ("", f"Blender error: {error}", 0, fps)
            
            if not os.path.exists(output_path):
                print(f"[Export] ERROR: File not created at {output_path}")
                if result.stderr:
                    print(f"[Export] Blender stderr: {result.stderr[:500]}")
                return ("", f"Error: {format_name} not created at {output_path}", 0, fps)
            
            file_size = os.path.getsize(output_path)
            file_size_mb = file_size / (1024 * 1024)
            
            status = f"Exported {len(sorted_indices)} frames as {format_name} (up={up_axis}, skeleton={skel_mode_str})"
            if not include_mesh:
                status += " skeleton only"
            if include_camera:
                status += " +camera"
            
            print(f"\n{'='*60}")
            print(f"[Export] SUCCESS!")
            print(f"[Export] {status}")
            print(f"[Export] File: {output_path}")
            print(f"[Export] Size: {file_size_mb:.2f} MB")
            print(f"{'='*60}\n")
            
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
