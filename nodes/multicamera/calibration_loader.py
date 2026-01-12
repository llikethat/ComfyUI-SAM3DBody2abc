"""
Camera Calibration Loader Node

Loads camera calibration data from JSON file or manual input.
Provides calibration data for multi-camera triangulation.
"""

import json
import os
from typing import Dict, Tuple, Optional, List

# Try to import logger
try:
    from ..lib.logger import get_logger
    log = get_logger("CameraCalibrationLoader")
except ImportError:
    class FallbackLogger:
        def info(self, msg): print(f"[Camera Calibration] {msg}")
        def warning(self, msg): print(f"[Camera Calibration] WARNING: {msg}")
        def error(self, msg): print(f"[Camera Calibration] ERROR: {msg}")
        def debug(self, msg): pass
    log = FallbackLogger()

from .utils.camera import Camera, convert_coordinate_system, compute_baseline, compute_angle_between_cameras


# Default calibration folder
DEFAULT_CALIBRATION_PATH = "/workspace/ComfyUI/models/calibrations"


class CameraCalibrationLoader:
    """
    Load camera calibration data from file or manual input.
    
    Supports:
    - JSON calibration files
    - Manual input of camera parameters
    - Override with MoGe2 intrinsics
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "calibration_source": (["JSON File", "Manual Input"], {
                    "default": "Manual Input",
                    "tooltip": "Source of calibration data"
                }),
            },
            "optional": {
                # JSON file input
                "calibration_file": ("STRING", {
                    "default": "",
                    "tooltip": f"Path to calibration JSON file. Default folder: {DEFAULT_CALIBRATION_PATH}"
                }),
                
                # Coordinate system
                "coordinate_system": (["Y-up (Maya/Unity)", "Z-up (Blender/Unreal)"], {
                    "default": "Y-up (Maya/Unity)",
                    "tooltip": "Coordinate system for position/rotation input and output"
                }),
                
                # Manual input for Camera A
                "cam_a_name": ("STRING", {"default": "Camera A"}),
                "cam_a_position_x": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Camera A X position (meters)"}),
                "cam_a_position_y": ("FLOAT", {"default": 1.5, "min": -100.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Camera A Y position (meters) - height for Y-up"}),
                "cam_a_position_z": ("FLOAT", {"default": 5.0, "min": -100.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Camera A Z position (meters)"}),
                "cam_a_rotation_x": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Camera A rotation X (pitch) in degrees"}),
                "cam_a_rotation_y": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Camera A rotation Y (yaw) in degrees"}),
                "cam_a_rotation_z": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Camera A rotation Z (roll) in degrees"}),
                "cam_a_focal_mm": ("FLOAT", {"default": 35.0, "min": 1.0, "max": 500.0, "step": 1.0,
                    "tooltip": "Camera A focal length in mm"}),
                "cam_a_sensor_width_mm": ("FLOAT", {"default": 36.0, "min": 1.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Camera A sensor width in mm (36 = full frame)"}),
                "cam_a_resolution_w": ("INT", {"default": 1920, "min": 1, "max": 8192,
                    "tooltip": "Camera A video width in pixels"}),
                "cam_a_resolution_h": ("INT", {"default": 1080, "min": 1, "max": 8192,
                    "tooltip": "Camera A video height in pixels"}),
                
                # Manual input for Camera B
                "cam_b_name": ("STRING", {"default": "Camera B"}),
                "cam_b_position_x": ("FLOAT", {"default": 5.0, "min": -100.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Camera B X position (meters)"}),
                "cam_b_position_y": ("FLOAT", {"default": 1.5, "min": -100.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Camera B Y position (meters)"}),
                "cam_b_position_z": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Camera B Z position (meters)"}),
                "cam_b_rotation_x": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Camera B rotation X (pitch) in degrees"}),
                "cam_b_rotation_y": ("FLOAT", {"default": -90.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Camera B rotation Y (yaw) in degrees"}),
                "cam_b_rotation_z": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Camera B rotation Z (roll) in degrees"}),
                "cam_b_focal_mm": ("FLOAT", {"default": 35.0, "min": 1.0, "max": 500.0, "step": 1.0,
                    "tooltip": "Camera B focal length in mm"}),
                "cam_b_sensor_width_mm": ("FLOAT", {"default": 36.0, "min": 1.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Camera B sensor width in mm"}),
                "cam_b_resolution_w": ("INT", {"default": 1920, "min": 1, "max": 8192,
                    "tooltip": "Camera B video width in pixels"}),
                "cam_b_resolution_h": ("INT", {"default": 1080, "min": 1, "max": 8192,
                    "tooltip": "Camera B video height in pixels"}),
                
                # Override with MoGe2 intrinsics
                "cam_a_intrinsics": ("CAMERA_INTRINSICS", {
                    "tooltip": "Optional: Override Camera A intrinsics from MoGe2"
                }),
                "cam_b_intrinsics": ("CAMERA_INTRINSICS", {
                    "tooltip": "Optional: Override Camera B intrinsics from MoGe2"
                }),
            }
        }
    
    RETURN_TYPES = ("CALIBRATION_DATA", "STRING")
    RETURN_NAMES = ("calibration", "calibration_info")
    FUNCTION = "load"
    CATEGORY = "SAM3DBody2abc/MultiCamera"
    
    def load(
        self,
        calibration_source: str,
        calibration_file: str = "",
        coordinate_system: str = "Y-up (Maya/Unity)",
        # Camera A manual inputs
        cam_a_name: str = "Camera A",
        cam_a_position_x: float = 0.0,
        cam_a_position_y: float = 1.5,
        cam_a_position_z: float = 5.0,
        cam_a_rotation_x: float = 0.0,
        cam_a_rotation_y: float = 0.0,
        cam_a_rotation_z: float = 0.0,
        cam_a_focal_mm: float = 35.0,
        cam_a_sensor_width_mm: float = 36.0,
        cam_a_resolution_w: int = 1920,
        cam_a_resolution_h: int = 1080,
        # Camera B manual inputs
        cam_b_name: str = "Camera B",
        cam_b_position_x: float = 5.0,
        cam_b_position_y: float = 1.5,
        cam_b_position_z: float = 0.0,
        cam_b_rotation_x: float = 0.0,
        cam_b_rotation_y: float = -90.0,
        cam_b_rotation_z: float = 0.0,
        cam_b_focal_mm: float = 35.0,
        cam_b_sensor_width_mm: float = 36.0,
        cam_b_resolution_w: int = 1920,
        cam_b_resolution_h: int = 1080,
        # MoGe2 intrinsics override
        cam_a_intrinsics: Optional[Dict] = None,
        cam_b_intrinsics: Optional[Dict] = None,
    ) -> Tuple[Dict, str]:
        """Load calibration data."""
        
        log.info("=" * 60)
        log.info("CAMERA CALIBRATION LOADER")
        log.info("=" * 60)
        
        # Determine internal coordinate system (always Y-up internally)
        user_system = "Y-up" if "Y-up" in coordinate_system else "Z-up"
        log.info(f"User coordinate system: {coordinate_system}")
        
        if calibration_source == "JSON File":
            calibration = self._load_from_json(calibration_file, user_system)
        else:
            calibration = self._load_from_manual(
                user_system,
                cam_a_name, cam_a_position_x, cam_a_position_y, cam_a_position_z,
                cam_a_rotation_x, cam_a_rotation_y, cam_a_rotation_z,
                cam_a_focal_mm, cam_a_sensor_width_mm, cam_a_resolution_w, cam_a_resolution_h,
                cam_b_name, cam_b_position_x, cam_b_position_y, cam_b_position_z,
                cam_b_rotation_x, cam_b_rotation_y, cam_b_rotation_z,
                cam_b_focal_mm, cam_b_sensor_width_mm, cam_b_resolution_w, cam_b_resolution_h,
            )
        
        # Override with MoGe2 intrinsics if provided
        if cam_a_intrinsics is not None:
            log.info("Overriding Camera A intrinsics from MoGe2")
            self._apply_intrinsics_override(calibration["cameras"]["camera_A"], cam_a_intrinsics)
        
        if cam_b_intrinsics is not None:
            log.info("Overriding Camera B intrinsics from MoGe2")
            self._apply_intrinsics_override(calibration["cameras"]["camera_B"], cam_b_intrinsics)
        
        # Create Camera objects
        camera_a = Camera.from_dict(calibration["cameras"]["camera_A"], "camera_A")
        camera_b = Camera.from_dict(calibration["cameras"]["camera_B"], "camera_B")
        
        # Store Camera objects in calibration
        calibration["camera_objects"] = {
            "camera_A": camera_a,
            "camera_B": camera_b
        }
        
        # Compute geometry info
        baseline = compute_baseline(camera_a, camera_b)
        angle = compute_angle_between_cameras(camera_a, camera_b)
        
        calibration["geometry"] = {
            "baseline_m": baseline,
            "angle_between_views_deg": angle
        }
        
        # Generate info string
        info = self._generate_info_string(calibration, camera_a, camera_b, baseline, angle)
        
        log.info(info)
        log.info("=" * 60)
        
        return (calibration, info)
    
    def _load_from_json(self, filepath: str, user_system: str) -> Dict:
        """Load calibration from JSON file."""
        
        # Try to find the file
        if not filepath:
            raise ValueError("No calibration file specified")
        
        # Check if path is absolute or relative
        if not os.path.isabs(filepath):
            # Try default calibration folder first
            default_path = os.path.join(DEFAULT_CALIBRATION_PATH, filepath)
            if os.path.exists(default_path):
                filepath = default_path
            elif os.path.exists(filepath):
                pass  # Use as-is
            else:
                raise FileNotFoundError(f"Calibration file not found: {filepath} or {default_path}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Calibration file not found: {filepath}")
        
        log.info(f"Loading calibration from: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Validate required fields
        if "cameras" not in data:
            raise ValueError("Invalid calibration file: missing 'cameras' section")
        
        cameras = data["cameras"]
        if len(cameras) < 2:
            raise ValueError("Calibration file must contain at least 2 cameras")
        
        # Get first two cameras
        camera_names = list(cameras.keys())[:2]
        
        # Normalize to camera_A, camera_B
        calibration = {
            "version": data.get("version", "1.0"),
            "name": data.get("name", "Loaded Calibration"),
            "source": filepath,
            "coordinate_system": data.get("coordinate_system", {
                "up": "Y" if user_system == "Y-up" else "Z",
                "unit": "meters"
            }),
            "cameras": {
                "camera_A": cameras[camera_names[0]],
                "camera_B": cameras[camera_names[1]]
            }
        }
        
        # Ensure camera names are set
        calibration["cameras"]["camera_A"]["name"] = cameras[camera_names[0]].get("name", camera_names[0])
        calibration["cameras"]["camera_B"]["name"] = cameras[camera_names[1]].get("name", camera_names[1])
        
        return calibration
    
    def _load_from_manual(
        self, user_system: str,
        cam_a_name, cam_a_pos_x, cam_a_pos_y, cam_a_pos_z,
        cam_a_rot_x, cam_a_rot_y, cam_a_rot_z,
        cam_a_focal, cam_a_sensor, cam_a_res_w, cam_a_res_h,
        cam_b_name, cam_b_pos_x, cam_b_pos_y, cam_b_pos_z,
        cam_b_rot_x, cam_b_rot_y, cam_b_rot_z,
        cam_b_focal, cam_b_sensor, cam_b_res_w, cam_b_res_h,
    ) -> Dict:
        """Load calibration from manual input."""
        
        log.info("Loading calibration from manual input")
        
        # Convert positions if needed (Z-up to Y-up)
        pos_a = [cam_a_pos_x, cam_a_pos_y, cam_a_pos_z]
        pos_b = [cam_b_pos_x, cam_b_pos_y, cam_b_pos_z]
        
        if user_system == "Z-up":
            pos_a = convert_coordinate_system(pos_a, "Z-up", "Y-up")
            pos_b = convert_coordinate_system(pos_b, "Z-up", "Y-up")
            log.info("Converted positions from Z-up to Y-up (internal)")
        
        calibration = {
            "version": "1.0",
            "name": "Manual Calibration",
            "source": "manual_input",
            "coordinate_system": {
                "up": "Y",  # Internal is always Y-up
                "unit": "meters"
            },
            "user_coordinate_system": user_system,
            "cameras": {
                "camera_A": {
                    "name": cam_a_name,
                    "position": pos_a,
                    "rotation_euler": [cam_a_rot_x, cam_a_rot_y, cam_a_rot_z],
                    "rotation_order": "XYZ",
                    "focal_length_mm": cam_a_focal,
                    "sensor_width_mm": cam_a_sensor,
                    "sensor_height_mm": cam_a_sensor * cam_a_res_h / cam_a_res_w,
                    "resolution": [cam_a_res_w, cam_a_res_h],
                    "principal_point": [cam_a_res_w / 2, cam_a_res_h / 2],
                    "distortion": {"k1": 0, "k2": 0, "p1": 0, "p2": 0}
                },
                "camera_B": {
                    "name": cam_b_name,
                    "position": pos_b,
                    "rotation_euler": [cam_b_rot_x, cam_b_rot_y, cam_b_rot_z],
                    "rotation_order": "XYZ",
                    "focal_length_mm": cam_b_focal,
                    "sensor_width_mm": cam_b_sensor,
                    "sensor_height_mm": cam_b_sensor * cam_b_res_h / cam_b_res_w,
                    "resolution": [cam_b_res_w, cam_b_res_h],
                    "principal_point": [cam_b_res_w / 2, cam_b_res_h / 2],
                    "distortion": {"k1": 0, "k2": 0, "p1": 0, "p2": 0}
                }
            }
        }
        
        return calibration
    
    def _apply_intrinsics_override(self, camera_data: Dict, intrinsics: Dict):
        """Apply MoGe2 intrinsics to camera data."""
        
        if "focal_length_mm" in intrinsics:
            camera_data["focal_length_mm"] = intrinsics["focal_length_mm"]
        elif "focal_length" in intrinsics:
            # Convert from pixels to mm if needed
            sensor_w = camera_data.get("sensor_width_mm", 36.0)
            res_w = camera_data["resolution"][0]
            focal_px = intrinsics["focal_length"]
            camera_data["focal_length_mm"] = focal_px * sensor_w / res_w
        
        if "width" in intrinsics and "height" in intrinsics:
            camera_data["resolution"] = [intrinsics["width"], intrinsics["height"]]
        
        if "cx" in intrinsics and "cy" in intrinsics:
            camera_data["principal_point"] = [intrinsics["cx"], intrinsics["cy"]]
        
        if "sensor_width_mm" in intrinsics:
            camera_data["sensor_width_mm"] = intrinsics["sensor_width_mm"]
    
    def _generate_info_string(
        self, calibration: Dict, 
        camera_a: Camera, camera_b: Camera,
        baseline: float, angle: float
    ) -> str:
        """Generate info string for display."""
        
        lines = [
            "=== CALIBRATION INFO ===",
            f"Name: {calibration.get('name', 'Unknown')}",
            f"Source: {calibration.get('source', 'Unknown')}",
            "",
            f"Camera A: {camera_a.name}",
            f"  Position: ({camera_a.position[0]:.2f}, {camera_a.position[1]:.2f}, {camera_a.position[2]:.2f}) m",
            f"  Rotation: ({camera_a.rotation_euler[0]:.1f}°, {camera_a.rotation_euler[1]:.1f}°, {camera_a.rotation_euler[2]:.1f}°)",
            f"  Focal: {camera_a.focal_mm:.1f}mm ({camera_a.focal_px:.0f}px)",
            f"  Resolution: {camera_a.width}x{camera_a.height}",
            "",
            f"Camera B: {camera_b.name}",
            f"  Position: ({camera_b.position[0]:.2f}, {camera_b.position[1]:.2f}, {camera_b.position[2]:.2f}) m",
            f"  Rotation: ({camera_b.rotation_euler[0]:.1f}°, {camera_b.rotation_euler[1]:.1f}°, {camera_b.rotation_euler[2]:.1f}°)",
            f"  Focal: {camera_b.focal_mm:.1f}mm ({camera_b.focal_px:.0f}px)",
            f"  Resolution: {camera_b.width}x{camera_b.height}",
            "",
            "=== GEOMETRY ===",
            f"Baseline: {baseline:.2f}m",
            f"Angle between views: {angle:.1f}°",
        ]
        
        # Quality assessment
        if angle < 30:
            lines.append("⚠️ Warning: Small angle between cameras - depth accuracy may be limited")
        elif angle > 150:
            lines.append("⚠️ Warning: Cameras nearly opposite - may have occlusion issues")
        else:
            lines.append("✓ Good camera geometry for triangulation")
        
        return "\n".join(lines)


# Node registration
NODE_CLASS_MAPPINGS = {
    "CameraCalibrationLoader": CameraCalibrationLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CameraCalibrationLoader": "📷 Camera Calibration Loader",
}
