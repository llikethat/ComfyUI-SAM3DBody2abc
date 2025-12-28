"""
COLMAP to CAMERA_EXTRINSICS Bridge Node

Converts COLMAP's CAMERA_DATA output to the standardized CAMERA_EXTRINSICS format
used by SAM3DBody2abc's FBX export.

Use this when you've run standalone COLMAP nodes and want to use the results
with SAM3DBody2abc export pipeline.
"""

import numpy as np
import math
from typing import Dict, Tuple, Any, Optional


def rotation_matrix_to_euler_yxz(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler angles in YXZ order (camera convention).
    
    Returns (pan, tilt, roll) in radians where:
    - pan: rotation around Y axis (horizontal)
    - tilt: rotation around X axis (vertical)
    - roll: rotation around Z axis (dutch angle)
    """
    # YXZ decomposition
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    
    if abs(sy) < 0.99999:
        pan = math.atan2(R[2, 0], R[2, 2])  # atan2(-R20, R22)
        tilt = math.asin(sy)                  # asin(-R20) with clamp
        roll = math.atan2(R[0, 1], R[1, 1])  # atan2(R01, R11)
    else:
        # Gimbal lock
        pan = math.atan2(-R[0, 2], R[0, 0])
        tilt = math.pi / 2 * np.sign(sy)
        roll = 0
    
    return pan, tilt, roll


class COLMAPToExtrinsicsBridge:
    """
    Convert COLMAP CAMERA_DATA to CAMERA_EXTRINSICS format.
    
    COLMAP outputs a class-based structure with:
    - intrinsics: CameraIntrinsics object
    - extrinsics: List of CameraExtrinsics objects (rotation_matrix, position)
    
    This bridge converts to:
    - CAMERA_EXTRINSICS: Dict with rotations list (pan, tilt, roll, tx, ty, tz)
    """
    
    COORD_SYSTEMS = ["Y-up", "Z-up", "colmap", "opencv", "opengl", "blender"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "colmap_camera_data": ("CAMERA_DATA",),
            },
            "optional": {
                "coordinate_system": (cls.COORD_SYSTEMS, {
                    "default": "Y-up",
                    "tooltip": "Target coordinate system for output"
                }),
                "frame_offset": ("INT", {
                    "default": 0,
                    "min": -1000,
                    "max": 1000,
                    "tooltip": "Offset to add to frame numbers"
                }),
                "auto_frame_offset": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-offset frames to start from 0 or 1"
                }),
                "scale_translation": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.001,
                    "max": 1000.0,
                    "step": 0.01,
                    "tooltip": "Scale factor for translation values"
                }),
            }
        }
    
    RETURN_TYPES = ("CAMERA_EXTRINSICS", "STRING")
    RETURN_NAMES = ("camera_extrinsics", "status")
    FUNCTION = "convert"
    CATEGORY = "SAM3DBody2abc/Camera"
    
    def convert(
        self,
        colmap_camera_data: Any,
        coordinate_system: str = "Y-up",
        frame_offset: int = 0,
        auto_frame_offset: bool = True,
        scale_translation: float = 1.0,
    ) -> Tuple[Dict, str]:
        """Convert COLMAP CAMERA_DATA to CAMERA_EXTRINSICS format."""
        
        print("[COLMAPBridge] Converting COLMAP camera data to extrinsics...")
        
        # Handle case where colmap_camera_data is None or empty
        if colmap_camera_data is None:
            return self._empty_result("No COLMAP data provided")
        
        # Check if it's the class-based COLMAP format
        if hasattr(colmap_camera_data, 'extrinsics'):
            return self._convert_class_format(
                colmap_camera_data, coordinate_system, frame_offset, 
                auto_frame_offset, scale_translation
            )
        
        # Check if it's already a dict (maybe from our camera solver)
        elif isinstance(colmap_camera_data, dict):
            if "rotations" in colmap_camera_data:
                # Already in our format, just pass through with coordinate conversion
                return self._convert_dict_format(
                    colmap_camera_data, coordinate_system, frame_offset,
                    auto_frame_offset, scale_translation
                )
            else:
                return self._empty_result("Dict format not recognized")
        
        else:
            return self._empty_result(f"Unknown format: {type(colmap_camera_data)}")
    
    def _convert_class_format(
        self,
        colmap_data: Any,
        coord_sys: str,
        frame_offset: int,
        auto_offset: bool,
        scale_trans: float,
    ) -> Tuple[Dict, str]:
        """Convert COLMAP's class-based format."""
        
        extrinsics_list = colmap_data.extrinsics
        
        if not extrinsics_list:
            return self._empty_result("No extrinsics in COLMAP data")
        
        # Get frame indices
        frame_indices = sorted([ext.frame_index for ext in extrinsics_list])
        
        # Auto-offset to start from 0
        calculated_offset = frame_offset
        if auto_offset and frame_indices:
            min_frame = min(frame_indices)
            if min_frame > 0:
                calculated_offset = -min_frame + frame_offset
                print(f"[COLMAPBridge] Auto frame offset: {calculated_offset} (original min frame: {min_frame})")
        
        # Convert each extrinsic
        rotations = []
        for ext in extrinsics_list:
            R = ext.rotation_matrix
            pos = ext.position
            
            # Convert rotation matrix to Euler angles
            pan, tilt, roll = rotation_matrix_to_euler_yxz(R)
            
            # Apply coordinate system conversion if needed
            if coord_sys == "Z-up":
                # Swap Y and Z for Z-up systems
                pan, tilt, roll = self._convert_y_to_z_up(pan, tilt, roll)
                tx, ty, tz = pos[0], pos[2], -pos[1]
            else:
                tx, ty, tz = pos[0], pos[1], pos[2]
            
            # Apply scale and offset
            frame_idx = ext.frame_index + calculated_offset
            
            rotations.append({
                "frame": frame_idx,
                "pan": float(pan),
                "tilt": float(tilt),
                "roll": float(roll),
                "pan_deg": float(np.degrees(pan)),
                "tilt_deg": float(np.degrees(tilt)),
                "roll_deg": float(np.degrees(roll)),
                "tx": float(tx) * scale_trans,
                "ty": float(ty) * scale_trans,
                "tz": float(tz) * scale_trans,
            })
        
        # Sort by frame
        rotations.sort(key=lambda x: x["frame"])
        
        # Get image dimensions from intrinsics
        img_w = colmap_data.intrinsics.width if hasattr(colmap_data, 'intrinsics') and colmap_data.intrinsics else 1920
        img_h = colmap_data.intrinsics.height if hasattr(colmap_data, 'intrinsics') and colmap_data.intrinsics else 1080
        
        # Check for translation
        has_trans = any(
            abs(r["tx"]) > 0.001 or abs(r["ty"]) > 0.001 or abs(r["tz"]) > 0.001
            for r in rotations
        )
        
        camera_extrinsics = {
            "num_frames": len(rotations),
            "image_width": img_w,
            "image_height": img_h,
            "source": "COLMAP",
            "solving_method": "full_6dof",
            "coordinate_system": coord_sys,
            "units": "radians",
            "has_translation": has_trans,
            "frame_offset_applied": calculated_offset,
            "scale_applied": scale_trans,
            "rotations": rotations,
        }
        
        status = f"Converted {len(rotations)} COLMAP frames to {coord_sys} extrinsics"
        print(f"[COLMAPBridge] {status}")
        
        return (camera_extrinsics, status)
    
    def _convert_dict_format(
        self,
        data: Dict,
        coord_sys: str,
        frame_offset: int,
        auto_offset: bool,
        scale_trans: float,
    ) -> Tuple[Dict, str]:
        """Convert dict-based format (passthrough with coordinate conversion)."""
        
        rotations = data.get("rotations", [])
        
        if not rotations:
            return self._empty_result("No rotations in data")
        
        # Auto-offset
        calculated_offset = frame_offset
        if auto_offset:
            frames = [r.get("frame", i) for i, r in enumerate(rotations)]
            min_frame = min(frames)
            if min_frame > 0:
                calculated_offset = -min_frame + frame_offset
        
        # Convert rotations
        new_rotations = []
        for rot in rotations:
            pan = rot.get("pan", 0)
            tilt = rot.get("tilt", 0)
            roll = rot.get("roll", 0)
            tx = rot.get("tx", 0)
            ty = rot.get("ty", 0)
            tz = rot.get("tz", 0)
            
            # Coordinate conversion if needed
            if coord_sys == "Z-up" and data.get("coordinate_system") != "Z-up":
                pan, tilt, roll = self._convert_y_to_z_up(pan, tilt, roll)
                tx, ty, tz = tx, tz, -ty
            
            frame_idx = rot.get("frame", len(new_rotations)) + calculated_offset
            
            new_rotations.append({
                "frame": frame_idx,
                "pan": pan,
                "tilt": tilt,
                "roll": roll,
                "pan_deg": np.degrees(pan),
                "tilt_deg": np.degrees(tilt),
                "roll_deg": np.degrees(roll),
                "tx": tx * scale_trans,
                "ty": ty * scale_trans,
                "tz": tz * scale_trans,
            })
        
        new_rotations.sort(key=lambda x: x["frame"])
        
        has_trans = any(
            abs(r["tx"]) > 0.001 or abs(r["ty"]) > 0.001 or abs(r["tz"]) > 0.001
            for r in new_rotations
        )
        
        camera_extrinsics = {
            "num_frames": len(new_rotations),
            "image_width": data.get("image_width", 1920),
            "image_height": data.get("image_height", 1080),
            "source": data.get("source", "COLMAP"),
            "solving_method": data.get("solving_method", "full_6dof"),
            "coordinate_system": coord_sys,
            "units": "radians",
            "has_translation": has_trans,
            "rotations": new_rotations,
        }
        
        status = f"Converted {len(new_rotations)} frames to {coord_sys}"
        return (camera_extrinsics, status)
    
    def _convert_y_to_z_up(self, pan: float, tilt: float, roll: float) -> Tuple[float, float, float]:
        """Convert rotation angles from Y-up to Z-up coordinate system."""
        # In Z-up, the "up" axis is Z instead of Y
        # Pan (horizontal) moves from Y-axis to Z-axis rotation
        new_pan = pan  # Pan stays around vertical axis
        new_tilt = tilt  # Tilt stays around horizontal axis
        new_roll = roll
        return new_pan, new_tilt, new_roll
    
    def _empty_result(self, reason: str) -> Tuple[Dict, str]:
        """Return empty result with error message."""
        print(f"[COLMAPBridge] {reason}")
        return ({
            "num_frames": 1,
            "image_width": 1920,
            "image_height": 1080,
            "source": "COLMAP",
            "solving_method": "none",
            "coordinate_system": "Y-up",
            "units": "radians",
            "has_translation": False,
            "rotations": [{
                "frame": 0,
                "pan": 0, "tilt": 0, "roll": 0,
                "pan_deg": 0, "tilt_deg": 0, "roll_deg": 0,
                "tx": 0, "ty": 0, "tz": 0
            }]
        }, f"Error: {reason}")


# Node registration
NODE_CLASS_MAPPINGS = {
    "COLMAPToExtrinsicsBridge": COLMAPToExtrinsicsBridge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "COLMAPToExtrinsicsBridge": "🔄 COLMAP → Extrinsics Bridge",
}
