"""
Camera Accumulator Node

Chains camera inputs serially to build a CAMERA_LIST for multi-camera
auto-calibration and triangulation.

Usage pattern (Option B - serial accumulator):

  LoadVideo1 → SAM3DBody → 📷 Camera Accumulator ─┐
                                                     │
  LoadVideo2 → SAM3DBody → 📷 Camera Accumulator ─┤→ 🎯 Auto-Calibrator
                                                     │
  LoadVideo3 → SAM3DBody → 📷 Camera Accumulator ─┘
                                    ↑
                              (chained via camera_list input)

Actual wiring:
  Cam1 mesh → Accumulator1 (no chain) ──→ camera_list
                                              ↓
  Cam2 mesh → Accumulator2 (chain=above) ──→ camera_list
                                              ↓
  Cam3 mesh → Accumulator3 (chain=above) ──→ camera_list → Triangulator
"""

import os
import importlib.util
from typing import Dict, Tuple, Optional, List

# Get the directory containing this file
_current_dir = os.path.dirname(os.path.abspath(__file__))
_utils_dir = os.path.join(_current_dir, "utils")


def _load_util_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None


# Try to import logger
try:
    _lib_dir = os.path.dirname(_current_dir)
    _lib_dir = os.path.dirname(_lib_dir)
    _logger_module = _load_util_module("logger", os.path.join(_lib_dir, "lib", "logger.py"))
    if _logger_module:
        log = _logger_module.get_logger("CameraAccumulator")
    else:
        raise ImportError()
except:
    class FallbackLogger:
        def info(self, msg): print(f"[Camera Accumulator] {msg}")
        def warning(self, msg): print(f"[Camera Accumulator] WARNING: {msg}")
        def error(self, msg): print(f"[Camera Accumulator] ERROR: {msg}")
        def debug(self, msg): pass
    log = FallbackLogger()


class CameraAccumulator:
    """
    Accumulates camera inputs into a CAMERA_LIST for multi-camera workflows.
    
    Each node adds one camera (mesh_sequence + optional images) to the list.
    Chain nodes together via the camera_list input to build up 2, 3, 4, 5+
    camera setups.
    
    CAMERA_LIST format:
    {
        "num_cameras": int,
        "cameras": [
            {
                "id": "cam_0",
                "label": str,            # User-provided label
                "mesh_sequence": Dict,    # MESH_SEQUENCE from SAM3DBody
                "images": Tensor|None,    # Optional video frames
            },
            ...
        ]
    }
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE", {
                    "tooltip": "Mesh sequence from SAM3DBody for this camera view"
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Video frames for this camera (for debug visualization)"
                }),
                "camera_list": ("CAMERA_LIST", {
                    "tooltip": "Chain from previous Camera Accumulator (leave empty for first camera)"
                }),
                "camera_label": ("STRING", {
                    "default": "",
                    "tooltip": "Label for this camera (e.g. 'Front', 'Side Left'). Auto-assigned if empty."
                }),
            }
        }
    
    RETURN_TYPES = ("CAMERA_LIST", "STRING")
    RETURN_NAMES = ("camera_list", "camera_info")
    FUNCTION = "accumulate"
    CATEGORY = "SAM3DBody2abc/MultiCamera"
    
    def accumulate(
        self,
        mesh_sequence: Dict,
        images=None,
        camera_list: Optional[Dict] = None,
        camera_label: str = "",
    ) -> Tuple[Dict, str]:
        """
        Add a camera to the list.
        
        If camera_list is None, creates a new list with this as the first camera.
        Otherwise, appends this camera to the existing list.
        """
        
        # Start from existing list or create new
        if camera_list is not None:
            # Deep copy the list structure (but shallow copy the heavy data)
            result = {
                "num_cameras": camera_list["num_cameras"],
                "cameras": list(camera_list["cameras"]),  # shallow copy of list
            }
        else:
            result = {
                "num_cameras": 0,
                "cameras": [],
            }
        
        # Determine camera index and label
        cam_idx = result["num_cameras"]
        
        if camera_label.strip():
            label = camera_label.strip()
        else:
            # Auto-assign: Camera A, Camera B, Camera C, ...
            label = f"Camera {chr(65 + cam_idx)}" if cam_idx < 26 else f"Camera {cam_idx + 1}"
        
        # Get frame count for logging
        frames = mesh_sequence.get("frames", {})
        num_frames = len(frames)
        fps = mesh_sequence.get("fps", 24.0)
        
        # Check for images
        has_images = images is not None
        img_info = ""
        if has_images:
            try:
                img_shape = images.shape
                img_info = f", images: {img_shape[0]} frames {img_shape[2]}x{img_shape[1]}"
            except:
                img_info = ", images: present"
        
        # Add this camera
        camera_entry = {
            "id": f"cam_{cam_idx}",
            "label": label,
            "mesh_sequence": mesh_sequence,
            "images": images,
        }
        
        result["cameras"].append(camera_entry)
        result["num_cameras"] = cam_idx + 1
        
        log.info(f"Added {label} (cam_{cam_idx}): {num_frames} frames @ {fps:.1f}fps{img_info}")
        log.info(f"Camera list now has {result['num_cameras']} camera(s)")
        
        # Build info string
        info_lines = [
            f"=== CAMERA LIST ({result['num_cameras']} cameras) ===",
        ]
        
        for i, cam in enumerate(result["cameras"]):
            cam_frames = cam["mesh_sequence"].get("frames", {})
            cam_fps = cam["mesh_sequence"].get("fps", 24.0)
            has_img = "✓" if cam["images"] is not None else "✗"
            info_lines.append(
                f"  [{cam['id']}] {cam['label']}: "
                f"{len(cam_frames)} frames @ {cam_fps:.1f}fps, "
                f"images: {has_img}"
            )
        
        # Frame count validation
        frame_counts = [len(c["mesh_sequence"].get("frames", {})) for c in result["cameras"]]
        if len(set(frame_counts)) > 1:
            min_f = min(frame_counts)
            max_f = max(frame_counts)
            info_lines.append(f"")
            info_lines.append(f"  ⚠ Frame count mismatch: {min_f}-{max_f}")
            info_lines.append(f"    Will use minimum ({min_f} frames)")
            log.warning(f"Frame count mismatch across cameras: {frame_counts}")
        
        camera_info = "\n".join(info_lines)
        
        return (result, camera_info)


# Node registration
NODE_CLASS_MAPPINGS = {
    "CameraAccumulator": CameraAccumulator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CameraAccumulator": "📷 Camera Accumulator",
}
