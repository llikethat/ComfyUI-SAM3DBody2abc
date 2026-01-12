"""
Multi-Camera Triangulation Module for SAM3DBody2abc

Provides nodes for:
- Camera calibration loading (JSON or manual input)
- Multi-camera triangulation (2+ cameras)
- Jitter-free 3D trajectory reconstruction

Usage:
1. Load camera calibration with CameraCalibrationLoader
2. Process video from each camera through SAM3DBody
3. Feed mesh sequences to MultiCameraTriangulator
4. Get jitter-free 3D trajectory

For calibration setup, see examples/calibrations/README.md
"""

from .calibration_loader import CameraCalibrationLoader
from .triangulator import MultiCameraTriangulator

# Combine node mappings
NODE_CLASS_MAPPINGS = {
    **CameraCalibrationLoader.NODE_CLASS_MAPPINGS if hasattr(CameraCalibrationLoader, 'NODE_CLASS_MAPPINGS') else {"CameraCalibrationLoader": CameraCalibrationLoader},
    **MultiCameraTriangulator.NODE_CLASS_MAPPINGS if hasattr(MultiCameraTriangulator, 'NODE_CLASS_MAPPINGS') else {"MultiCameraTriangulator": MultiCameraTriangulator},
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CameraCalibrationLoader": "📷 Camera Calibration Loader",
    "MultiCameraTriangulator": "🔺 Multi-Camera Triangulator",
}

__all__ = [
    "CameraCalibrationLoader",
    "MultiCameraTriangulator",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
