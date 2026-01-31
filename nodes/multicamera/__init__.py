"""
Multi-Camera Module for SAM3DBody2abc

Provides nodes for:
- Camera accumulation (build CAMERA_LIST from multiple views)
- Camera calibration loading (JSON or manual input)
- Camera auto-calibration (from person keypoints)
- Multi-camera triangulation (2+ cameras)
- Silhouette refinement (SMPL-based differentiable rendering)
- Jitter-free 3D trajectory reconstruction

Usage (Option B - Serial Accumulator with Refinement):

  LoadVideo1 → SAM3DBody → 📷 Camera Accumulator ──┐
                                                      │ (chain)
  LoadVideo2 → SAM3DBody → 📷 Camera Accumulator ──┤
                                                      │ (chain)
  LoadVideo3 → SAM3DBody → 📷 Camera Accumulator ──┘
                                                      ↓
                                               CAMERA_LIST
                                              ↓            ↓
                                 🎯 Auto-Calibrator   📷 Calibration Loader
                                              ↓            ↓
                                         CALIBRATION_DATA
                                              ↓
                                 🔺 Multi-Camera Triangulator ← CAMERA_LIST
                                              ↓
                                        TRAJECTORY_3D
                                              ↓
                                 🎭 Silhouette Refiner ← CAMERA_LIST, CALIBRATION_DATA
                                              ↓
                                   REFINED_TRAJECTORY_3D

For calibration setup, see examples/calibrations/README.md
"""

from .camera_accumulator import CameraAccumulator
from .calibration_loader import CameraCalibrationLoader
from .auto_calibrator import CameraAutoCalibrator
from .triangulator import MultiCameraTriangulator
from .silhouette_refiner import SilhouetteRefiner

# Combine node mappings
NODE_CLASS_MAPPINGS = {
    "CameraAccumulator": CameraAccumulator,
    "CameraCalibrationLoader": CameraCalibrationLoader,
    "CameraAutoCalibrator": CameraAutoCalibrator,
    "MultiCameraTriangulator": MultiCameraTriangulator,
    "SilhouetteRefiner": SilhouetteRefiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CameraAccumulator": "📷 Camera Accumulator",
    "CameraCalibrationLoader": "📷 Camera Calibration Loader",
    "CameraAutoCalibrator": "🎯 Camera Auto-Calibrator",
    "MultiCameraTriangulator": "🔺 Multi-Camera Triangulator",
    "SilhouetteRefiner": "🎭 Silhouette Refiner",
}

__all__ = [
    "CameraAccumulator",
    "CameraCalibrationLoader",
    "CameraAutoCalibrator",
    "MultiCameraTriangulator",
    "SilhouetteRefiner",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
