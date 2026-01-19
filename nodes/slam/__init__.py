"""
SLAM Integration for SAM3DBody2abc
==================================

Visual SLAM backends for camera pose estimation from monocular video.

Supported backends:
- DPVO: Deep Patch Visual Odometry (recommended)
- Feature-Based: Fallback using ORB + homography

Usage:
    The SLAM Camera Solver node provides camera poses that can be used
    with Motion Analyzer and FBX Export for world-coordinate motion capture.
"""

from .slam_camera_solver import SLAMCameraSolver, DPVO_AVAILABLE

NODE_CLASS_MAPPINGS = {
    "SLAMCameraSolver": SLAMCameraSolver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SLAMCameraSolver": "📹 SLAM Camera Solver",
}

__all__ = [
    "SLAMCameraSolver",
    "DPVO_AVAILABLE",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
