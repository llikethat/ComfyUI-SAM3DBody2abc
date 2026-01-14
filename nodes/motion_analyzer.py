# Copyright (c) 2025 SAM3DBody2abc
# SPDX-License-Identifier: MIT
"""
Motion Analyzer Node for SAM3DBody2abc

Analyzes subject motion from SAM3DBody mesh sequence outputs:
- Height estimation from mesh (with user override)
- Pelvis/joint tracking (2D + 3D positions)
- Foot contact detection
- Motion vector debug overlay
- Body world trajectory (Full Skeleton MHR mode)

Compatible with SAM3DBody2abc mesh_sequence format:
{
    "frames": {0: {...}, 1: {...}, ...},
    "fps": 24.0,
    ...
}

Joint Index Reference:
- SAM3DBody outputs 18-joint keypoints (pred_keypoints_2d/3d)
- Also outputs 127-joint full skeleton (pred_joint_coords)
- This module uses 18-joint by default, 127-joint for full skeleton mode
- Body World (Global Trajectory) is joint_coords[0] in MHR rig
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Optional, Tuple, Any


# ============================================================================
# Body World / Global Trajectory Utilities
# ============================================================================

def get_body_world_point(mesh_sequence: Dict, frame_index: int, scale_factor: float = 1.0) -> Optional[np.ndarray]:
    """
    Returns the Body World (Global Trajectory) coordinate for a single frame.
    
    Computed from pred_cam_t (camera translation parameters).
    SAM3DBody outputs mesh in pelvis-centered coords (pelvis at origin).
    The actual character position is encoded in pred_cam_t [tx, ty, tz].
    
    Args:
        mesh_sequence: The mesh sequence dict from SAM3DBody2abc
        frame_index: Frame index to retrieve
        scale_factor: Scale factor to apply (default 1.0)
    
    Returns:
        np.ndarray [X, Y, Z] of body world position, or None if unavailable
    
    Note:
        - X = tx * tz (horizontal position)
        - Y = ty * tz (vertical position)
        - Z = tz (depth from camera)
        - This tracks global character position in camera space
    """
    # Handle both formats: frames dict or params dict
    if "frames" in mesh_sequence:
        frames = mesh_sequence.get("frames", {})
        if frame_index not in frames:
            return None
        frame_data = frames[frame_index]
        camera_t = frame_data.get("pred_cam_t")
    else:
        params = mesh_sequence.get("params", {})
        camera_t_list = params.get("camera_t", [])
        if frame_index >= len(camera_t_list):
            return None
        camera_t = camera_t_list[frame_index]
    
    if camera_t is None:
        return None
    
    # Handle tensor vs numpy
    if hasattr(camera_t, "cpu"):
        camera_t = camera_t.cpu().numpy()
    
    # Handle shape
    if hasattr(camera_t, 'ndim') and camera_t.ndim > 1:
        camera_t = camera_t.flatten()[:3]
    
    # Compute body world from pred_cam_t
    tx, ty, tz = camera_t[0], camera_t[1], camera_t[2]
    
    return np.array([
        tx * tz * scale_factor,  # X = tx * tz
        ty * tz * scale_factor,  # Y = ty * tz
        tz * scale_factor        # Z = depth
    ])


def get_body_world_trajectory(mesh_sequence: Dict, scale_factor: float = 1.0) -> Optional[np.ndarray]:
    """
    Returns the full Body World trajectory for all frames.
    
    Args:
        mesh_sequence: The mesh sequence dict from SAM3DBody2abc
        scale_factor: Scale factor to apply (default 1.0)
    
    Returns:
        np.ndarray [N, 3] of body world positions, or None if unavailable
    """
    # Determine number of frames
    if "frames" in mesh_sequence:
        frames = mesh_sequence.get("frames", {})
        num_frames = len(frames)
        frame_indices = sorted(frames.keys())
    else:
        params = mesh_sequence.get("params", {})
        camera_t_list = params.get("camera_t", [])
        num_frames = len(camera_t_list)
        frame_indices = list(range(num_frames))
    
    if num_frames == 0:
        return None
    
    trajectory = []
    for i in frame_indices:
        point = get_body_world_point(mesh_sequence, i, scale_factor)
        if point is None:
            return None  # If any frame missing, return None
        trajectory.append(point)
    
    return np.array(trajectory)


# ============================================================================
# SAM3DBody 18-Joint Skeleton (Simple Mode)
# This is what pred_keypoints_2d and pred_keypoints_3d use
# ============================================================================
class SAM3DJoints:
    """SAM3DBody keypoint indices for pred_keypoints_2d/3d (COCO-based 70-joint format).
    
    The first 17 joints follow COCO keypoint ordering.
    Note: pred_keypoints_3d has 70 joints total (17 body + face/hands).
    """
    # COCO format (first 17 joints)
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    # Aliases for compatibility
    HEAD = NOSE  # Use nose as head proxy
    PELVIS = 11  # Use left hip as pelvis proxy (center would be between 11 and 12)
    NECK = 0     # Use nose as neck proxy (no explicit neck in COCO)
    
    NUM_JOINTS = 17  # Core body joints
    
    # Skeleton connections for visualization (COCO format)
    CONNECTIONS = [
        # Face
        (LEFT_EYE, NOSE), (RIGHT_EYE, NOSE),
        (LEFT_EAR, LEFT_EYE), (RIGHT_EAR, RIGHT_EYE),
        # Shoulders
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        # Left arm
        (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
        # Right arm
        (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
        # Torso
        (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
        (LEFT_HIP, RIGHT_HIP),
        # Left leg
        (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
        # Right leg
        (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    ]


# ============================================================================
# SMPL-H 127-Joint Skeleton (Full Mode) - Standard SMPL-H ordering
# This is what joint_coords (pred_joint_coords) uses
# ============================================================================
class SMPLHJoints:
    """SMPL-H 127-joint skeleton indices (joint_coords from SAM3DBody).
    
    Standard SMPL-H body joint ordering (first 22 joints).
    Joints 22-126 are hand joints (not used for body analysis).
    """
    # Core body joints (0-21)
    PELVIS = 0
    LEFT_HIP = 1
    RIGHT_HIP = 2
    SPINE1 = 3
    LEFT_KNEE = 4
    RIGHT_KNEE = 5
    SPINE2 = 6
    LEFT_ANKLE = 7
    RIGHT_ANKLE = 8
    SPINE3 = 9
    LEFT_FOOT = 10   # Ball of foot - use for ground contact
    RIGHT_FOOT = 11  # Ball of foot - use for ground contact
    NECK = 12
    LEFT_COLLAR = 13
    RIGHT_COLLAR = 14
    HEAD = 15
    LEFT_SHOULDER = 16
    RIGHT_SHOULDER = 17
    LEFT_ELBOW = 18
    RIGHT_ELBOW = 19
    LEFT_WRIST = 20
    RIGHT_WRIST = 21
    
    NUM_BODY_JOINTS = 22
    NUM_TOTAL_JOINTS = 127
    
    # Skeleton connections for body visualization
    CONNECTIONS = [
        # Spine
        (PELVIS, SPINE1), (SPINE1, SPINE2), (SPINE2, SPINE3), (SPINE3, NECK), (NECK, HEAD),
        # Left leg
        (PELVIS, LEFT_HIP), (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE), (LEFT_ANKLE, LEFT_FOOT),
        # Right leg
        (PELVIS, RIGHT_HIP), (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE), (RIGHT_ANKLE, RIGHT_FOOT),
        # Left arm
        (SPINE3, LEFT_COLLAR), (LEFT_COLLAR, LEFT_SHOULDER), (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
        # Right arm
        (SPINE3, RIGHT_COLLAR), (RIGHT_COLLAR, RIGHT_SHOULDER), (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    ]


# Keep MHRJoints as alias for backward compatibility
MHRJoints = SMPLHJoints


def to_numpy(data):
    """Convert tensor or list to numpy array."""
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    if isinstance(data, np.ndarray):
        return data.copy()
    return np.array(data)


def project_points_to_2d(
    points_3d: np.ndarray,
    focal_length: float,
    cam_t: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """
    Project 3D points to 2D using SAM3DBody's camera model.
    
    SAM3DBody's coordinates are already image-aligned:
    - Positive Y = UP in image space (lower Y pixel value)
    - NO Y negation needed
    
    Args:
        points_3d: (N, 3) array of 3D points
        focal_length: focal length in pixels
        cam_t: camera translation [tx, ty, tz]
        image_width, image_height: image dimensions
        
    Returns:
        points_2d: (N, 2) array of 2D points
    """
    points_3d = np.array(points_3d)
    cam_t = np.array(cam_t).flatten()
    
    # Camera center (principal point)
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    if len(cam_t) < 3:
        # Fallback if cam_t is incomplete
        return np.column_stack([
            np.full(len(points_3d), cx),
            np.full(len(points_3d), cy)
        ])
    
    tx, ty, tz = cam_t[0], cam_t[1], cam_t[2]
    
    # SAM3DBody camera model:
    # Points in camera space = points_3d + cam_t
    # NO Y negation - coordinates are already image-aligned
    X = points_3d[:, 0] + tx
    Y = points_3d[:, 1] + ty
    Z = points_3d[:, 2] + tz
    
    # Avoid division by zero
    Z = np.where(np.abs(Z) < 1e-6, 1e-6, Z)
    
    # Perspective projection
    x_2d = focal_length * X / Z + cx
    y_2d = focal_length * Y / Z + cy
    
    return np.stack([x_2d, y_2d], axis=1)


def estimate_height_from_keypoints(
    keypoints_3d: np.ndarray,
    skeleton_mode: str = "simple",
) -> Dict[str, float]:
    """
    Estimate subject height from 3D keypoints.
    
    Args:
        keypoints_3d: [J, 3] keypoint positions
        skeleton_mode: "simple" (18-joint) or "full" (127-joint)
    
    Returns:
        dict with height measurements
    """
    if skeleton_mode == "simple":
        # SAM3DBody 18-joint
        J = SAM3DJoints
        pelvis = keypoints_3d[J.PELVIS]
        head = keypoints_3d[J.HEAD]
        left_hip = keypoints_3d[J.LEFT_HIP]
        left_knee = keypoints_3d[J.LEFT_KNEE]
        left_ankle = keypoints_3d[J.LEFT_ANKLE]
        right_hip = keypoints_3d[J.RIGHT_HIP]
        right_knee = keypoints_3d[J.RIGHT_KNEE]
        right_ankle = keypoints_3d[J.RIGHT_ANKLE]
    else:
        # SMPL-H 127-joint (use first 22)
        J = SMPLHJoints
        pelvis = keypoints_3d[J.PELVIS]
        head = keypoints_3d[J.HEAD]
        left_hip = keypoints_3d[J.LEFT_HIP]
        left_knee = keypoints_3d[J.LEFT_KNEE]
        left_ankle = keypoints_3d[J.LEFT_ANKLE]
        right_hip = keypoints_3d[J.RIGHT_HIP]
        right_knee = keypoints_3d[J.RIGHT_KNEE]
        right_ankle = keypoints_3d[J.RIGHT_ANKLE]
    
    # Leg length: hip → knee → ankle
    left_upper_leg = np.linalg.norm(left_knee - left_hip)
    left_lower_leg = np.linalg.norm(left_ankle - left_knee)
    left_leg = left_upper_leg + left_lower_leg
    
    right_upper_leg = np.linalg.norm(right_knee - right_hip)
    right_lower_leg = np.linalg.norm(right_ankle - right_knee)
    right_leg = right_upper_leg + right_lower_leg
    
    avg_leg_length = (left_leg + right_leg) / 2
    
    # Torso + head: pelvis → head
    torso_head_length = np.linalg.norm(head - pelvis)
    
    # Estimate full standing height
    # Full height ≈ leg_length + torso_head_length (with overlap adjustment)
    estimated_height = avg_leg_length + torso_head_length * 0.95
    
    return {
        "estimated_height": float(estimated_height),
        "leg_length": float(avg_leg_length),
        "torso_head_length": float(torso_head_length),
        "left_leg_length": float(left_leg),
        "right_leg_length": float(right_leg),
    }


def estimate_height_from_mesh(vertices: np.ndarray) -> Dict[str, float]:
    """
    Estimate height from mesh bounding box.
    """
    mesh_min_y = vertices[:, 1].min()
    mesh_max_y = vertices[:, 1].max()
    mesh_height = mesh_max_y - mesh_min_y
    
    return {
        "mesh_height": float(mesh_height),
        "mesh_min_y": float(mesh_min_y),
        "mesh_max_y": float(mesh_max_y),
    }


def detect_foot_contact(
    keypoints_3d: np.ndarray,
    vertices: np.ndarray,
    skeleton_mode: str = "simple",
    threshold_ratio: float = 0.05,
    frame_idx: int = -1,
    debug: bool = False,
) -> str:
    """
    Detect if feet are in contact with ground.
    
    Args:
        keypoints_3d: [J, 3] keypoint positions (may be pelvis-centered)
        vertices: [V, 3] mesh vertices (in ground-relative coordinates)
        skeleton_mode: "simple" (17-joint COCO) or "full" (127-joint SMPL-H)
        threshold_ratio: Threshold as ratio of mesh height
        frame_idx: Frame index for debug output
        debug: Whether to print debug info
    
    Returns:
        "both", "left", "right", or "none"
    
    Note:
        For accurate ground contact detection, we use MESH VERTICES because:
        - joint_coords is in pelvis-centered space (pelvis at origin)
        - mesh vertices are in ground-relative space (feet at ground level)
        
        We find the lowest mesh vertices on left and right sides of the body
        and compare them to the mesh minimum (ground plane).
    """
    from datetime import datetime, timezone, timedelta
    
    # Ground plane estimate (lowest point of mesh)
    ground_y = vertices[:, 1].min()
    
    # Use mesh vertices to find foot positions (they're in ground-relative coords)
    # Split vertices by X position: negative X = left, positive X = right
    x_values = vertices[:, 0]
    y_values = vertices[:, 1]
    
    # Find lowest Y on each side of the body
    left_mask = x_values <= 0  # Left side of body
    right_mask = x_values > 0   # Right side of body
    
    if left_mask.sum() > 0:
        left_foot_y = y_values[left_mask].min()
    else:
        left_foot_y = ground_y + 1.0  # Fallback: not on ground
        
    if right_mask.sum() > 0:
        right_foot_y = y_values[right_mask].min()
    else:
        right_foot_y = ground_y + 1.0  # Fallback: not on ground
    
    # Calculate mesh height for adaptive threshold
    mesh_height = vertices[:, 1].max() - vertices[:, 1].min()
    
    # Adaptive threshold based on mesh height (5% of body height)
    threshold = mesh_height * threshold_ratio
    
    # Calculate distances from ground
    left_dist = abs(left_foot_y - ground_y)
    right_dist = abs(right_foot_y - ground_y)
    
    # Debug output for first few frames
    if debug and frame_idx >= 0 and frame_idx < 3:
        # Get timestamp in UTC+05:30 (IST)
        ist = timezone(timedelta(hours=5, minutes=30))
        timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
        print(f"[{timestamp}] [Foot Contact DEBUG] Frame {frame_idx}:")
        print(f"  Method: MESH VERTICES (ground-relative coordinates)")
        print(f"  Ground Y (mesh min): {ground_y:.4f}")
        print(f"  Left foot Y (lowest left vertex): {left_foot_y:.4f}, distance: {left_dist:.4f}")
        print(f"  Right foot Y (lowest right vertex): {right_foot_y:.4f}, distance: {right_dist:.4f}")
        print(f"  Mesh height: {mesh_height:.4f}, threshold ({threshold_ratio*100:.0f}%): {threshold:.4f}")
        print(f"  Left contact: {left_dist:.4f} < {threshold:.4f} = {left_dist < threshold}")
        print(f"  Right contact: {right_dist:.4f} < {threshold:.4f} = {right_dist < threshold}")
    
    # Check contact
    left_contact = left_dist < threshold
    right_contact = right_dist < threshold
    
    if left_contact and right_contact:
        return "both"
    elif left_contact:
        return "left"
    elif right_contact:
        return "right"
    else:
        return "none"


def create_motion_debug_overlay(
    images: np.ndarray,
    subject_motion: Dict,
    scale_info: Dict,
    skeleton_mode: str = "simple",
    arrow_scale: float = 10.0,
    show_skeleton: bool = True,
) -> np.ndarray:
    """
    Create debug visualization with joint markers overlaid on video.
    
    Note: Skeleton lines are NOT drawn because joints have independent
    translational data that doesn't match hierarchical bone structure.
    Only individual joint dots are shown for accurate alignment.
    
    Uses pred_keypoints_2d directly for accurate positioning.
    """
    # Convert to uint8 if needed
    if images.dtype == np.float32 or images.dtype == np.float64:
        if images.max() <= 1.0:
            images = (images * 255).astype(np.uint8)
        else:
            images = images.astype(np.uint8)
    
    output = images.copy()
    num_frames = len(images)
    
    # Colors (BGR for OpenCV)
    COLOR_PELVIS = (0, 255, 0)       # Green
    COLOR_VELOCITY = (0, 255, 255)   # Yellow
    COLOR_JOINTS = (255, 128, 128)   # Light blue
    COLOR_HEAD = (0, 255, 255)       # Yellow - for head
    COLOR_HANDS = (255, 0, 255)      # Magenta - for wrists
    COLOR_FEET = (255, 128, 0)       # Orange - for ankles
    COLOR_GROUNDED = (0, 255, 0)     # Green
    COLOR_AIRBORNE = (0, 0, 255)     # Red
    COLOR_PARTIAL = (0, 255, 255)    # Yellow
    COLOR_TEXT = (255, 255, 255)     # White
    
    # Get joint indices based on mode
    if skeleton_mode == "simple":
        pelvis_idx = SAM3DJoints.PELVIS
        head_idx = SAM3DJoints.HEAD
        left_wrist_idx = SAM3DJoints.LEFT_WRIST
        right_wrist_idx = SAM3DJoints.RIGHT_WRIST
        left_ankle_idx = SAM3DJoints.LEFT_ANKLE
        right_ankle_idx = SAM3DJoints.RIGHT_ANKLE
    else:
        pelvis_idx = SMPLHJoints.PELVIS
        head_idx = SMPLHJoints.HEAD
        left_wrist_idx = SMPLHJoints.LEFT_WRIST
        right_wrist_idx = SMPLHJoints.RIGHT_WRIST
        left_ankle_idx = SMPLHJoints.LEFT_ANKLE
        right_ankle_idx = SMPLHJoints.RIGHT_ANKLE
    
    # Special joint indices for coloring
    special_joints = {
        pelvis_idx: (COLOR_PELVIS, 8),      # Green, large
        head_idx: (COLOR_HEAD, 6),          # Yellow, medium
        left_wrist_idx: (COLOR_HANDS, 5),   # Magenta
        right_wrist_idx: (COLOR_HANDS, 5),  # Magenta
        left_ankle_idx: (COLOR_FEET, 5),    # Orange
        right_ankle_idx: (COLOR_FEET, 5),   # Orange
    }
    
    for i in range(num_frames):
        frame = output[i]
        
        # Get 2D joint positions for this frame
        joints_2d = subject_motion.get("joints_2d")
        if joints_2d is not None and i < len(joints_2d) and joints_2d[i] is not None:
            joints_2d_frame = np.array(joints_2d[i])
            
            # Draw joint dots only (no skeleton lines)
            # Lines are removed because joints have independent translational data
            if show_skeleton:
                for j, pt in enumerate(joints_2d_frame):
                    if pt is not None:
                        # Use special color/size for key joints
                        if j in special_joints:
                            color, radius = special_joints[j]
                        else:
                            color = COLOR_JOINTS
                            radius = 4
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), radius, color, -1)
                        # Add black outline for visibility
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), radius, (0, 0, 0), 1)
        
        # Draw pelvis position with larger black outline
        pelvis_2d = subject_motion.get("pelvis_2d")
        if pelvis_2d is not None and i < len(pelvis_2d):
            px, py = pelvis_2d[i]
            cv2.circle(frame, (int(px), int(py)), 10, (0, 0, 0), 2)  # Black outline
        
        # Draw velocity arrow
        velocity_2d = subject_motion.get("velocity_2d")
        if velocity_2d is not None and i > 0 and (i-1) < len(velocity_2d):
            vx, vy = velocity_2d[i-1]
            if pelvis_2d is not None and i < len(pelvis_2d):
                px, py = pelvis_2d[i]
                # Scale velocity for visibility
                end_x = int(px + vx * arrow_scale)
                end_y = int(py + vy * arrow_scale)
                cv2.arrowedLine(frame, (int(px), int(py)), (end_x, end_y),
                               COLOR_VELOCITY, 2, tipLength=0.3)
        
        # Draw info text
        y_offset = 30
        line_height = 25
        
        # Foot contact
        foot_contact = subject_motion.get("foot_contact", [])
        if i < len(foot_contact):
            foot = foot_contact[i]
            foot_colors = {
                "both": COLOR_GROUNDED,
                "left": COLOR_PARTIAL,
                "right": COLOR_PARTIAL,
                "none": COLOR_AIRBORNE,
            }
            foot_color = foot_colors.get(foot, COLOR_TEXT)
            cv2.putText(frame, f"Feet: {foot}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, foot_color, 2)
            y_offset += line_height
        
        # Depth estimate
        depth_estimate = subject_motion.get("depth_estimate", [])
        if i < len(depth_estimate):
            depth = depth_estimate[i]
            cv2.putText(frame, f"Depth: {depth:.2f}m", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
            y_offset += line_height
        
        # Apparent height
        apparent_height = subject_motion.get("apparent_height", [])
        if i < len(apparent_height):
            height_px = apparent_height[i]
            cv2.putText(frame, f"Height: {height_px:.0f}px", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
            y_offset += line_height
        
        # Frame number
        cv2.putText(frame, f"Frame: {i}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    return output


class MotionAnalyzer:
    """
    Analyze subject motion from SAM3DBody mesh sequence.
    
    This node extracts:
    - Subject height (estimated from keypoints, with user override option)
    - Per-frame pelvis/joint positions (2D screen + 3D world)
    - Per-frame velocity (2D and 3D)
    - Foot contact detection (both/left/right/none)
    - Apparent height in pixels (depth indicator)
    
    Uses pred_keypoints_2d directly when available for accurate 2D positions.
    Falls back to projection from pred_keypoints_3d if 2D not available.
    
    Skeleton Modes:
    - "Simple Skeleton" (default): Uses 18-joint keypoints
    - "Full Skeleton": Uses 127-joint SMPL-H skeleton (for future MHR integration)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE", {
                    "tooltip": "Mesh sequence from SAM3DBody2abc Accumulator"
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Original video frames for debug overlay"
                }),
                "camera_extrinsics": ("CAMERA_EXTRINSICS", {
                    "tooltip": "Camera extrinsics from CameraSolver (for camera-compensated trajectory)"
                }),
                "skeleton_mode": (["Simple Skeleton", "Full Skeleton"], {
                    "default": "Simple Skeleton",
                    "tooltip": "Simple: 18-joint keypoints, Full: 127-joint SMPL-H"
                }),
                "subject_height_m": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "Subject height in meters. 0 = auto-estimate (~1.70m)"
                }),
                "reference_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Frame to use for height estimation (should be standing pose)"
                }),
                "default_height_m": ("FLOAT", {
                    "default": 1.70,
                    "min": 0.5,
                    "max": 2.5,
                    "step": 0.01,
                    "tooltip": "Default height assumption when auto-estimating"
                }),
                "foot_contact_threshold": ("FLOAT", {
                    "default": 0.10,
                    "min": 0.01,
                    "max": 0.30,
                    "step": 0.01,
                    "tooltip": "Foot contact threshold as ratio of leg length"
                }),
                "show_debug": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Generate debug overlay with motion vectors"
                }),
                "show_skeleton": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Draw joint markers on debug overlay (no lines, just dots)"
                }),
                "arrow_scale": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 50.0,
                    "tooltip": "Scale factor for velocity arrows in debug view"
                }),
                "sensor_width_mm": ("FLOAT", {
                    "default": 36.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Camera sensor width in mm (36mm = full frame, 23.5mm = APS-C)"
                }),
            }
        }
    
    RETURN_TYPES = ("SUBJECT_MOTION", "SCALE_INFO", "IMAGE", "STRING")
    RETURN_NAMES = ("subject_motion", "scale_info", "debug_overlay", "debug_info")
    FUNCTION = "analyze"
    CATEGORY = "SAM3DBody2abc/Motion"
    
    def _convert_mesh_sequence(self, mesh_sequence: Dict) -> Dict:
        """
        Convert SAM3DBody2abc format to Motion Analyzer format.
        
        Input format (SAM3DBody2abc):
        {
            "frames": {
                0: {"vertices": ..., "joint_coords": ..., "keypoints_2d": ..., ...},
                1: {...},
            },
            "fps": 24.0,
            ...
        }
        
        Output format (Motion Analyzer expected):
        {
            "vertices": [frame0_verts, frame1_verts, ...],
            "params": {
                "keypoints_2d": [...],
                "keypoints_3d": [...],
                "joint_coords": [...],
                "camera_t": [...],
                "focal_length": [...],
            }
        }
        """
        frames_dict = mesh_sequence.get("frames", {})
        
        # Handle case where mesh_sequence is already in the expected format
        if "vertices" in mesh_sequence and isinstance(mesh_sequence.get("vertices"), list):
            print("[Motion Analyzer] Input already in expected format")
            return mesh_sequence
        
        sorted_indices = sorted(frames_dict.keys())
        
        if len(sorted_indices) == 0:
            print("[Motion Analyzer] WARNING: No frames in mesh_sequence!")
            return {"vertices": [], "params": {}}
        
        converted = {
            "vertices": [],
            "params": {
                "keypoints_2d": [],
                "keypoints_3d": [],
                "joint_coords": [],
                "camera_t": [],
                "focal_length": [],
            }
        }
        
        # Debug: show available keys in first frame
        if sorted_indices:
            first_frame = frames_dict[sorted_indices[0]]
            print(f"[Motion Analyzer] First frame keys: {list(first_frame.keys())}")
        
        for idx in sorted_indices:
            frame = frames_dict[idx]
            converted["vertices"].append(frame.get("vertices"))
            converted["params"]["joint_coords"].append(frame.get("joint_coords"))
            
            # Handle pred_cam_t vs camera field
            cam_t = frame.get("pred_cam_t")
            if cam_t is None:
                cam_t = frame.get("camera")
            converted["params"]["camera_t"].append(cam_t)
            
            converted["params"]["focal_length"].append(frame.get("focal_length"))
            
            # Check both naming conventions for keypoints
            # Use explicit None checks because numpy arrays don't work with `or`
            kp2d = frame.get("keypoints_2d")
            if kp2d is None:
                kp2d = frame.get("pred_keypoints_2d")
            
            kp3d = frame.get("keypoints_3d")
            if kp3d is None:
                kp3d = frame.get("pred_keypoints_3d")
            
            converted["params"]["keypoints_2d"].append(kp2d)
            converted["params"]["keypoints_3d"].append(kp3d)
            
            # Debug first frame keypoints
            if idx == sorted_indices[0]:
                print(f"[Motion Analyzer] Frame 0 keypoints_2d source: {'keypoints_2d' if frame.get('keypoints_2d') is not None else ('pred_keypoints_2d' if frame.get('pred_keypoints_2d') is not None else 'None')}")
                print(f"[Motion Analyzer] Frame 0 keypoints_3d source: {'keypoints_3d' if frame.get('keypoints_3d') is not None else ('pred_keypoints_3d' if frame.get('pred_keypoints_3d') is not None else 'None')}")
                if kp2d is not None:
                    kp2d_shape = kp2d.shape if hasattr(kp2d, 'shape') else f"len={len(kp2d)}"
                    print(f"[Motion Analyzer] Frame 0 kp2d shape: {kp2d_shape}")
                if kp3d is not None:
                    kp3d_shape = kp3d.shape if hasattr(kp3d, 'shape') else f"len={len(kp3d)}"
                    print(f"[Motion Analyzer] Frame 0 kp3d shape: {kp3d_shape}")
        
        print(f"[Motion Analyzer] Converted {len(sorted_indices)} frames from SAM3DBody2abc format")
        
        # Log what data is available
        has_kp2d = any(k is not None for k in converted["params"]["keypoints_2d"])
        has_kp3d = any(k is not None for k in converted["params"]["keypoints_3d"])
        has_jc = any(k is not None for k in converted["params"]["joint_coords"])
        print(f"[Motion Analyzer] Data available: keypoints_2d={has_kp2d}, keypoints_3d={has_kp3d}, joint_coords={has_jc}")
        
        return converted
    
    def analyze(
        self,
        mesh_sequence: Dict,
        images: torch.Tensor = None,
        camera_extrinsics: Optional[Dict] = None,
        skeleton_mode: str = "Simple Skeleton",
        subject_height_m: float = 0.0,
        reference_frame: int = 0,
        default_height_m: float = 1.70,
        foot_contact_threshold: float = 0.10,
        show_debug: bool = True,
        show_skeleton: bool = True,
        arrow_scale: float = 10.0,
        sensor_width_mm: float = 36.0,
    ) -> Tuple[Dict, Dict, torch.Tensor, str]:
        """
        Analyze subject motion from mesh sequence.
        
        If camera_extrinsics is provided, also computes camera-compensated trajectory
        (removes camera pan/tilt effects from body_world trajectory).
        """
        print("\n[Motion Analyzer] ========== SUBJECT MOTION ANALYSIS ==========")
        
        # Convert from SAM3DBody2abc format if needed
        mesh_sequence = self._convert_mesh_sequence(mesh_sequence)
        
        # Determine skeleton mode
        use_simple = skeleton_mode == "Simple Skeleton"
        mode_str = "simple" if use_simple else "full"
        print(f"[Motion Analyzer] Skeleton mode: {skeleton_mode}")
        
        # Extract data from mesh sequence
        vertices_list = mesh_sequence.get("vertices", [])
        params = mesh_sequence.get("params", {})
        
        # Get keypoint data
        keypoints_2d_list = params.get("keypoints_2d", [])
        keypoints_3d_list = params.get("keypoints_3d", [])
        joint_coords_list = params.get("joint_coords", [])  # 127-joint fallback
        camera_t_list = params.get("camera_t", [])
        focal_length_list = params.get("focal_length", [])
        
        num_frames = len(vertices_list)
        if num_frames == 0:
            print("[Motion Analyzer] ERROR: No frames in mesh sequence!")
            return ({}, {}, torch.zeros(1, 64, 64, 3), "Error: No frames")
        
        print(f"[Motion Analyzer] Processing {num_frames} frames...")
        
        # Check what keypoint data is available
        has_kp_2d = len(keypoints_2d_list) > 0 and keypoints_2d_list[0] is not None
        has_kp_3d = len(keypoints_3d_list) > 0 and keypoints_3d_list[0] is not None
        has_joint_coords = len(joint_coords_list) > 0 and joint_coords_list[0] is not None
        
        print(f"[Motion Analyzer] Data available: keypoints_2d={has_kp_2d}, keypoints_3d={has_kp_3d}, joint_coords={has_joint_coords}")
        
        # Debug: Print more details about available data
        print(f"[Motion Analyzer] ===== KEYPOINT DATA DEBUG =====")
        print(f"[Motion Analyzer] keypoints_2d_list length: {len(keypoints_2d_list)}")
        if len(keypoints_2d_list) > 0 and keypoints_2d_list[0] is not None:
            kp2d_sample = to_numpy(keypoints_2d_list[0])
            print(f"[Motion Analyzer] keypoints_2d[0] shape: {kp2d_sample.shape if hasattr(kp2d_sample, 'shape') else 'no shape'}")
            if kp2d_sample is not None and len(kp2d_sample) > 0:
                print(f"[Motion Analyzer] keypoints_2d[0] first 3 points: {kp2d_sample[:3] if len(kp2d_sample) >= 3 else kp2d_sample}")
        else:
            print(f"[Motion Analyzer] keypoints_2d[0] is None or empty")
        
        print(f"[Motion Analyzer] keypoints_3d_list length: {len(keypoints_3d_list)}")
        if len(keypoints_3d_list) > 0 and keypoints_3d_list[0] is not None:
            kp3d_sample = to_numpy(keypoints_3d_list[0])
            print(f"[Motion Analyzer] keypoints_3d[0] shape: {kp3d_sample.shape if hasattr(kp3d_sample, 'shape') else 'no shape'}")
        
        print(f"[Motion Analyzer] joint_coords_list length: {len(joint_coords_list)}")
        if len(joint_coords_list) > 0 and joint_coords_list[0] is not None:
            jc_sample = to_numpy(joint_coords_list[0])
            print(f"[Motion Analyzer] joint_coords[0] shape: {jc_sample.shape if hasattr(jc_sample, 'shape') else 'no shape'}")
        print(f"[Motion Analyzer] =================================")
        
        # Decide which 3D keypoints to use
        if use_simple and has_kp_3d:
            kp_source = "keypoints_3d"
            print(f"[Motion Analyzer] Using 18-joint keypoints_3d for analysis")
        elif has_joint_coords:
            kp_source = "joint_coords"
            print(f"[Motion Analyzer] Using 127-joint joint_coords for analysis")
        elif has_kp_3d:
            kp_source = "keypoints_3d"
            print(f"[Motion Analyzer] Fallback to 18-joint keypoints_3d")
        else:
            print("[Motion Analyzer] ERROR: No 3D keypoint data available!")
            return ({}, {}, torch.zeros(1, 64, 64, 3), "Error: No keypoint data")
        
        # Get image size
        image_size = (1920, 1080)  # Default
        if images is not None:
            _, H, W, _ = images.shape
            image_size = (W, H)
            print(f"[Motion Analyzer] Image size: {W}x{H}")
        
        # ===== HEIGHT ESTIMATION =====
        ref_frame = min(reference_frame, num_frames - 1)
        ref_vertices = to_numpy(vertices_list[ref_frame])
        
        # Get reference keypoints for height estimation
        if kp_source == "keypoints_3d":
            ref_keypoints = to_numpy(keypoints_3d_list[ref_frame])
        else:
            ref_keypoints = to_numpy(joint_coords_list[ref_frame])
        
        # Handle shape
        if ref_keypoints is not None and ref_keypoints.ndim == 3:
            ref_keypoints = ref_keypoints.squeeze(0)
        
        # Estimate height from mesh and keypoints
        mesh_height_info = estimate_height_from_mesh(ref_vertices)
        kp_height_info = estimate_height_from_keypoints(ref_keypoints, mode_str if kp_source == "keypoints_3d" else "full")
        
        # Determine actual height
        if subject_height_m > 0:
            actual_height = subject_height_m
            height_source = "user_input"
            print(f"[Motion Analyzer] Using user-specified height: {actual_height:.2f}m")
        else:
            actual_height = default_height_m
            height_source = "auto_estimate"
            print(f"[Motion Analyzer] Using default height: {actual_height:.2f}m")
        
        # Calculate scale factor
        estimated_height = kp_height_info["estimated_height"]
        if estimated_height > 0:
            scale_factor = actual_height / estimated_height
        else:
            scale_factor = 1.0
        
        scale_info = {
            "mesh_height": mesh_height_info["mesh_height"],
            "estimated_height": estimated_height,
            "actual_height_m": actual_height,
            "scale_factor": scale_factor,
            "leg_length": kp_height_info["leg_length"],
            "torso_head_length": kp_height_info["torso_head_length"],
            "height_source": height_source,
            "reference_frame": ref_frame,
            "skeleton_mode": skeleton_mode,
            "keypoint_source": kp_source,
        }
        
        print(f"[Motion Analyzer] Mesh height: {mesh_height_info['mesh_height']:.3f} units")
        print(f"[Motion Analyzer] Estimated height (from joints): {estimated_height:.3f} units")
        print(f"[Motion Analyzer] Scale factor: {scale_factor:.3f}")
        print(f"[Motion Analyzer] Leg length: {kp_height_info['leg_length']:.3f} units")
        print(f"[Motion Analyzer] Torso+head: {kp_height_info['torso_head_length']:.3f} units")
        
        # ===== PER-FRAME ANALYSIS =====
        # Get joint indices based on mode
        if kp_source == "keypoints_3d":
            pelvis_idx = SAM3DJoints.PELVIS
            head_idx = SAM3DJoints.HEAD
            left_ankle_idx = SAM3DJoints.LEFT_ANKLE
            right_ankle_idx = SAM3DJoints.RIGHT_ANKLE
        else:
            pelvis_idx = SMPLHJoints.PELVIS
            head_idx = SMPLHJoints.HEAD
            left_ankle_idx = SMPLHJoints.LEFT_ANKLE
            right_ankle_idx = SMPLHJoints.RIGHT_ANKLE
        
        subject_motion = {
            "pelvis_2d": [],
            "pelvis_3d": [],
            "body_world_3d": [],  # Global trajectory (mesh centroid at ground level)
            "joints_2d": [],
            "joints_3d": [],
            "velocity_2d": [],
            "velocity_3d": [],
            "body_world_velocity": [],  # Trajectory velocity
            "apparent_height": [],
            "depth_estimate": [],
            "foot_contact": [],
            "camera_t": [],
            "focal_length": [],
            "image_size": image_size,
            "num_frames": num_frames,
            "scale_factor": scale_factor,
            "skeleton_mode": skeleton_mode,
            "keypoint_source": kp_source,
        }
        
        for i in range(num_frames):
            # Get frame data
            vertices = to_numpy(vertices_list[i])
            camera_t = to_numpy(camera_t_list[i]) if i < len(camera_t_list) else np.array([0, 0, 5])
            focal = focal_length_list[i] if i < len(focal_length_list) else 1000.0
            
            if isinstance(focal, torch.Tensor):
                focal = focal.cpu().item()
            if camera_t is None:
                camera_t = np.array([0, 0, 5])
            if len(camera_t.shape) > 1:
                camera_t = camera_t.flatten()[:3]
            
            subject_motion["camera_t"].append(camera_t.copy())
            subject_motion["focal_length"].append(float(focal))
            
            # Get 3D keypoints
            if kp_source == "keypoints_3d":
                keypoints_3d = to_numpy(keypoints_3d_list[i]) if i < len(keypoints_3d_list) else None
            else:
                keypoints_3d = to_numpy(joint_coords_list[i]) if i < len(joint_coords_list) else None
            
            if keypoints_3d is None:
                print(f"[Motion Analyzer] Warning: No keypoints for frame {i}")
                keypoints_3d = np.zeros((18 if kp_source == "keypoints_3d" else 127, 3))
            
            # Handle shape
            if keypoints_3d.ndim == 3:
                keypoints_3d = keypoints_3d.squeeze(0)
            
            # Get 2D keypoints (use directly if available, otherwise project)
            if has_kp_2d and i < len(keypoints_2d_list) and keypoints_2d_list[i] is not None:
                keypoints_2d = to_numpy(keypoints_2d_list[i])
                if keypoints_2d.ndim == 3:
                    keypoints_2d = keypoints_2d.squeeze(0)
                # Take only x,y (might have confidence as 3rd column)
                if keypoints_2d.shape[1] >= 2:
                    joints_2d = keypoints_2d[:, :2]
                else:
                    joints_2d = keypoints_2d
                if i == 0:
                    print(f"[Motion Analyzer] Frame 0: Using pred_keypoints_2d DIRECTLY (shape={joints_2d.shape})")
                    print(f"[Motion Analyzer] Frame 0: First 3 2D joints: {joints_2d[:3]}")
            else:
                # Project 3D to 2D
                joints_2d = project_points_to_2d(
                    keypoints_3d, focal, camera_t, image_size[0], image_size[1]
                )
                if i == 0:
                    print(f"[Motion Analyzer] Frame 0: PROJECTING 3D→2D (focal={focal}, cam_t={camera_t})")
                    print(f"[Motion Analyzer] Frame 0: First 3 projected joints: {joints_2d[:3]}")
            
            subject_motion["joints_2d"].append(joints_2d)
            subject_motion["joints_3d"].append(keypoints_3d * scale_factor)
            
            # Pelvis position
            pelvis_3d = keypoints_3d[pelvis_idx] * scale_factor
            pelvis_2d = joints_2d[pelvis_idx]
            subject_motion["pelvis_3d"].append(pelvis_3d.copy())
            subject_motion["pelvis_2d"].append(pelvis_2d.copy())
            
            # Body world position - computed from pred_cam_t
            # SAM3DBody outputs mesh in pelvis-centered coords (pelvis always at origin)
            # The actual character position is encoded in pred_cam_t [tx, ty, tz]
            # For weak perspective projection:
            #   screen_x = focal * tx / tz + cx
            #   screen_y = focal * ty / tz + cy
            # So world position: X = tx * tz, Y = ty * tz, Z = tz (depth)
            # This tracks the character's global trajectory in camera space
            tz = camera_t[2]  # Depth
            
            # RAW trajectory (includes camera effects from focal length variation)
            body_world_3d_raw = np.array([
                camera_t[0] * tz * scale_factor,  # X = tx * tz * scale (left/right)
                camera_t[1] * tz * scale_factor,  # Y = ty * tz * scale (up/down)
                tz * scale_factor                  # Z = depth
            ])
            
            # Store raw trajectory and focal for later compensation
            subject_motion["body_world_3d"].append(body_world_3d_raw.copy())
            
            # Debug output for first 3 frames
            if i < 3:
                from datetime import datetime, timezone, timedelta
                ist = timezone(timedelta(hours=5, minutes=30))
                timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
                print(f"[{timestamp}] [Body World DEBUG] Frame {i}:")
                print(f"  pred_cam_t: tx={camera_t[0]:.4f}, ty={camera_t[1]:.4f}, tz={camera_t[2]:.4f}")
                print(f"  focal_length: {focal:.1f}px")
                print(f"  body_world_3d: X={body_world_3d_raw[0]:.4f}m, Y={body_world_3d_raw[1]:.4f}m, Z={body_world_3d_raw[2]:.4f}m")
            
            # Apparent height (pixels)
            head_2d = joints_2d[head_idx]
            left_ankle_2d = joints_2d[left_ankle_idx]
            right_ankle_2d = joints_2d[right_ankle_idx]
            feet_y = max(left_ankle_2d[1], right_ankle_2d[1])
            apparent_height = abs(feet_y - head_2d[1])
            subject_motion["apparent_height"].append(apparent_height)
            
            # Depth estimate
            depth_m = camera_t[2] * scale_factor
            subject_motion["depth_estimate"].append(depth_m)
            
            # Foot contact detection
            skeleton_mode_str = "simple" if kp_source == "keypoints_3d" else "full"
            foot_contact = detect_foot_contact(
                keypoints_3d, vertices, skeleton_mode_str, foot_contact_threshold,
                frame_idx=i, debug=True
            )
            subject_motion["foot_contact"].append(foot_contact)
        
        # ===== CALCULATE VELOCITIES =====
        pelvis_2d_arr = np.array(subject_motion["pelvis_2d"])
        pelvis_3d_arr = np.array(subject_motion["pelvis_3d"])
        body_world_arr = np.array(subject_motion["body_world_3d"])  # Raw trajectory (camera-space)
        focal_arr = np.array(subject_motion["focal_length"])
        
        # ===== CAMERA-COMPENSATED TRAJECTORY =====
        # Step 1: Compensate for focal length variation (intrinsics)
        # Variable focal length affects the projection - normalize to reference focal
        
        ref_focal = np.mean(focal_arr)  # Use average as reference
        focal_min = np.min(focal_arr)
        focal_max = np.max(focal_arr)
        focal_variation = (focal_max - focal_min) / ref_focal * 100 if ref_focal > 0 else 0
        
        # Convert focal length px to mm
        # Formula: focal_mm = focal_px * sensor_width_mm / image_width_px
        image_width = image_size[0] if image_size else 1920
        ref_focal_mm = ref_focal * sensor_width_mm / image_width if image_width > 0 else 0
        focal_min_mm = focal_min * sensor_width_mm / image_width if image_width > 0 else 0
        focal_max_mm = focal_max * sensor_width_mm / image_width if image_width > 0 else 0
        
        print(f"\n[Motion Analyzer] ----- CAMERA COMPENSATION -----")
        print(f"[Motion Analyzer] Sensor: {sensor_width_mm}mm, Image width: {image_width}px")
        print(f"[Motion Analyzer] Focal length (px): ref={ref_focal:.1f}, min={focal_min:.1f}, max={focal_max:.1f}")
        print(f"[Motion Analyzer] Focal length (mm): ref={ref_focal_mm:.1f}, min={focal_min_mm:.1f}, max={focal_max_mm:.1f}")
        print(f"[Motion Analyzer] Focal variation: {focal_variation:.1f}%")
        
        # Apply focal length compensation
        # When focal length increases (zoom in), same screen motion = less world motion
        # Compensation: world_compensated = world_raw * (ref_focal / frame_focal)
        body_world_focal_compensated = []
        for i, (pos, focal) in enumerate(zip(body_world_arr, focal_arr)):
            if focal > 0:
                focal_ratio = ref_focal / focal
                # Scale X and Y (lateral motion), keep Z (depth) as is
                compensated = np.array([
                    pos[0] * focal_ratio,
                    pos[1] * focal_ratio,
                    pos[2]  # Depth not affected by focal length
                ])
            else:
                compensated = pos.copy()
            body_world_focal_compensated.append(compensated)
        
        body_world_focal_compensated = np.array(body_world_focal_compensated)
        
        # Step 2: Optionally compensate for camera rotation (extrinsics)
        body_world_compensated = body_world_focal_compensated.copy()
        has_extrinsics_compensation = False
        
        if camera_extrinsics is not None:
            try:
                # Get per-frame rotations from camera solver
                per_frame_data = camera_extrinsics.get("per_frame", [])
                
                if len(per_frame_data) >= len(body_world_arr):
                    print(f"[Motion Analyzer] Applying camera extrinsics compensation (pan/tilt)...")
                    body_world_extrinsics_compensated = []
                    
                    # Get reference (first frame) rotation to define world space
                    ref_pan = np.radians(per_frame_data[0].get("pan", 0))
                    ref_tilt = np.radians(per_frame_data[0].get("tilt", 0))
                    
                    for i, pos in enumerate(body_world_focal_compensated):
                        if i < len(per_frame_data):
                            # Get camera rotation for this frame
                            pan = np.radians(per_frame_data[i].get("pan", 0))
                            tilt = np.radians(per_frame_data[i].get("tilt", 0))
                            
                            # Delta rotation from reference frame
                            delta_pan = pan - ref_pan
                            delta_tilt = tilt - ref_tilt
                            
                            # Apply inverse rotation to remove camera motion effect
                            # Rotation around Y axis (pan) in Y-up coordinate system
                            cos_pan = np.cos(-delta_pan)
                            sin_pan = np.sin(-delta_pan)
                            
                            # Rotate X and Z (horizontal plane)
                            x_comp = pos[0] * cos_pan - pos[2] * sin_pan
                            z_comp = pos[0] * sin_pan + pos[2] * cos_pan
                            
                            # For tilt, rotate around X axis
                            cos_tilt = np.cos(-delta_tilt)
                            sin_tilt = np.sin(-delta_tilt)
                            y_comp = pos[1] * cos_tilt - z_comp * sin_tilt
                            z_final = pos[1] * sin_tilt + z_comp * cos_tilt
                            
                            body_world_extrinsics_compensated.append([x_comp, y_comp, z_final])
                        else:
                            body_world_extrinsics_compensated.append(pos.tolist())
                    
                    body_world_compensated = np.array(body_world_extrinsics_compensated)
                    has_extrinsics_compensation = True
                    
                    total_pan = per_frame_data[-1].get("pan", 0) - per_frame_data[0].get("pan", 0)
                    total_tilt = per_frame_data[-1].get("tilt", 0) - per_frame_data[0].get("tilt", 0)
                    print(f"[Motion Analyzer] Camera extrinsics: total pan={total_pan:.2f}°, total tilt={total_tilt:.2f}°")
                else:
                    print(f"[Motion Analyzer] Warning: Not enough extrinsics frames ({len(per_frame_data)} < {len(body_world_arr)})")
            except Exception as e:
                print(f"[Motion Analyzer] Warning: Could not apply extrinsics compensation: {e}")
        
        # Store all trajectory versions
        subject_motion["body_world_3d_raw"] = [pos.tolist() for pos in body_world_arr]  # Original
        subject_motion["body_world_3d_compensated"] = [pos.tolist() for pos in body_world_compensated]  # Fully compensated
        subject_motion["focal_length_ref_px"] = float(ref_focal)
        subject_motion["focal_length_ref_mm"] = float(ref_focal_mm)
        subject_motion["focal_length_min_mm"] = float(focal_min_mm)
        subject_motion["focal_length_max_mm"] = float(focal_max_mm)
        subject_motion["focal_variation_percent"] = float(focal_variation)
        subject_motion["sensor_width_mm"] = float(sensor_width_mm)
        subject_motion["has_extrinsics_compensation"] = has_extrinsics_compensation
        
        if num_frames > 1:
            velocity_2d = np.diff(pelvis_2d_arr, axis=0)
            velocity_3d = np.diff(pelvis_3d_arr, axis=0)
            body_world_velocity = np.diff(body_world_arr, axis=0)
        else:
            velocity_2d = np.zeros((0, 2))
            velocity_3d = np.zeros((0, 3))
            body_world_velocity = np.zeros((0, 3))
        
        subject_motion["velocity_2d"] = velocity_2d.tolist()
        subject_motion["velocity_3d"] = velocity_3d.tolist()
        subject_motion["body_world_velocity"] = body_world_velocity.tolist()
        
        # ===== STATISTICS =====
        avg_velocity_2d = np.mean(np.abs(velocity_2d)) if len(velocity_2d) > 0 else 0
        max_velocity_2d = np.max(np.abs(velocity_2d)) if len(velocity_2d) > 0 else 0
        
        # Body world trajectory statistics (from pred_cam_t)
        body_world_start = body_world_arr[0]
        body_world_end = body_world_arr[-1]
        body_world_displacement = body_world_end - body_world_start
        body_world_total_distance = np.sum(np.linalg.norm(body_world_velocity, axis=1)) if len(body_world_velocity) > 0 else 0
        
        # Calculate velocity and direction
        fps_val = mesh_sequence.get("fps", 24.0)
        if body_world_total_distance > 0 and num_frames > 1:
            # Average speed in m/s
            duration_sec = (num_frames - 1) / fps_val
            avg_speed_ms = body_world_total_distance / duration_sec if duration_sec > 0 else 0
            
            # Direction of movement (XZ plane - horizontal movement)
            # Using displacement vector to get overall direction
            disp_xz = np.array([body_world_displacement[0], body_world_displacement[2]])
            disp_magnitude = np.linalg.norm(disp_xz)
            
            # Normalized direction vector (full 3D)
            disp_3d = np.array(body_world_displacement)
            disp_3d_magnitude = np.linalg.norm(disp_3d)
            if disp_3d_magnitude > 0.01:
                direction_vector = disp_3d / disp_3d_magnitude
            else:
                direction_vector = np.array([0.0, 0.0, 0.0])
            
            if disp_magnitude > 0.01:  # Significant movement
                # Angle in degrees (0° = forward/+Z, 90° = right/+X, -90° = left/-X, 180° = backward/-Z)
                direction_angle = np.degrees(np.arctan2(body_world_displacement[0], body_world_displacement[2]))
                
                # Dominant direction description
                if abs(direction_angle) < 45:
                    direction_desc = "Forward (toward camera)" if body_world_displacement[2] < 0 else "Backward (away from camera)"
                elif abs(direction_angle) > 135:
                    direction_desc = "Backward (away from camera)" if body_world_displacement[2] > 0 else "Forward (toward camera)"
                elif direction_angle > 0:
                    direction_desc = "Right"
                else:
                    direction_desc = "Left"
            else:
                direction_angle = 0
                direction_desc = "Stationary"
        else:
            avg_speed_ms = 0
            direction_angle = 0
            direction_desc = "Stationary"
            direction_vector = np.array([0.0, 0.0, 0.0])
        
        # Store in subject_motion (RAW values)
        subject_motion["avg_speed_ms"] = avg_speed_ms
        subject_motion["direction_angle"] = direction_angle
        subject_motion["direction_desc"] = direction_desc
        subject_motion["direction_vector"] = direction_vector.tolist()  # [X, Y, Z] normalized
        subject_motion["total_distance_m"] = body_world_total_distance
        subject_motion["duration_sec"] = (num_frames - 1) / fps_val if num_frames > 1 else 0
        
        # ===== COMPENSATED TRAJECTORY STATISTICS =====
        body_world_comp_velocity = np.diff(body_world_compensated, axis=0) if len(body_world_compensated) > 1 else np.zeros((0, 3))
        comp_displacement = body_world_compensated[-1] - body_world_compensated[0]
        comp_total_distance = np.sum(np.linalg.norm(body_world_comp_velocity, axis=1)) if len(body_world_comp_velocity) > 0 else 0
        
        if comp_total_distance > 0 and num_frames > 1:
            duration_sec = (num_frames - 1) / fps_val
            comp_avg_speed_ms = comp_total_distance / duration_sec if duration_sec > 0 else 0
            
            comp_disp_3d_mag = np.linalg.norm(comp_displacement)
            if comp_disp_3d_mag > 0.01:
                comp_direction_vector = comp_displacement / comp_disp_3d_mag
            else:
                comp_direction_vector = np.array([0.0, 0.0, 0.0])
            
            comp_disp_xz_mag = np.linalg.norm([comp_displacement[0], comp_displacement[2]])
            if comp_disp_xz_mag > 0.01:
                comp_direction_angle = np.degrees(np.arctan2(comp_displacement[0], comp_displacement[2]))
                if abs(comp_direction_angle) < 45:
                    comp_direction_desc = "Forward" if comp_displacement[2] < 0 else "Backward"
                elif abs(comp_direction_angle) > 135:
                    comp_direction_desc = "Backward" if comp_displacement[2] > 0 else "Forward"
                elif comp_direction_angle > 0:
                    comp_direction_desc = "Right"
                else:
                    comp_direction_desc = "Left"
            else:
                comp_direction_angle = 0
                comp_direction_desc = "Stationary"
        else:
            comp_avg_speed_ms = 0
            comp_direction_angle = 0
            comp_direction_desc = "Stationary"
            comp_direction_vector = np.array([0.0, 0.0, 0.0])
        
        # Store compensated values
        subject_motion["avg_speed_ms_compensated"] = comp_avg_speed_ms
        subject_motion["direction_angle_compensated"] = comp_direction_angle
        subject_motion["direction_desc_compensated"] = comp_direction_desc
        subject_motion["direction_vector_compensated"] = comp_direction_vector.tolist()
        subject_motion["total_distance_m_compensated"] = comp_total_distance
        
        grounded_count = sum(1 for fc in subject_motion["foot_contact"] if fc in ["both", "left", "right"])
        airborne_count = sum(1 for fc in subject_motion["foot_contact"] if fc == "none")
        
        print(f"\n[Motion Analyzer] ----- MOTION STATISTICS -----")
        print(f"[Motion Analyzer] Frames: {num_frames}, Duration: {subject_motion['duration_sec']:.2f}s @ {fps_val}fps")
        print(f"[Motion Analyzer] Avg 2D velocity: {avg_velocity_2d:.2f} px/frame")
        print(f"[Motion Analyzer] Max 2D velocity: {max_velocity_2d:.2f} px/frame")
        print(f"[Motion Analyzer] Grounded frames: {grounded_count} ({100*grounded_count/num_frames:.1f}%)")
        print(f"[Motion Analyzer] Airborne frames: {airborne_count} ({100*airborne_count/num_frames:.1f}%)")
        print(f"[Motion Analyzer] Depth range: {min(subject_motion['depth_estimate']):.2f}m - {max(subject_motion['depth_estimate']):.2f}m")
        
        print(f"[Motion Analyzer] ----- TRAJECTORY (RAW - includes camera effects) -----")
        print(f"[Motion Analyzer] Displacement: X={body_world_displacement[0]:.3f}m, Y={body_world_displacement[1]:.3f}m, Z={body_world_displacement[2]:.3f}m")
        print(f"[Motion Analyzer] Total distance traveled: {body_world_total_distance:.3f}m")
        print(f"[Motion Analyzer] Average speed: {avg_speed_ms:.3f} m/s ({avg_speed_ms * 3.6:.2f} km/h)")
        print(f"[Motion Analyzer] Direction: {direction_desc} ({direction_angle:.1f}°)")
        print(f"[Motion Analyzer] Direction vector (Y-up): X={direction_vector[0]:+.3f}, Y={direction_vector[1]:+.3f}, Z={direction_vector[2]:+.3f}")
        
        print(f"[Motion Analyzer] ----- TRAJECTORY (COMPENSATED - camera effects removed) -----")
        print(f"[Motion Analyzer] Focal compensation: ref={ref_focal_mm:.1f}mm ({ref_focal:.0f}px), variation={focal_variation:.1f}%")
        if has_extrinsics_compensation:
            print(f"[Motion Analyzer] Extrinsics compensation: YES (pan/tilt removed)")
        else:
            print(f"[Motion Analyzer] Extrinsics compensation: NO (connect CameraSolver to enable)")
        print(f"[Motion Analyzer] Displacement: X={comp_displacement[0]:.3f}m, Y={comp_displacement[1]:.3f}m, Z={comp_displacement[2]:.3f}m")
        print(f"[Motion Analyzer] Total distance traveled: {comp_total_distance:.3f}m")
        print(f"[Motion Analyzer] Average speed: {comp_avg_speed_ms:.3f} m/s ({comp_avg_speed_ms * 3.6:.2f} km/h)")
        print(f"[Motion Analyzer] Direction: {comp_direction_desc} ({comp_direction_angle:.1f}°)")
        print(f"[Motion Analyzer] Direction vector (Y-up): X={comp_direction_vector[0]:+.3f}, Y={comp_direction_vector[1]:+.3f}, Z={comp_direction_vector[2]:+.3f}")
        
        # ===== DEBUG INFO STRING =====
        extrinsics_status = "YES" if has_extrinsics_compensation else "NO"
        debug_info = (
            f"=== Motion Analysis Results ===\n"
            f"Frames: {num_frames}, Duration: {subject_motion['duration_sec']:.2f}s @ {fps_val}fps\n"
            f"Skeleton: {skeleton_mode} ({kp_source})\n"
            f"Subject height: {actual_height:.2f}m ({height_source})\n"
            f"Scale factor: {scale_factor:.3f}\n"
            f"Avg 2D velocity: {avg_velocity_2d:.2f} px/frame\n"
            f"Max 2D velocity: {max_velocity_2d:.2f} px/frame\n"
            f"Grounded: {grounded_count}/{num_frames} frames\n"
            f"Depth range: {min(subject_motion['depth_estimate']):.2f}m - {max(subject_motion['depth_estimate']):.2f}m\n"
            f"\n=== Trajectory RAW (camera-space) ===\n"
            f"Displacement: X={body_world_displacement[0]:.3f}m, Y={body_world_displacement[1]:.3f}m, Z={body_world_displacement[2]:.3f}m\n"
            f"Total distance: {body_world_total_distance:.3f}m\n"
            f"Average speed: {avg_speed_ms:.3f} m/s ({avg_speed_ms * 3.6:.2f} km/h)\n"
            f"Direction: {direction_desc} ({direction_angle:.1f}°)\n"
            f"Direction vector: X={direction_vector[0]:+.3f}, Y={direction_vector[1]:+.3f}, Z={direction_vector[2]:+.3f}\n"
            f"\n=== Trajectory COMPENSATED (camera effects removed) ===\n"
            f"Sensor: {sensor_width_mm}mm, Focal ref: {ref_focal_mm:.1f}mm ({ref_focal:.0f}px)\n"
            f"Focal range: {focal_min_mm:.1f}mm - {focal_max_mm:.1f}mm, variation: {focal_variation:.1f}%\n"
            f"Extrinsics compensation: {extrinsics_status}\n"
            f"Displacement: X={comp_displacement[0]:.3f}m, Y={comp_displacement[1]:.3f}m, Z={comp_displacement[2]:.3f}m\n"
            f"Total distance: {comp_total_distance:.3f}m\n"
            f"Average speed: {comp_avg_speed_ms:.3f} m/s ({comp_avg_speed_ms * 3.6:.2f} km/h)\n"
            f"Direction: {comp_direction_desc} ({comp_direction_angle:.1f}°)\n"
            f"Direction vector: X={comp_direction_vector[0]:+.3f}, Y={comp_direction_vector[1]:+.3f}, Z={comp_direction_vector[2]:+.3f}\n"
        )
        
        # ===== DEBUG OVERLAY =====
        if show_debug and images is not None:
            print(f"[Motion Analyzer] Generating debug overlay...")
            
            images_np = images.cpu().numpy() if isinstance(images, torch.Tensor) else images
            
            overlay = create_motion_debug_overlay(
                images_np,
                subject_motion,
                scale_info,
                skeleton_mode=skeleton_mode_str,
                arrow_scale=arrow_scale,
                show_skeleton=show_skeleton,
            )
            
            if overlay.dtype == np.uint8:
                overlay = overlay.astype(np.float32) / 255.0
            debug_overlay = torch.from_numpy(overlay).float()
        else:
            debug_overlay = torch.zeros(1, 64, 64, 3)
        
        print(f"[Motion Analyzer] =============================================\n")
        
        return (subject_motion, scale_info, debug_overlay, debug_info)


class ScaleInfoDisplay:
    """Display scale/height information from Motion Analyzer."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scale_info": ("SCALE_INFO",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "display"
    CATEGORY = "SAM3DBody2abc/Motion"
    OUTPUT_NODE = True
    
    def display(self, scale_info: Dict) -> Tuple[str]:
        info = (
            "=== Scale Info ===\n"
            f"Skeleton: {scale_info.get('skeleton_mode', 'N/A')}\n"
            f"Keypoint source: {scale_info.get('keypoint_source', 'N/A')}\n"
            f"Actual height: {scale_info.get('actual_height_m', 'N/A'):.2f}m\n"
            f"Mesh height: {scale_info.get('mesh_height', 'N/A'):.3f} units\n"
            f"Estimated height: {scale_info.get('estimated_height', 'N/A'):.3f} units\n"
            f"Scale factor: {scale_info.get('scale_factor', 'N/A'):.3f}\n"
            f"Leg length: {scale_info.get('leg_length', 'N/A'):.3f} units\n"
            f"Torso+head: {scale_info.get('torso_head_length', 'N/A'):.3f} units\n"
            f"Source: {scale_info.get('height_source', 'N/A')}\n"
            f"Reference frame: {scale_info.get('reference_frame', 'N/A')}\n"
        )
        print(info)
        return (info,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "MotionAnalyzer": MotionAnalyzer,
    "ScaleInfoDisplay": ScaleInfoDisplay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MotionAnalyzer": "📊 Motion Analyzer",
    "ScaleInfoDisplay": "📏 Scale Info Display",
}
