"""
Verification Overlay Node for SAM3DBody2abc
Projects 3D mesh/skeleton back onto original image to verify correct person tracking.

This helps verify:
- Is the correct person being tracked (matching the mask)?
- Is the geometry aligning properly with the person in the frame?
- Are the camera parameters correct?
"""

import numpy as np
import torch
import cv2
from typing import Dict, Tuple, Any, Optional, List

# Import logger
try:
    from ..lib.logger import log, set_module
    set_module("Verify Overlay")
except ImportError:
    class _FallbackLog:
        def info(self, msg): print(f"[Verify Overlay] {msg}")
        def debug(self, msg): pass
        def warn(self, msg): print(f"[Verify Overlay] WARN: {msg}")
        def error(self, msg): print(f"[Verify Overlay] ERROR: {msg}")
        def progress(self, current, total, task="", interval=10):
            if current == 0 or current == total - 1 or (current + 1) % interval == 0:
                print(f"[Verify Overlay] {task}: {current + 1}/{total}")
    log = _FallbackLog()


def to_numpy(data):
    """Convert tensor to numpy."""
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    if isinstance(data, np.ndarray):
        return data.copy()
    return np.array(data)


def project_points_to_2d(points_3d, focal_length, cam_t, image_width, image_height, cx=None, cy=None):
    """
    Project 3D points to 2D using SAM3DBody's camera model.
    
    SAM3DBody's pred_keypoints_3d are already in a coordinate system where:
    - Positive Y = UP in image space (lower Y pixel value)
    - The projection formula does NOT require Y negation
    
    This should match pred_keypoints_2d when applied to pred_keypoints_3d.
    
    Args:
        points_3d: (N, 3) array of 3D points (vertices or joints)
        focal_length: focal length in pixels
        cam_t: camera translation [tx, ty, tz]
        image_width, image_height: image dimensions
        cx, cy: optional custom principal point (default: image center)
        
    Returns:
        points_2d: (N, 2) array of 2D points
    """
    points_3d = np.array(points_3d)
    cam_t = np.array(cam_t).flatten()
    
    # Camera center (principal point) - use custom if provided
    if cx is None:
        cx = image_width / 2.0
    if cy is None:
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
    # NO Y negation needed - SAM3DBody's coordinates are already image-aligned
    X = points_3d[:, 0] + tx
    Y = points_3d[:, 1] + ty  # NO negation!
    Z = points_3d[:, 2] + tz
    
    # Avoid division by zero
    Z = np.where(np.abs(Z) < 1e-6, 1e-6, Z)
    
    # Perspective projection
    x_2d = focal_length * X / Z + cx
    y_2d = focal_length * Y / Z + cy
    
    return np.stack([x_2d, y_2d], axis=1)


# Define key body joint indices for MHR model visualization
# Based on visual inspection of joint labels in overlay output
MHR_KEY_JOINTS = {
    # Head joints (0-4)
    'head': 0,
    'head_1': 1,
    'head_2': 2, 
    'head_3': 3,
    'neck': 4,
    # Upper body (5-8)
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    # Hips (9-10)
    'left_hip': 9,
    'right_hip': 10,
    # Left leg (11-14)
    'left_knee': 11,
    'left_ankle': 12,
    'left_heel': 13,
    'left_toe': 14,
    # Right leg (15-19)
    'right_knee': 15,
    'right_ankle': 16,
    'right_heel': 17,
    'right_toe_1': 18,
    'right_toe_2': 19,
    # Wrists (20-21)
    'left_wrist': 20,
    'right_wrist': 21,
}

# Fallback skeleton connections if joint_parents not available
# Based on MHR joint ordering (0-21 body joints)
FALLBACK_SKELETON_CONNECTIONS = [
    # Head to neck
    (0, 4),
    # Shoulders from neck
    (4, 5), (4, 6),
    # Arms
    (5, 7), (7, 20),  # Left arm: shoulder -> elbow -> wrist
    (6, 8), (8, 21),  # Right arm: shoulder -> elbow -> wrist
    # Hips from neck (spine simplified)
    (4, 9), (4, 10),
    # Left leg
    (9, 11), (11, 12), (12, 14),  # hip -> knee -> ankle -> toe
    # Right leg
    (10, 15), (15, 16), (16, 18),  # hip -> knee -> ankle -> toe
]


def get_skeleton_connections(joint_parents):
    """
    Build skeleton connections from joint_parents array.
    joint_parents[i] = parent index of joint i (-1 for root)
    """
    if joint_parents is None:
        return FALLBACK_SKELETON_CONNECTIONS
    
    connections = []
    for i, parent in enumerate(joint_parents):
        if parent >= 0:
            connections.append((int(parent), i))
    return connections


class VerifyOverlay:
    """
    Project 3D mesh/skeleton onto original image for verification.
    
    This helps verify that SAM3DBody is tracking the correct person
    (the one you masked) and not mixing with other people in the frame.
    
    Outputs an overlay image showing:
    - Joint positions as colored circles
    - Optionally: skeleton connections as lines
    - Optionally: mesh wireframe
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Original input image"
                }),
                "mesh_data": ("SAM3D_OUTPUT", {
                    "tooltip": "mesh_data from SAM3DBody Process"
                }),
            },
            "optional": {
                "show_joints": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Draw joint positions as circles"
                }),
                "show_skeleton": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Draw skeleton connections"
                }),
                "show_mesh": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Draw mesh"
                }),
                "mesh_render_mode": (["Wireframe", "Solid"], {
                    "default": "Wireframe",
                    "tooltip": "Wireframe: edge lines only. Solid: filled triangles."
                }),
                "joint_radius": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "tooltip": "Radius of joint circles"
                }),
                "line_thickness": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Thickness of skeleton/mesh lines"
                }),
                "joint_color": (["red", "green", "blue", "yellow", "cyan", "magenta", "white"], {
                    "default": "green",
                    "tooltip": "Color for joint markers"
                }),
                "skeleton_color": (["red", "green", "blue", "yellow", "cyan", "magenta", "white"], {
                    "default": "cyan",
                    "tooltip": "Color for skeleton lines"
                }),
                "mesh_color": (["red", "green", "blue", "yellow", "cyan", "magenta", "white"], {
                    "default": "yellow",
                    "tooltip": "Color for mesh"
                }),
                "opacity": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Opacity of overlay"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("overlay_image", "info")
    FUNCTION = "create_overlay"
    CATEGORY = "SAM3DBody2abc/Debug"
    
    def _get_color(self, color_name):
        """Convert color name to BGR tuple."""
        colors = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
            "cyan": (255, 255, 0),
            "magenta": (255, 0, 255),
            "white": (255, 255, 255),
        }
        return colors.get(color_name, (0, 255, 0))
    
    def create_overlay(
        self,
        image,
        mesh_data: Dict,
        show_joints: bool = True,
        show_skeleton: bool = True,
        show_mesh: bool = False,
        mesh_render_mode: str = "Wireframe",
        joint_radius: int = 5,
        line_thickness: int = 2,
        joint_color: str = "green",
        skeleton_color: str = "cyan",
        mesh_color: str = "yellow",
        opacity: float = 0.7,
    ) -> Tuple[Any, str]:
        """Create verification overlay."""
        
        # Convert ComfyUI image to numpy (H, W, C) float 0-1
        if isinstance(image, torch.Tensor):
            img_np = image.cpu().numpy()
        else:
            img_np = np.array(image)
        
        # Handle batch dimension
        if img_np.ndim == 4:
            img_np = img_np[0]
        
        # Convert to uint8 BGR for OpenCV
        img_bgr = (img_np * 255).astype(np.uint8)
        if img_bgr.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        
        h, w = img_bgr.shape[:2]
        
        # Create overlay layer
        overlay = img_bgr.copy()
        
        # Debug: print mesh_data keys
        log.info(f"mesh_data keys: {list(mesh_data.keys())}")
        
        # Get projection parameters
        focal_length = mesh_data.get("focal_length")
        cam_t = to_numpy(mesh_data.get("camera"))
        
        # Also check for 2D keypoints (more reliable if available)
        keypoints_2d = to_numpy(mesh_data.get("pred_keypoints_2d"))
        
        if keypoints_2d is not None:
            log.info(f"Using pred_keypoints_2d directly (shape: {keypoints_2d.shape})")
        
        if focal_length is None or cam_t is None:
            if keypoints_2d is None:
                return (image, "Error: Missing camera parameters (focal_length or camera) and no pred_keypoints_2d")
        
        # Handle focal length format
        if focal_length is not None:
            if isinstance(focal_length, (list, tuple, np.ndarray)):
                focal_length = float(focal_length[0]) if len(focal_length) > 0 else float(focal_length)
        
        info_parts = [f"Image: {w}x{h}"]
        if focal_length is not None:
            info_parts.append(f"Focal: {focal_length:.1f}px")
        if cam_t is not None:
            info_parts.append(f"cam_t: [{cam_t[0]:.2f}, {cam_t[1]:.2f}, {cam_t[2]:.2f}]")
        
        # Get colors
        joint_bgr = self._get_color(joint_color)
        skeleton_bgr = self._get_color(skeleton_color)
        mesh_bgr = self._get_color(mesh_color)
        
        # Project and draw joints
        joint_coords = to_numpy(mesh_data.get("joint_coords"))
        joints_2d = None
        
        if show_joints or show_skeleton:
            # Prefer 2D keypoints if available (already in image coordinates)
            if keypoints_2d is not None:
                # pred_keypoints_2d might be (N, 2) or (N, 3) with confidence
                if keypoints_2d.ndim == 2:
                    joints_2d = keypoints_2d[:, :2] if keypoints_2d.shape[1] >= 2 else keypoints_2d
                else:
                    joints_2d = keypoints_2d
                info_parts.append(f"Keypoints2D: {len(joints_2d)}")
            elif joint_coords is not None and focal_length is not None and cam_t is not None:
                # Fall back to projecting 3D joints
                joints_2d = project_points_to_2d(joint_coords, focal_length, cam_t, w, h)
                info_parts.append(f"Joints3D→2D: {len(joint_coords)}")
                
                # Debug: print some joint positions
                log.info(f"First 5 joints 3D: {joint_coords[:5]}")
                log.info(f"First 5 joints 2D: {joints_2d[:5]}")
                log.info(f"cam_t: {cam_t}, focal: {focal_length}")
        
        if joints_2d is not None:
            if show_joints:
                for i, pt in enumerate(joints_2d):
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(overlay, (x, y), joint_radius, joint_bgr, -1)
                        # Draw joint index for first few joints
                        if i < 20:
                            cv2.putText(overlay, str(i), (x+5, y-5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            if show_skeleton and len(joints_2d) > 15:
                # Draw skeleton connections
                for (i, j) in SKELETON_CONNECTIONS:
                    if i < len(joints_2d) and j < len(joints_2d):
                        pt1 = (int(joints_2d[i][0]), int(joints_2d[i][1]))
                        pt2 = (int(joints_2d[j][0]), int(joints_2d[j][1]))
                        # Check if points are within image
                        if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                            0 <= pt2[0] < w and 0 <= pt2[1] < h):
                            cv2.line(overlay, pt1, pt2, skeleton_bgr, line_thickness)
        
        # Project and draw mesh (optional)
        if show_mesh:
            vertices = to_numpy(mesh_data.get("vertices"))
            faces = to_numpy(mesh_data.get("faces"))
            
            if vertices is not None and faces is not None and focal_length is not None and cam_t is not None:
                verts_2d = project_points_to_2d(vertices, focal_length, cam_t, w, h)
                info_parts.append(f"Mesh: {len(vertices)} verts, {len(faces)} faces ({mesh_render_mode})")
                
                if mesh_render_mode == "Solid":
                    # Solid rendering: fill triangles
                    # Sort faces by depth (average Z) for proper occlusion
                    face_depths = []
                    for face_idx, face in enumerate(faces):
                        avg_z = np.mean([vertices[face[k]][2] for k in range(3)])
                        face_depths.append((avg_z, face_idx))
                    face_depths.sort(reverse=True)  # Back to front
                    
                    # Draw filled triangles
                    for _, face_idx in face_depths:
                        face = faces[face_idx]
                        pts = np.array([
                            [int(verts_2d[face[0]][0]), int(verts_2d[face[0]][1])],
                            [int(verts_2d[face[1]][0]), int(verts_2d[face[1]][1])],
                            [int(verts_2d[face[2]][0]), int(verts_2d[face[2]][1])],
                        ], dtype=np.int32)
                        
                        # Check if triangle is within image bounds
                        if np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] < w) and \
                           np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] < h):
                            cv2.fillPoly(overlay, [pts], mesh_bgr)
                else:
                    # Wireframe rendering: draw edges
                    step = max(1, len(faces) // 500)  # Limit to ~500 edges for performance
                    for face_idx in range(0, len(faces), step):
                        face = faces[face_idx]
                        for k in range(3):
                            i, j = face[k], face[(k + 1) % 3]
                            pt1 = (int(verts_2d[i][0]), int(verts_2d[i][1]))
                            pt2 = (int(verts_2d[j][0]), int(verts_2d[j][1]))
                            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                                cv2.line(overlay, pt1, pt2, mesh_bgr, 1)
        
        # Blend overlay with original
        result = cv2.addWeighted(overlay, opacity, img_bgr, 1 - opacity, 0)
        
        # Convert back to RGB float for ComfyUI
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result_float = result_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        result_tensor = torch.from_numpy(result_float).unsqueeze(0)
        
        info = " | ".join(info_parts)
        log.info(f"{info}")
        
        return (result_tensor, info)


class VerifyOverlayBatch:
    """
    Create verification overlay for ALL frames in a sequence.
    Outputs a video/batch of overlay images.
    
    Can compare SAM3DBody intrinsics vs MoGe2 intrinsics to verify camera calibration.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "All video frames (batch)"
                }),
                "mesh_sequence": ("MESH_SEQUENCE", {
                    "tooltip": "Accumulated mesh sequence"
                }),
            },
            "optional": {
                "camera_intrinsics": ("CAMERA_INTRINSICS", {
                    "tooltip": "Camera intrinsics from MoGe2 (optional - for comparison)"
                }),
                "intrinsics_source": (["SAM3DBody (Default)", "MoGe2", "Compare Both"], {
                    "default": "SAM3DBody (Default)",
                    "tooltip": "Which camera intrinsics to use for projection"
                }),
                "show_joints": ("BOOLEAN", {"default": True}),
                "show_skeleton": ("BOOLEAN", {"default": True}),
                "show_mesh": ("BOOLEAN", {"default": False}),
                "mesh_render_mode": (["Wireframe", "Solid"], {
                    "default": "Wireframe",
                    "tooltip": "Wireframe: edge lines only. Solid: filled triangles."
                }),
                "mesh_alignment": (["Auto (Match Joints)", "None (Direct Projection)"], {
                    "default": "None (Direct Projection)",
                    "tooltip": "Auto aligns mesh to joints (can cause jitter). None uses direct projection."
                }),
                "joint_radius": ("INT", {"default": 5, "min": 1, "max": 20}),
                "line_thickness": ("INT", {"default": 2, "min": 1, "max": 10}),
                "joint_color": (["red", "green", "blue", "yellow", "cyan", "magenta", "white"], {"default": "green"}),
                "skeleton_color": (["red", "green", "blue", "yellow", "cyan", "magenta", "white"], {"default": "cyan"}),
                "mesh_color": (["red", "green", "blue", "yellow", "cyan", "magenta", "white"], {"default": "yellow"}),
                "opacity": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.1}),
                "show_intrinsics_info": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show camera intrinsics info on overlay"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("overlay_images", "info")
    FUNCTION = "create_overlay_batch"
    CATEGORY = "SAM3DBody2abc/Debug"
    
    def _get_color(self, color_name):
        """Convert color name to BGR tuple."""
        colors = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
            "cyan": (255, 255, 0),
            "magenta": (255, 0, 255),
            "white": (255, 255, 255),
        }
        return colors.get(color_name, (0, 255, 0))
    
    def create_overlay_batch(
        self,
        images: torch.Tensor,
        mesh_sequence: Dict,
        camera_intrinsics: Optional[Dict] = None,
        intrinsics_source: str = "SAM3DBody (Default)",
        show_joints: bool = True,
        show_skeleton: bool = True,
        show_mesh: bool = False,
        mesh_render_mode: str = "Wireframe",
        mesh_alignment: str = "None (Direct Projection)",
        joint_radius: int = 5,
        line_thickness: int = 2,
        joint_color: str = "green",
        skeleton_color: str = "cyan",
        mesh_color: str = "yellow",
        opacity: float = 0.7,
        show_intrinsics_info: bool = True,
    ) -> Tuple[Any, str]:
        """Create overlay for all frames with optional intrinsics comparison."""
        
        frames = mesh_sequence.get("frames", {})
        faces = mesh_sequence.get("faces")
        joint_parents = mesh_sequence.get("joint_parents")
        
        if not frames:
            return (images, "Error: No frames in mesh_sequence")
        
        # Build skeleton connections from joint_parents
        skeleton_connections = get_skeleton_connections(joint_parents)
        if joint_parents is not None:
            log.info(f"Using {len(skeleton_connections)} skeleton connections from joint_parents")
        else:
            log.info(f"Using fallback skeleton connections")
        
        # Get MoGe2 intrinsics if available
        moge_focal = None
        moge_cx = None
        moge_cy = None
        if camera_intrinsics:
            # Debug: print all keys to find the right ones
            log.info(f"camera_intrinsics keys: {list(camera_intrinsics.keys())}")
            
            # Try multiple possible key names (different MoGe2 packages use different keys)
            moge_focal = (
                camera_intrinsics.get("focal_length_px") or
                camera_intrinsics.get("focal_length") or
                camera_intrinsics.get("focal") or
                camera_intrinsics.get("fx")
            )
            moge_cx = (
                camera_intrinsics.get("principal_point_x") or
                camera_intrinsics.get("cx") or
                camera_intrinsics.get("principal_x")
            )
            moge_cy = (
                camera_intrinsics.get("principal_point_y") or
                camera_intrinsics.get("cy") or
                camera_intrinsics.get("principal_y")
            )
            
            if moge_focal is not None:
                cx_str = f"{moge_cx:.1f}" if moge_cx is not None else "N/A"
                cy_str = f"{moge_cy:.1f}" if moge_cy is not None else "N/A"
                log.info(f"MoGe2 intrinsics: focal={moge_focal:.1f}px, cx={cx_str}, cy={cy_str}")
            else:
                log.info(f"MoGe2 intrinsics: Connected but focal_length key not found")
        
        # Get colors
        joint_bgr = self._get_color(joint_color)
        skeleton_bgr = self._get_color(skeleton_color)
        mesh_bgr = self._get_color(mesh_color)
        
        # Secondary color for comparison mode (slightly different shade)
        compare_joint_bgr = (0, 128, 255)  # Orange for MoGe2
        compare_skeleton_bgr = (255, 128, 0)  # Light blue for MoGe2
        
        # Convert images to numpy
        if isinstance(images, torch.Tensor):
            images_np = images.cpu().numpy()
        else:
            images_np = np.array(images)
        
        num_images = images_np.shape[0]
        h, w = images_np.shape[1], images_np.shape[2]
        
        log.info(f"Processing {num_images} images, {len(frames)} frames in sequence")
        log.info(f"Intrinsics source: {intrinsics_source}")
        
        # Get sorted frame indices
        sorted_frame_indices = sorted(frames.keys())
        
        result_frames = []
        
        for img_idx in range(num_images):
            img_np = images_np[img_idx]
            img_bgr = (img_np * 255).astype(np.uint8)
            if img_bgr.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
            
            overlay = img_bgr.copy()
            
            # Find corresponding frame in mesh_sequence
            # Try to match by index or use closest available
            frame_idx = img_idx
            if frame_idx not in frames:
                # Find closest frame index
                if sorted_frame_indices:
                    frame_idx = min(sorted_frame_indices, key=lambda x: abs(x - img_idx))
            
            if frame_idx in frames:
                frame = frames[frame_idx]
                
                # Debug first frame
                if img_idx == 0:
                    log.info(f"Frame keys: {list(frame.keys())}")
                
                # Draw bounding box if available (helps debug detection)
                bbox = frame.get("bbox")
                if bbox is not None:
                    bbox = np.array(bbox).flatten()
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow bbox
                        cv2.putText(overlay, "Detection", (x1, y1-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        if img_idx == 0:
                            log.info(f"Detection bbox: [{x1}, {y1}, {x2}, {y2}]")
                elif img_idx == 0:
                    log.info(f"No bbox in frame data")
                
                # Get 2D keypoints if available (most reliable)
                keypoints_2d = frame.get("pred_keypoints_2d")
                keypoints_3d = frame.get("pred_keypoints_3d")  # Same 70 joints as keypoints_2d
                joint_coords = frame.get("joint_coords")  # 127 full skeleton joints
                sam3d_focal_length = frame.get("focal_length")
                cam_t = frame.get("pred_cam_t")
                if cam_t is None:
                    cam_t = frame.get("camera")
                
                # Determine which focal length to use based on intrinsics_source
                focal_length = sam3d_focal_length  # Default
                active_source = "SAM3DBody"
                cx, cy = w / 2.0, h / 2.0  # Default principal point
                
                if intrinsics_source == "MoGe2" and moge_focal is not None:
                    focal_length = moge_focal
                    active_source = "MoGe2"
                    if moge_cx is not None:
                        cx = moge_cx
                    if moge_cy is not None:
                        cy = moge_cy
                elif intrinsics_source == "Compare Both":
                    # Use SAM3DBody as primary, will draw MoGe2 comparison below
                    active_source = "Compare"
                
                # Log intrinsics comparison on first frame
                if img_idx == 0 and sam3d_focal_length is not None:
                    log.info(f"========== INTRINSICS COMPARISON ==========")
                    log.info(f"SAM3DBody focal: {sam3d_focal_length:.1f}px")
                    if moge_focal is not None:
                        focal_diff = moge_focal - sam3d_focal_length
                        focal_diff_pct = 100 * focal_diff / sam3d_focal_length
                        log.info(f"MoGe2 focal: {moge_focal:.1f}px (diff: {focal_diff:+.1f}px, {focal_diff_pct:+.1f}%)")
                        if moge_cx is not None and moge_cy is not None:
                            log.info(f"MoGe2 principal point: ({moge_cx:.1f}, {moge_cy:.1f})")
                            log.info(f"Image center: ({w/2:.1f}, {h/2:.1f})")
                    else:
                        log.info(f"MoGe2 intrinsics: Not connected")
                    log.info(f"Using: {active_source}")
                    log.info(f"============================================\n")
                
                joints_2d = None
                joints_2d_moge = None  # For comparison mode
                
                # DEBUG: Compare ground truth vs our projection (frame 0 only)
                if img_idx == 0 and keypoints_2d is not None and focal_length is not None and cam_t is not None:
                    
                    cam_t_np = np.array(cam_t).flatten()
                    tx, ty, tz = cam_t_np[0], cam_t_np[1], cam_t_np[2]
                    
                    gt_2d = np.array(keypoints_2d)
                    if gt_2d.ndim == 2:
                        gt_2d = gt_2d[:, :2]
                    
                    # Check what 3D data is available
                    kp_3d = None
                    source_name = None
                    
                    if keypoints_3d is not None:
                        kp_3d_raw = np.array(keypoints_3d)
                        # Handle various shapes (squeeze if needed)
                        if kp_3d_raw.ndim == 3 and kp_3d_raw.shape[0] == 1:
                            kp_3d_raw = kp_3d_raw.squeeze(0)
                        if kp_3d_raw.ndim == 2 and kp_3d_raw.shape[1] == 3:
                            kp_3d = kp_3d_raw
                            source_name = f"pred_keypoints_3d ({kp_3d.shape[0]} joints - SAME as ground truth)"
                        else:
                            log.debug(f"pred_keypoints_3d has unexpected shape: {kp_3d_raw.shape}")
                    
                    if kp_3d is None and joint_coords is not None:
                        kp_3d_raw = np.array(joint_coords)
                        if kp_3d_raw.ndim == 2 and kp_3d_raw.shape[1] == 3:
                            kp_3d = kp_3d_raw
                            source_name = f"joint_coords ({kp_3d.shape[0]} joints - DIFFERENT from ground truth!)"
                    
                    
                    if kp_3d is not None:
                        our_2d = project_points_to_2d(kp_3d, focal_length, cam_t_np, w, h)
                        
                        
                        num_compare = min(10, len(gt_2d), len(our_2d))
                        total_dx, total_dy = 0.0, 0.0
                        for i in range(num_compare):
                            gt_x, gt_y = gt_2d[i][0], gt_2d[i][1]
                            our_x, our_y = our_2d[i][0], our_2d[i][1]
                            dx, dy = our_x - gt_x, our_y - gt_y
                            total_dx += dx
                            total_dy += dy
                        
                        avg_dx, avg_dy = total_dx / num_compare, total_dy / num_compare
                        
                        if abs(avg_dx) < 5 and abs(avg_dy) < 5:
                            log.debug("Projection formula appears CORRECT!")
                        else:
                            log.debug(f"Large offset detected: dx={avg_dx:.1f}px, dy={avg_dy:.1f}px")
                    else:
                        log.debug("Cannot compare - no 3D keypoints available")
                
                if keypoints_2d is not None:
                    # Use 2D keypoints directly
                    keypoints_2d = np.array(keypoints_2d)
                    if keypoints_2d.ndim == 2:
                        joints_2d = keypoints_2d[:, :2] if keypoints_2d.shape[1] >= 2 else keypoints_2d
                    if img_idx == 0:
                        log.info(f"Using pred_keypoints_2d: {joints_2d.shape}")
                
                elif joint_coords is not None and focal_length is not None and cam_t is not None:
                    # Project 3D to 2D
                    joint_coords = np.array(joint_coords)
                    cam_t = np.array(cam_t)
                    joints_2d = project_points_to_2d(joint_coords, focal_length, cam_t, w, h)
                    if img_idx == 0:
                        log.info(f"Projecting 3D joints: focal={focal_length}, cam_t={cam_t}")
                
                # Draw joints
                if joints_2d is not None:
                    if show_joints:
                        for i, pt in enumerate(joints_2d):
                            x, y = int(pt[0]), int(pt[1])
                            if 0 <= x < w and 0 <= y < h:
                                cv2.circle(overlay, (x, y), joint_radius, joint_bgr, -1)
                                if i < 20:
                                    cv2.putText(overlay, str(i), (x+5, y-5), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    
                    if show_skeleton and len(joints_2d) > 15:
                        for (i, j) in skeleton_connections:
                            if i < len(joints_2d) and j < len(joints_2d):
                                pt1 = (int(joints_2d[i][0]), int(joints_2d[i][1]))
                                pt2 = (int(joints_2d[j][0]), int(joints_2d[j][1]))
                                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                                    cv2.line(overlay, pt1, pt2, skeleton_bgr, line_thickness)
                
                # Draw mesh wireframe if requested
                if show_mesh:
                    vertices = frame.get("vertices")
                    joint_coords_3d = frame.get("joint_coords")  # 127 joints
                    
                    if vertices is not None and faces is not None and focal_length is not None and cam_t is not None:
                        vertices = np.array(vertices)
                        cam_t_np = np.array(cam_t)
                        
                        # Project mesh vertices with selected intrinsics
                        verts_2d = project_points_to_2d(vertices, focal_length, cam_t_np, w, h, cx, cy)
                        
                        # In Compare Both mode, also project with MoGe2 intrinsics
                        verts_2d_moge = None
                        if intrinsics_source == "Compare Both" and moge_focal is not None:
                            moge_cx_use = moge_cx if moge_cx is not None else w / 2.0
                            moge_cy_use = moge_cy if moge_cy is not None else h / 2.0
                            verts_2d_moge = project_points_to_2d(vertices, moge_focal, cam_t_np, w, h, moge_cx_use, moge_cy_use)
                        
                        # Compute offset to align mesh with detected keypoints (optional)
                        offset_x, offset_y = 0.0, 0.0
                        
                        if mesh_alignment == "Auto (Match Joints)" and joints_2d is not None:
                            # Method 1: Use centroid of joints vs projected mesh
                            # The pred_keypoints_2d are the ground truth positions
                            # The projected mesh should align with them
                            # NOTE: This can cause jitter if joints_2d is noisy
                            
                            # Get centroid of visible joints (the red dots)
                            valid_joints = []
                            for pt in joints_2d:
                                if 0 < pt[0] < w and 0 < pt[1] < h:
                                    valid_joints.append(pt)
                            
                            if len(valid_joints) > 3:
                                valid_joints = np.array(valid_joints)
                                joints_center = np.mean(valid_joints, axis=0)
                                
                                # Get centroid of projected mesh vertices (within reasonable bounds)
                                valid_verts = []
                                for pt in verts_2d:
                                    # Use slightly larger bounds for mesh
                                    if -w < pt[0] < 2*w and -h < pt[1] < 2*h:
                                        valid_verts.append(pt)
                                
                                if len(valid_verts) > 100:
                                    valid_verts = np.array(valid_verts)
                                    mesh_center = np.mean(valid_verts, axis=0)
                                    
                                    # Offset to align mesh center with joints center
                                    offset_x = joints_center[0] - mesh_center[0]
                                    offset_y = joints_center[1] - mesh_center[1]
                                    
                                    if img_idx == 0:
                                        log.info(f"Mesh alignment (Auto):")
                                        log.debug(f" Joints center: ({joints_center[0]:.1f}, {joints_center[1]:.1f})")
                                        log.debug(f" Mesh center: ({mesh_center[0]:.1f}, {mesh_center[1]:.1f})")
                                        log.debug(f" Offset: ({offset_x:.1f}, {offset_y:.1f})")
                        elif img_idx == 0 and show_mesh:
                            log.info(f"Mesh alignment: Direct projection (no offset)")
                        
                        # Apply offset to mesh vertices
                        verts_2d[:, 0] += offset_x
                        verts_2d[:, 1] += offset_y
                        
                        if mesh_render_mode == "Solid":
                            # Solid rendering: fill triangles with depth sorting
                            face_depths = []
                            for face_idx, face in enumerate(faces):
                                avg_z = np.mean([vertices[int(face[k])][2] for k in range(3)])
                                face_depths.append((avg_z, face_idx))
                            face_depths.sort(reverse=True)  # Back to front
                            
                            for _, face_idx in face_depths:
                                face = faces[face_idx]
                                vi0, vi1, vi2 = int(face[0]), int(face[1]), int(face[2])
                                if vi0 < len(verts_2d) and vi1 < len(verts_2d) and vi2 < len(verts_2d):
                                    pts = np.array([
                                        [int(verts_2d[vi0][0]), int(verts_2d[vi0][1])],
                                        [int(verts_2d[vi1][0]), int(verts_2d[vi1][1])],
                                        [int(verts_2d[vi2][0]), int(verts_2d[vi2][1])],
                                    ], dtype=np.int32)
                                    
                                    if np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] < w) and \
                                       np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] < h):
                                        cv2.fillPoly(overlay, [pts], mesh_bgr)
                        else:
                            # Wireframe rendering - SAM3DBody intrinsics (primary)
                            step = max(1, len(faces) // 300)  # Limit edges for performance
                            for face_idx in range(0, len(faces), step):
                                face = faces[face_idx]
                                for k in range(3):
                                    vi, vj = int(face[k]), int(face[(k + 1) % 3])
                                    if vi < len(verts_2d) and vj < len(verts_2d):
                                        pt1 = (int(verts_2d[vi][0]), int(verts_2d[vi][1]))
                                        pt2 = (int(verts_2d[vj][0]), int(verts_2d[vj][1]))
                                        if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                                            0 <= pt2[0] < w and 0 <= pt2[1] < h):
                                            cv2.line(overlay, pt1, pt2, mesh_bgr, 1)
                            
                            # Compare Both mode: also draw MoGe2 mesh in different color
                            if verts_2d_moge is not None:
                                moge_mesh_color = (255, 128, 0)  # Orange for MoGe2
                                for face_idx in range(0, len(faces), step):
                                    face = faces[face_idx]
                                    for k in range(3):
                                        vi, vj = int(face[k]), int(face[(k + 1) % 3])
                                        if vi < len(verts_2d_moge) and vj < len(verts_2d_moge):
                                            pt1 = (int(verts_2d_moge[vi][0]), int(verts_2d_moge[vi][1]))
                                            pt2 = (int(verts_2d_moge[vj][0]), int(verts_2d_moge[vj][1]))
                                            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                                                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                                                cv2.line(overlay, pt1, pt2, moge_mesh_color, 1)
                
                # Draw intrinsics info on overlay if requested
                if show_intrinsics_info and sam3d_focal_length is not None:
                    y_pos = 30
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    
                    # SAM3DBody focal
                    cv2.putText(overlay, f"SAM3D focal: {sam3d_focal_length:.1f}px", 
                               (10, y_pos), font, font_scale, (0, 255, 0), 1)
                    y_pos += 20
                    
                    # MoGe2 focal if available
                    if moge_focal is not None:
                        focal_diff = moge_focal - sam3d_focal_length
                        cv2.putText(overlay, f"MoGe2 focal: {moge_focal:.1f}px (diff: {focal_diff:+.1f})", 
                                   (10, y_pos), font, font_scale, (255, 128, 0), 1)
                        y_pos += 20
                    
                    # Active source
                    cv2.putText(overlay, f"Using: {active_source}", 
                               (10, y_pos), font, font_scale, (255, 255, 255), 1)
                    
                    # Legend for Compare Both mode
                    if intrinsics_source == "Compare Both" and moge_focal is not None:
                        y_pos += 25
                        cv2.putText(overlay, "Green=SAM3D, Orange=MoGe2", 
                                   (10, y_pos), font, font_scale, (255, 255, 255), 1)
            
            # Blend with opacity
            blended = cv2.addWeighted(overlay, opacity, img_bgr, 1 - opacity, 0)
            
            # Convert back to RGB
            result_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
            result_float = result_rgb.astype(np.float32) / 255.0
            result_frames.append(result_float)
            
            # Progress logging
            log.progress(img_idx, num_images, "Overlay rendering", interval=10)
        
        # Stack all frames
        result_batch = torch.from_numpy(np.stack(result_frames, axis=0))
        
        info = f"Processed {num_images} frames with {len(frames)} mesh frames"
        log.info(f"{info}")
        
        return (result_batch, info)


# Node registration
NODE_CLASS_MAPPINGS = {
    "VerifyOverlay": VerifyOverlay,
    "VerifyOverlayBatch": VerifyOverlayBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VerifyOverlay": "SAM3DBody2abc: Verify Overlay",
    "VerifyOverlayBatch": "SAM3DBody2abc: Verify Overlay (Sequence)",
}
