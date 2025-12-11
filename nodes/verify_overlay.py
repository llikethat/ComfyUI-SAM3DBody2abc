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


def to_numpy(data):
    """Convert tensor to numpy."""
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    if isinstance(data, np.ndarray):
        return data.copy()
    return np.array(data)


def project_points_to_2d(points_3d, focal_length, cam_t, image_width, image_height):
    """
    Project 3D points to 2D using SAM3DBody's camera model.
    
    SAM3DBody uses weak perspective projection:
    - cam_t = [tx, ty, tz] where tz is depth/scale
    - Mesh is in camera space, offset by cam_t
    - 180 degree X rotation applied
    
    Args:
        points_3d: (N, 3) array of 3D points
        focal_length: focal length in pixels
        cam_t: camera translation [tx, ty, tz]
        image_width, image_height: image dimensions
        
    Returns:
        points_2d: (N, 2) array of 2D points
    """
    points_3d = np.array(points_3d)
    cam_t = np.array(cam_t)
    
    # Apply camera translation
    # SAM3DBody flips X in camera translation
    camera_translation = cam_t.copy()
    camera_translation[0] *= -1.0
    
    # Apply 180 degree rotation around X axis (same as renderer)
    # This flips Y and Z
    points_rotated = points_3d.copy()
    points_rotated[:, 1] *= -1
    points_rotated[:, 2] *= -1
    
    # Add camera translation
    points_translated = points_rotated + camera_translation
    
    # Perspective projection
    # x_2d = fx * X / Z + cx
    # y_2d = fy * Y / Z + cy
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    # Avoid division by zero
    z = points_translated[:, 2]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    
    x_2d = focal_length * points_translated[:, 0] / z + cx
    y_2d = focal_length * points_translated[:, 1] / z + cy
    
    return np.stack([x_2d, y_2d], axis=1)


# Define skeleton connections for visualization (based on common body keypoints)
# These are approximate - actual MHR skeleton may differ
SKELETON_CONNECTIONS = [
    # Spine
    (0, 1), (1, 2), (2, 3),
    # Left arm
    (2, 4), (4, 5), (5, 6),
    # Right arm  
    (2, 7), (7, 8), (8, 9),
    # Left leg
    (0, 10), (10, 11), (11, 12),
    # Right leg
    (0, 13), (13, 14), (14, 15),
]


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
                    "tooltip": "Draw mesh wireframe (slow for dense mesh)"
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
                    "tooltip": "Color for mesh wireframe"
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
        
        # Get projection parameters
        focal_length = mesh_data.get("focal_length")
        cam_t = to_numpy(mesh_data.get("camera"))
        
        if focal_length is None or cam_t is None:
            return (image, "Error: Missing camera parameters (focal_length or camera)")
        
        # Handle focal length format
        if isinstance(focal_length, (list, tuple, np.ndarray)):
            focal_length = float(focal_length[0]) if len(focal_length) > 0 else float(focal_length)
        
        info_parts = [f"Image: {w}x{h}", f"Focal: {focal_length:.1f}px", f"cam_t: [{cam_t[0]:.2f}, {cam_t[1]:.2f}, {cam_t[2]:.2f}]"]
        
        # Get colors
        joint_bgr = self._get_color(joint_color)
        skeleton_bgr = self._get_color(skeleton_color)
        mesh_bgr = self._get_color(mesh_color)
        
        # Project and draw joints
        joint_coords = to_numpy(mesh_data.get("joint_coords"))
        joints_2d = None
        
        if joint_coords is not None and (show_joints or show_skeleton):
            joints_2d = project_points_to_2d(joint_coords, focal_length, cam_t, w, h)
            info_parts.append(f"Joints: {len(joint_coords)}")
            
            if show_joints:
                for i, (x, y) in enumerate(joints_2d):
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(overlay, (int(x), int(y)), joint_radius, joint_bgr, -1)
                        # Draw joint index for first few joints
                        if i < 20:
                            cv2.putText(overlay, str(i), (int(x)+5, int(y)-5), 
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
        
        # Project and draw mesh wireframe (optional, can be slow)
        if show_mesh:
            vertices = to_numpy(mesh_data.get("vertices"))
            faces = to_numpy(mesh_data.get("faces"))
            
            if vertices is not None and faces is not None:
                verts_2d = project_points_to_2d(vertices, focal_length, cam_t, w, h)
                info_parts.append(f"Vertices: {len(vertices)}, Faces: {len(faces)}")
                
                # Draw subset of edges (every Nth face to avoid too dense)
                step = max(1, len(faces) // 500)  # Limit to ~500 edges
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
        print(f"[VerifyOverlay] {info}")
        
        return (result_tensor, info)


class VerifyOverlayBatch:
    """
    Create verification overlay for a batch/sequence of frames.
    Useful for checking alignment across video frames.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Current frame image"
                }),
                "mesh_sequence": ("MESH_SEQUENCE", {
                    "tooltip": "Accumulated mesh sequence"
                }),
                "frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Frame index to visualize"
                }),
            },
            "optional": {
                "show_joints": ("BOOLEAN", {"default": True}),
                "show_mesh": ("BOOLEAN", {"default": False}),
                "joint_radius": ("INT", {"default": 5, "min": 1, "max": 20}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("overlay_image", "info")
    FUNCTION = "create_overlay"
    CATEGORY = "SAM3DBody2abc/Debug"
    
    def create_overlay(
        self,
        image,
        mesh_sequence: Dict,
        frame_index: int = 0,
        show_joints: bool = True,
        show_mesh: bool = False,
        joint_radius: int = 5,
    ) -> Tuple[Any, str]:
        """Create overlay from mesh sequence."""
        
        frames = mesh_sequence.get("frames", {})
        if frame_index not in frames:
            return (image, f"Error: Frame {frame_index} not in sequence")
        
        frame = frames[frame_index]
        
        # Build mesh_data dict from frame
        # Handle numpy array boolean check properly
        pred_cam_t = frame.get("pred_cam_t")
        if pred_cam_t is None:
            pred_cam_t = frame.get("camera")
        
        mesh_data = {
            "vertices": frame.get("vertices"),
            "joint_coords": frame.get("joint_coords"),
            "camera": pred_cam_t,
            "focal_length": frame.get("focal_length"),
            "faces": mesh_sequence.get("faces"),
        }
        
        # Use the single frame overlay
        overlay_node = VerifyOverlay()
        return overlay_node.create_overlay(
            image, mesh_data,
            show_joints=show_joints,
            show_skeleton=True,
            show_mesh=show_mesh,
            joint_radius=joint_radius,
        )


# Node registration
NODE_CLASS_MAPPINGS = {
    "VerifyOverlay": VerifyOverlay,
    "VerifyOverlayBatch": VerifyOverlayBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VerifyOverlay": "SAM3DBody2abc: Verify Overlay",
    "VerifyOverlayBatch": "SAM3DBody2abc: Verify Overlay (Sequence)",
}
