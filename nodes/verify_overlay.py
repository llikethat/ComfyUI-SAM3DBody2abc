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
    
    SAM3DBody computes pred_keypoints_2d from pred_keypoints_3d using perspective projection.
    The camera model is:
    - 3D points are in body-centered coordinates (Y points UP)
    - Image coordinates have Y pointing DOWN (y=0 is top)
    - cam_t = [tx, ty, tz] is the camera translation
    - Projection requires Y negation to convert 3D Y-up to image Y-down
    
    This should match pred_keypoints_2d when applied to pred_keypoints_3d.
    
    Args:
        points_3d: (N, 3) array of 3D points (vertices or joints)
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
    
    # SAM3DBody camera model:
    # Points in camera space = points_3d + cam_t
    # Then perspective projection with Y-axis flip (3D Y-up → image Y-down)
    tx, ty, tz = cam_t[0], cam_t[1], cam_t[2]
    
    # Add camera translation to get points in camera space
    X = points_3d[:, 0] + tx
    Y = -(points_3d[:, 1] + ty)  # NEGATE Y: 3D Y-up → image Y-down
    Z = points_3d[:, 2] + tz
    
    # Avoid division by zero
    Z = np.where(np.abs(Z) < 1e-6, 1e-6, Z)
    
    # Perspective projection
    x_2d = focal_length * X / Z + cx
    y_2d = focal_length * Y / Z + cy
    
    return np.stack([x_2d, y_2d], axis=1)


# Define key body joint indices for MHR model visualization
# These are approximate mappings for the main body landmarks
MHR_KEY_JOINTS = {
    'pelvis': 0,
    'spine1': 1,
    'spine2': 2, 
    'spine3': 3,
    'neck': 4,
    'head': 5,
    'left_shoulder': 6,
    'left_elbow': 7,
    'left_wrist': 8,
    'right_shoulder': 9,
    'right_elbow': 10,
    'right_wrist': 11,
    'left_hip': 12,
    'left_knee': 13,
    'left_ankle': 14,
    'right_hip': 15,
    'right_knee': 16,
    'right_ankle': 17,
}

# Fallback skeleton connections if joint_parents not available
FALLBACK_SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),  # Spine to head
    (3, 6), (6, 7), (7, 8),  # Left arm
    (3, 9), (9, 10), (10, 11),  # Right arm
    (0, 12), (12, 13), (13, 14),  # Left leg
    (0, 15), (15, 16), (16, 17),  # Right leg
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
        
        # Debug: print mesh_data keys
        print(f"[VerifyOverlay] mesh_data keys: {list(mesh_data.keys())}")
        
        # Get projection parameters
        focal_length = mesh_data.get("focal_length")
        cam_t = to_numpy(mesh_data.get("camera"))
        
        # Also check for 2D keypoints (more reliable if available)
        keypoints_2d = to_numpy(mesh_data.get("pred_keypoints_2d"))
        
        if keypoints_2d is not None:
            print(f"[VerifyOverlay] Using pred_keypoints_2d directly (shape: {keypoints_2d.shape})")
        
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
                print(f"[VerifyOverlay] First 5 joints 3D: {joint_coords[:5]}")
                print(f"[VerifyOverlay] First 5 joints 2D: {joints_2d[:5]}")
                print(f"[VerifyOverlay] cam_t: {cam_t}, focal: {focal_length}")
        
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
        
        # Project and draw mesh wireframe (optional, can be slow)
        if show_mesh:
            vertices = to_numpy(mesh_data.get("vertices"))
            faces = to_numpy(mesh_data.get("faces"))
            
            if vertices is not None and faces is not None and focal_length is not None and cam_t is not None:
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
    Create verification overlay for ALL frames in a sequence.
    Outputs a video/batch of overlay images.
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
                "show_joints": ("BOOLEAN", {"default": True}),
                "show_skeleton": ("BOOLEAN", {"default": True}),
                "show_mesh": ("BOOLEAN", {"default": False}),
                "joint_radius": ("INT", {"default": 5, "min": 1, "max": 20}),
                "line_thickness": ("INT", {"default": 2, "min": 1, "max": 10}),
                "joint_color": (["red", "green", "blue", "yellow", "cyan", "magenta", "white"], {"default": "green"}),
                "skeleton_color": (["red", "green", "blue", "yellow", "cyan", "magenta", "white"], {"default": "cyan"}),
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
        show_joints: bool = True,
        show_skeleton: bool = True,
        show_mesh: bool = False,
        joint_radius: int = 5,
        line_thickness: int = 2,
        joint_color: str = "green",
        skeleton_color: str = "cyan",
    ) -> Tuple[Any, str]:
        """Create overlay for all frames."""
        
        frames = mesh_sequence.get("frames", {})
        faces = mesh_sequence.get("faces")
        joint_parents = mesh_sequence.get("joint_parents")
        
        if not frames:
            return (images, "Error: No frames in mesh_sequence")
        
        # Build skeleton connections from joint_parents
        skeleton_connections = get_skeleton_connections(joint_parents)
        if joint_parents is not None:
            print(f"[VerifyOverlayBatch] Using {len(skeleton_connections)} skeleton connections from joint_parents")
        else:
            print(f"[VerifyOverlayBatch] Using fallback skeleton connections")
        
        # Get colors
        joint_bgr = self._get_color(joint_color)
        skeleton_bgr = self._get_color(skeleton_color)
        
        # Convert images to numpy
        if isinstance(images, torch.Tensor):
            images_np = images.cpu().numpy()
        else:
            images_np = np.array(images)
        
        num_images = images_np.shape[0]
        h, w = images_np.shape[1], images_np.shape[2]
        
        print(f"[VerifyOverlayBatch] Processing {num_images} images, {len(frames)} frames in sequence")
        
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
                    print(f"[VerifyOverlayBatch] Frame keys: {list(frame.keys())}")
                
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
                            print(f"[VerifyOverlayBatch] Detection bbox: [{x1}, {y1}, {x2}, {y2}]")
                elif img_idx == 0:
                    print(f"[VerifyOverlayBatch] No bbox in frame data")
                
                # Get 2D keypoints if available (most reliable)
                keypoints_2d = frame.get("pred_keypoints_2d")
                keypoints_3d = frame.get("pred_keypoints_3d")  # Same 70 joints as keypoints_2d
                joint_coords = frame.get("joint_coords")  # 127 full skeleton joints
                focal_length = frame.get("focal_length")
                cam_t = frame.get("pred_cam_t")
                if cam_t is None:
                    cam_t = frame.get("camera")
                
                joints_2d = None
                
                # DEBUG: Compare ground truth vs our projection (frame 0 only)
                if img_idx == 0:
                    print(f"\n[DEBUG] ========== PROJECTION COMPARISON (Frame 0) ==========")
                    print(f"[DEBUG] Image size: {w}x{h}, Focal: {focal_length}")
                    print(f"[DEBUG] pred_cam_t: {cam_t}")
                    print(f"[DEBUG] Data availability:")
                    print(f"[DEBUG]   pred_keypoints_2d: {keypoints_2d.shape if keypoints_2d is not None else 'None'}")
                    print(f"[DEBUG]   pred_keypoints_3d: {np.array(keypoints_3d).shape if keypoints_3d is not None else 'None'}")
                    print(f"[DEBUG]   joint_coords (127): {np.array(joint_coords).shape if joint_coords is not None else 'None'}")
                    
                    if keypoints_2d is not None and focal_length is not None and cam_t is not None:
                        gt_2d = np.array(keypoints_2d)[:, :2] if np.array(keypoints_2d).ndim == 2 else np.array(keypoints_2d)
                        cam_t_np = np.array(cam_t)
                        
                        # Use pred_keypoints_3d if available (same 70 joints), else fall back to joint_coords (127 joints)
                        if keypoints_3d is not None:
                            kp_3d = np.array(keypoints_3d)
                            print(f"[DEBUG] Using pred_keypoints_3d (same {kp_3d.shape[0]} joints as ground truth)")
                        else:
                            kp_3d = np.array(joint_coords)
                            print(f"[DEBUG] WARNING: pred_keypoints_3d not available!")
                            print(f"[DEBUG] Falling back to joint_coords ({kp_3d.shape[0]} joints) - DIFFERENT joint set!")
                            print(f"[DEBUG] Comparison may not be meaningful!")
                        
                        our_2d = project_points_to_2d(kp_3d, focal_length, cam_t_np, w, h)
                        
                        print(f"[DEBUG]")
                        print(f"[DEBUG] Joint | Ground Truth (x,y) | Our Projection (x,y) | Diff (dx, dy)")
                        print(f"[DEBUG] ------|-------------------|---------------------|---------------")
                        
                        num_compare = min(10, len(gt_2d), len(our_2d))
                        total_dx, total_dy = 0.0, 0.0
                        for i in range(num_compare):
                            gt_x, gt_y = gt_2d[i][0], gt_2d[i][1]
                            our_x, our_y = our_2d[i][0], our_2d[i][1]
                            dx, dy = our_x - gt_x, our_y - gt_y
                            total_dx += dx
                            total_dy += dy
                            print(f"[DEBUG]   {i:2d}  | ({gt_x:7.1f}, {gt_y:6.1f}) | ({our_x:7.1f}, {our_y:6.1f}) | ({dx:+6.1f}, {dy:+6.1f})")
                        
                        avg_dx, avg_dy = total_dx / num_compare, total_dy / num_compare
                        print(f"[DEBUG] ------|-------------------|---------------------|---------------")
                        print(f"[DEBUG] AVERAGE OFFSET: dx={avg_dx:+.1f}px, dy={avg_dy:+.1f}px")
                        print(f"[DEBUG] ==========================================================\n")
                
                if keypoints_2d is not None:
                    # Use 2D keypoints directly
                    keypoints_2d = np.array(keypoints_2d)
                    if keypoints_2d.ndim == 2:
                        joints_2d = keypoints_2d[:, :2] if keypoints_2d.shape[1] >= 2 else keypoints_2d
                    if img_idx == 0:
                        print(f"[VerifyOverlayBatch] Using pred_keypoints_2d: {joints_2d.shape}")
                
                elif joint_coords is not None and focal_length is not None and cam_t is not None:
                    # Project 3D to 2D
                    joint_coords = np.array(joint_coords)
                    cam_t = np.array(cam_t)
                    joints_2d = project_points_to_2d(joint_coords, focal_length, cam_t, w, h)
                    if img_idx == 0:
                        print(f"[VerifyOverlayBatch] Projecting 3D joints: focal={focal_length}, cam_t={cam_t}")
                
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
                        
                        # Project mesh vertices first
                        verts_2d = project_points_to_2d(vertices, focal_length, cam_t_np, w, h)
                        
                        # Compute offset to align mesh with detected keypoints
                        offset_x, offset_y = 0.0, 0.0
                        
                        if joints_2d is not None:
                            # Method 1: Use centroid of joints vs projected mesh
                            # The pred_keypoints_2d are the ground truth positions
                            # The projected mesh should align with them
                            
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
                                        print(f"[VerifyOverlayBatch] Mesh alignment:")
                                        print(f"  Joints center: ({joints_center[0]:.1f}, {joints_center[1]:.1f})")
                                        print(f"  Mesh center: ({mesh_center[0]:.1f}, {mesh_center[1]:.1f})")
                                        print(f"  Offset: ({offset_x:.1f}, {offset_y:.1f})")
                        
                        # Apply offset to mesh vertices
                        verts_2d[:, 0] += offset_x
                        verts_2d[:, 1] += offset_y
                        
                        # Draw subset of edges (every Nth face to avoid too dense)
                        mesh_bgr = (0, 255, 255)  # Yellow mesh
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
            
            # Convert back to RGB
            result_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            result_float = result_rgb.astype(np.float32) / 255.0
            result_frames.append(result_float)
        
        # Stack all frames
        result_batch = torch.from_numpy(np.stack(result_frames, axis=0))
        
        info = f"Processed {num_images} frames with {len(frames)} mesh frames"
        print(f"[VerifyOverlayBatch] {info}")
        
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
