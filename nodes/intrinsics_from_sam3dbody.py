"""
Intrinsics from SAM3DBody - Extract Camera Intrinsics for SAM3DBody2abc v5.0

This node extracts camera intrinsics from SAM3DBody's mesh_data output.

Usage:
    SAM3DBody Process → mesh_data → IntrinsicsFromSAM3DBody → INTRINSICS
                                              ↓
                                        debug_overlay
                                              ↓
                                      CameraSolverV2

Version: 5.0.0
Author: SAM3DBody2abc Project
"""

import numpy as np
import torch
import cv2
from typing import Dict, Tuple, Any, Optional


def to_numpy(data):
    """Convert tensor to numpy."""
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    if isinstance(data, np.ndarray):
        return data.copy()
    try:
        return np.array(data)
    except:
        return data


class IntrinsicsFromSAM3DBody:
    """
    Extract camera intrinsics from SAM3DBody mesh_data output.
    
    Takes mesh_data (SAM3D_OUTPUT) and extracts:
    - focal_length (in pixels)
    - pred_cam_t (camera translation)
    - Builds standardized INTRINSICS for v5.0 pipeline
    
    Generates a debug overlay showing the mesh projection.
    """
    
    FRAME_SELECTION = ["first", "middle", "last", "specific"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT", {
                    "tooltip": "mesh_data output from SAM3DBody Process node"
                }),
                "images": ("IMAGE", {
                    "tooltip": "Video frames for debug overlay"
                }),
            },
            "optional": {
                # === Mask for overlay visualization ===
                "mask": ("MASK", {
                    "tooltip": "Foreground mask from SAM3 - shown on debug overlay"
                }),
                
                # === Frame Selection ===
                "frame_selection": (cls.FRAME_SELECTION, {
                    "default": "specific",
                    "tooltip": "Which frame to use for debug overlay"
                }),
                "specific_frame": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 10000,
                    "tooltip": "Frame number when frame_selection='specific'"
                }),
                
                # === Detection Settings ===
                "bbox_threshold": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Detection confidence threshold (for reference)"
                }),
                
                # === Sensor Configuration ===
                "sensor_width_mm": ("FLOAT", {
                    "default": 36.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Assumed sensor width for focal length conversion to mm"
                }),
                
                # === Debug Overlay Settings ===
                "overlay_opacity": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Mesh overlay opacity"
                }),
                "show_skeleton": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show skeleton joints on overlay"
                }),
                "show_mesh": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show mesh points on overlay"
                }),
                "show_mask_outline": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show mask outline on overlay"
                }),
            }
        }
    
    RETURN_TYPES = ("INTRINSICS", "IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("intrinsics", "debug_overlay", "focal_length_mm", "status")
    FUNCTION = "extract_intrinsics"
    CATEGORY = "SAM3DBody2abc/Camera"
    
    def extract_intrinsics(
        self,
        mesh_data: Dict,
        images: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        frame_selection: str = "specific",
        specific_frame: int = 1,
        bbox_threshold: float = 0.8,
        sensor_width_mm: float = 36.0,
        overlay_opacity: float = 0.6,
        show_skeleton: bool = True,
        show_mesh: bool = True,
        show_mask_outline: bool = True,
    ) -> Tuple[Dict, torch.Tensor, float, str]:
        """
        Extract intrinsics from SAM3DBody mesh_data.
        """
        
        num_frames = images.shape[0]
        H, W = images.shape[1], images.shape[2]
        
        # Select reference frame for overlay
        if frame_selection == "first":
            frame_idx = 0
        elif frame_selection == "middle":
            frame_idx = num_frames // 2
        elif frame_selection == "last":
            frame_idx = num_frames - 1
        else:  # specific
            frame_idx = min(specific_frame, num_frames - 1)
        
        print(f"\n{'='*60}")
        print(f"[IntrinsicsFromSAM3DBody] Extracting intrinsics from mesh_data")
        print(f"[IntrinsicsFromSAM3DBody] Video: {num_frames} frames, {W}x{H}")
        print(f"[IntrinsicsFromSAM3DBody] Debug overlay frame: {frame_idx}")
        print(f"[IntrinsicsFromSAM3DBody] mesh_data keys: {list(mesh_data.keys())}")
        print(f"{'='*60}")
        
        # Extract focal length
        focal_length_px = mesh_data.get("focal_length")
        if focal_length_px is None:
            print("[IntrinsicsFromSAM3DBody] WARNING: No focal_length in mesh_data, using fallback")
            focal_length_px = float(W)  # Fallback to image width
        else:
            if hasattr(focal_length_px, 'item'):
                focal_length_px = float(focal_length_px.item())
            elif hasattr(focal_length_px, 'cpu'):
                focal_length_px = float(focal_length_px.cpu().numpy())
            else:
                focal_length_px = float(focal_length_px)
        
        print(f"[IntrinsicsFromSAM3DBody] Focal length: {focal_length_px:.1f}px")
        
        # Extract camera translation
        pred_cam_t = mesh_data.get("camera")
        if pred_cam_t is None:
            pred_cam_t = mesh_data.get("pred_cam_t")
        
        cam_t_list = None
        pred_cam_t_np = None
        if pred_cam_t is not None:
            pred_cam_t_np = to_numpy(pred_cam_t)
            if pred_cam_t_np is not None:
                if pred_cam_t_np.ndim > 1:
                    pred_cam_t_np = pred_cam_t_np.flatten()[:3]
                cam_t_list = pred_cam_t_np.tolist()
                print(f"[IntrinsicsFromSAM3DBody] pred_cam_t: [{pred_cam_t_np[0]:.3f}, {pred_cam_t_np[1]:.3f}, {pred_cam_t_np[2]:.3f}]")
        
        # Convert to mm
        focal_length_mm = focal_length_px * sensor_width_mm / W
        
        # Compute FOV
        fov_x_deg = 2 * np.degrees(np.arctan(W / (2 * focal_length_px)))
        fov_y_deg = 2 * np.degrees(np.arctan(H / (2 * focal_length_px)))
        
        # Build INTRINSICS output
        intrinsics = {
            "focal_px": float(focal_length_px),
            "focal_mm": float(focal_length_mm),
            "sensor_width_mm": float(sensor_width_mm),
            "cx": float(W / 2),
            "cy": float(H / 2),
            "width": int(W),
            "height": int(H),
            "fov_x_deg": float(fov_x_deg),
            "fov_y_deg": float(fov_y_deg),
            "aspect_ratio": float(W / H),
            "source": "sam3dbody",
            "confidence": 0.85,
            "k_matrix": [
                [focal_length_px, 0.0, W / 2],
                [0.0, focal_length_px, H / 2],
                [0.0, 0.0, 1.0]
            ],
            "pred_cam_t": cam_t_list,
            "reference_frame": frame_idx,
            "per_frame": None,
            "is_variable": False,
        }
        
        # Generate debug overlay
        debug_overlay = self._generate_overlay(
            images, frame_idx, mesh_data, focal_length_px, pred_cam_t_np,
            W, H, overlay_opacity, show_skeleton, show_mesh,
            mask, show_mask_outline
        )
        
        status = f"SAM3DBody: {focal_length_mm:.1f}mm ({fov_x_deg:.1f}° FOV)"
        print(f"[IntrinsicsFromSAM3DBody] {status}")
        
        return (intrinsics, debug_overlay, focal_length_mm, status)
    
    def _generate_overlay(
        self,
        images: torch.Tensor,
        frame_idx: int,
        mesh_data: Dict,
        focal_length: float,
        pred_cam_t: Optional[np.ndarray],
        W: int, H: int,
        opacity: float,
        show_skeleton: bool,
        show_mesh: bool,
        mask: Optional[torch.Tensor] = None,
        show_mask_outline: bool = True,
    ) -> torch.Tensor:
        """Generate debug overlay with mesh, skeleton, and mask outline."""
        
        # Get the selected frame
        frame = images[frame_idx]
        
        # Convert to numpy
        frame_np = to_numpy(frame)
        if frame_np.max() <= 1.0:
            frame_np = (frame_np * 255).astype(np.uint8)
        else:
            frame_np = frame_np.astype(np.uint8)
        
        overlay = frame_np.copy()
        
        # Draw mask outline first (so it's behind other elements)
        if show_mask_outline and mask is not None:
            mask_np = to_numpy(mask)
            if mask_np is not None:
                # Get the correct frame of mask
                if mask_np.ndim == 3:
                    if frame_idx < mask_np.shape[0]:
                        mask_frame = mask_np[frame_idx]
                    else:
                        mask_frame = mask_np[0]
                elif mask_np.ndim == 4:
                    if frame_idx < mask_np.shape[0]:
                        mask_frame = mask_np[frame_idx, 0]
                    else:
                        mask_frame = mask_np[0, 0]
                else:
                    mask_frame = mask_np
                
                # Resize mask if needed
                if mask_frame.shape[0] != H or mask_frame.shape[1] != W:
                    mask_frame = cv2.resize(mask_frame.astype(np.float32), (W, H))
                
                # Find contours
                mask_binary = (mask_frame > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw contours (cyan color)
                cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
        
        # Get vertices
        vertices = to_numpy(mesh_data.get("vertices"))
        if vertices is not None and vertices.ndim == 3:
            vertices = vertices[0]
        
        # Get joints
        joints = to_numpy(mesh_data.get("joint_coords") or mesh_data.get("joints"))
        if joints is not None and joints.ndim == 3:
            joints = joints[0]
        
        # Draw mesh points
        if show_mesh and vertices is not None:
            pts_2d = self._project_points(vertices, focal_length, pred_cam_t, W, H)
            if pts_2d is not None:
                mesh_overlay = overlay.copy()
                for i in range(0, len(pts_2d), 5):  # Subsample for speed
                    x, y = int(pts_2d[i, 0]), int(pts_2d[i, 1])
                    if 0 <= x < W and 0 <= y < H:
                        cv2.circle(mesh_overlay, (x, y), 1, (100, 255, 100), -1)
                overlay = cv2.addWeighted(mesh_overlay, opacity, overlay, 1 - opacity, 0)
        
        # Draw skeleton
        if show_skeleton and joints is not None:
            pts_2d = self._project_points(joints, focal_length, pred_cam_t, W, H)
            if pts_2d is not None:
                # Draw connections (simplified SMPL)
                connections = [
                    (0, 1), (0, 2), (1, 4), (2, 5), (4, 7), (5, 8),
                    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
                    (9, 13), (13, 16), (16, 18), (18, 20),
                    (9, 14), (14, 17), (17, 19), (19, 21),
                ]
                for i, j in connections:
                    if i < len(pts_2d) and j < len(pts_2d):
                        pt1 = (int(pts_2d[i, 0]), int(pts_2d[i, 1]))
                        pt2 = (int(pts_2d[j, 0]), int(pts_2d[j, 1]))
                        if (0 <= pt1[0] < W and 0 <= pt1[1] < H and
                            0 <= pt2[0] < W and 0 <= pt2[1] < H):
                            cv2.line(overlay, pt1, pt2, (255, 255, 0), 2)
                
                # Draw joints
                for i, pt in enumerate(pts_2d):
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= x < W and 0 <= y < H:
                        color = (0, 0, 255) if i in [0, 3, 6, 9, 12, 15] else (255, 0, 0)
                        cv2.circle(overlay, (x, y), 5, color, -1)
                        cv2.circle(overlay, (x, y), 5, (255, 255, 255), 1)
        
        # Add info text
        focal_mm = focal_length * 36.0 / W
        cv2.putText(overlay, f"Focal: {focal_length:.0f}px ({focal_mm:.1f}mm)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(overlay, f"FOV: {2*np.degrees(np.arctan(W/(2*focal_length))):.1f} deg", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(overlay, f"Source: SAM3DBody | Frame: {frame_idx}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if pred_cam_t is not None:
            cv2.putText(overlay, f"cam_t: [{pred_cam_t[0]:.2f}, {pred_cam_t[1]:.2f}, {pred_cam_t[2]:.2f}]", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Convert back to tensor
        overlay_tensor = torch.from_numpy(overlay).float() / 255.0
        overlay_tensor = overlay_tensor.unsqueeze(0)
        
        return overlay_tensor
    
    def _project_points(
        self,
        points_3d: np.ndarray,
        focal_length: float,
        pred_cam_t: Optional[np.ndarray],
        W: int, H: int,
    ) -> Optional[np.ndarray]:
        """Project 3D points to 2D using SAM3DBody's camera model."""
        
        if points_3d is None or len(points_3d) == 0:
            return None
        
        points = points_3d.copy()
        
        if pred_cam_t is not None and len(pred_cam_t) >= 3:
            tx, ty, tz = pred_cam_t[0], pred_cam_t[1], pred_cam_t[2]
            points[:, 0] = points[:, 0] + tx
            points[:, 1] = points[:, 1] + ty
            points[:, 2] = points[:, 2] + tz
        else:
            points[:, 2] = points[:, 2] + 5.0
        
        z = points[:, 2:3]
        z = np.maximum(z, 0.1)
        
        pts_2d = np.zeros((len(points), 2))
        pts_2d[:, 0] = points[:, 0] * focal_length / z[:, 0] + W / 2
        pts_2d[:, 1] = points[:, 1] * focal_length / z[:, 0] + H / 2
        
        return pts_2d


# Node registration
NODE_CLASS_MAPPINGS = {
    "IntrinsicsFromSAM3DBody": IntrinsicsFromSAM3DBody,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IntrinsicsFromSAM3DBody": "📷 Intrinsics from SAM3DBody",
}
