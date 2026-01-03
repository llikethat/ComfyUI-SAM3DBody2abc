"""
Intrinsics from SAM3DBody - Single Frame Estimation for SAM3DBody2abc v5.0

This node runs SAM3DBody on a SINGLE frame to:
1. Extract camera intrinsics (focal length) from SAM3DBody's estimation
2. Generate a debug overlay showing the mesh on the frame
3. Allow user to verify pose estimation quality before full video processing

This is the recommended way to get intrinsics for the v5.0 pipeline because:
- SAM3DBody's focal length is tuned for human body reconstruction
- Debug overlay lets you verify the estimation is good
- Single frame is fast (~1-2 seconds)
- Same intrinsics will be used consistently throughout pipeline

Usage:
    Load Video → Select Frame → IntrinsicsFromSAM3DBody → INTRINSICS
                                        ↓
                                  debug_overlay (verify mesh fits person)
                                        ↓
                              CameraSolverV2 (TAPIR)

Version: 5.0.0
Author: SAM3DBody2abc Project
"""

import numpy as np
import torch
import cv2
from typing import Dict, Tuple, Any, Optional


class IntrinsicsFromSAM3DBody:
    """
    Extract camera intrinsics from SAM3DBody using a single reference frame.
    
    This node:
    1. Selects a reference frame from the video
    2. Runs SAM3DBody inference on that single frame
    3. Extracts focal length from SAM3DBody's prediction
    4. Generates a debug overlay showing the mesh fit
    5. Outputs INTRINSICS for use in camera solver
    
    The debug overlay helps verify that:
    - The person is detected correctly
    - The mesh aligns well with the body
    - The focal length estimation is reasonable
    """
    
    FRAME_SELECTION = ["first", "middle", "last", "specific"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("SAM3D_MODEL", {
                    "tooltip": "SAM3DBody model from Load SAM3DBody node"
                }),
                "images": ("IMAGE", {
                    "tooltip": "Video frames to extract intrinsics from"
                }),
            },
            "optional": {
                # === Frame Selection ===
                "frame_selection": (cls.FRAME_SELECTION, {
                    "default": "middle",
                    "tooltip": "Which frame to use for intrinsics estimation"
                }),
                "specific_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "tooltip": "Frame number when frame_selection='specific'"
                }),
                
                # === Mask (optional, improves detection) ===
                "mask": ("MASK", {
                    "tooltip": "Optional mask from SAM3 to help detection"
                }),
                
                # === SAM3DBody Settings ===
                "bbox_threshold": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Detection confidence threshold"
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
                    "tooltip": "Show mesh wireframe on overlay"
                }),
            }
        }
    
    RETURN_TYPES = ("INTRINSICS", "IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("intrinsics", "debug_overlay", "focal_length_mm", "status")
    FUNCTION = "extract_intrinsics"
    CATEGORY = "SAM3DBody2abc/Camera"
    
    def extract_intrinsics(
        self,
        model,
        images: torch.Tensor,
        frame_selection: str = "middle",
        specific_frame: int = 0,
        mask: Optional[torch.Tensor] = None,
        bbox_threshold: float = 0.8,
        sensor_width_mm: float = 36.0,
        overlay_opacity: float = 0.6,
        show_skeleton: bool = True,
        show_mesh: bool = True,
    ) -> Tuple[Dict, torch.Tensor, float, str]:
        """
        Extract intrinsics from single frame SAM3DBody inference.
        """
        
        num_frames = images.shape[0]
        H, W = images.shape[1], images.shape[2]
        
        # Select reference frame
        if frame_selection == "first":
            frame_idx = 0
        elif frame_selection == "middle":
            frame_idx = num_frames // 2
        elif frame_selection == "last":
            frame_idx = num_frames - 1
        else:  # specific
            frame_idx = min(specific_frame, num_frames - 1)
        
        print(f"\n{'='*60}")
        print(f"[IntrinsicsFromSAM3DBody] Extracting intrinsics from frame {frame_idx}")
        print(f"[IntrinsicsFromSAM3DBody] Video: {num_frames} frames, {W}x{H}")
        print(f"{'='*60}")
        
        # Get the reference frame
        ref_frame = images[frame_idx:frame_idx+1]  # Keep batch dimension
        
        # Get mask for this frame if provided
        ref_mask = None
        if mask is not None:
            if mask.dim() == 3:  # [N, H, W]
                ref_mask = mask[frame_idx:frame_idx+1]
            elif mask.dim() == 4:  # [N, 1, H, W]
                ref_mask = mask[frame_idx:frame_idx+1, 0]
        
        # Run SAM3DBody inference
        try:
            output = self._run_sam3dbody(
                model, ref_frame, ref_mask, bbox_threshold
            )
        except Exception as e:
            print(f"[IntrinsicsFromSAM3DBody] SAM3DBody inference failed: {e}")
            return self._fallback_result(images, frame_idx, sensor_width_mm, str(e))
        
        if output is None:
            print("[IntrinsicsFromSAM3DBody] No person detected in frame")
            return self._fallback_result(images, frame_idx, sensor_width_mm, "No person detected")
        
        # Extract focal length
        focal_length_px = output.get("focal_length")
        if focal_length_px is None:
            focal_length_px = W  # Fallback to image width
            print(f"[IntrinsicsFromSAM3DBody] No focal length in output, using fallback: {focal_length_px}px")
        else:
            if hasattr(focal_length_px, 'item'):
                focal_length_px = float(focal_length_px.item())
            else:
                focal_length_px = float(focal_length_px)
            print(f"[IntrinsicsFromSAM3DBody] SAM3DBody focal length: {focal_length_px:.1f}px")
        
        # Convert to mm
        focal_length_mm = focal_length_px * sensor_width_mm / W
        
        # Get pred_cam_t for additional info
        pred_cam_t = output.get("pred_cam_t")
        if pred_cam_t is not None:
            if hasattr(pred_cam_t, 'cpu'):
                pred_cam_t = pred_cam_t.cpu().numpy()
            if pred_cam_t.ndim > 1:
                pred_cam_t = pred_cam_t[0]
            print(f"[IntrinsicsFromSAM3DBody] pred_cam_t: tx={pred_cam_t[0]:.3f}, ty={pred_cam_t[1]:.3f}, tz={pred_cam_t[2]:.3f}")
        
        # Generate debug overlay
        debug_overlay = self._generate_overlay(
            ref_frame[0], output, overlay_opacity, show_skeleton, show_mesh
        )
        
        # Build INTRINSICS output
        fov_x_deg = 2 * np.degrees(np.arctan(W / (2 * focal_length_px)))
        fov_y_deg = 2 * np.degrees(np.arctan(H / (2 * focal_length_px)))
        
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
            "confidence": 0.85,  # SAM3DBody is generally reliable
            "k_matrix": [
                [focal_length_px, 0.0, W / 2],
                [0.0, focal_length_px, H / 2],
                [0.0, 0.0, 1.0]
            ],
            "reference_frame": frame_idx,
            "pred_cam_t": pred_cam_t.tolist() if pred_cam_t is not None else None,
            "per_frame": None,
            "is_variable": False,
        }
        
        status = f"SAM3DBody: {focal_length_mm:.1f}mm ({fov_x_deg:.1f}° FOV) from frame {frame_idx}"
        print(f"[IntrinsicsFromSAM3DBody] {status}")
        
        return (intrinsics, debug_overlay, focal_length_mm, status)
    
    def _run_sam3dbody(
        self,
        model,
        frame: torch.Tensor,
        mask: Optional[torch.Tensor],
        bbox_threshold: float,
    ) -> Optional[Dict]:
        """Run SAM3DBody inference on single frame."""
        
        # Get device from model
        device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cuda'
        
        # Prepare image - SAM3DBody expects [B, C, H, W] in range [0, 1]
        if frame.dim() == 4:  # [B, H, W, C]
            img = frame.permute(0, 3, 1, 2)  # -> [B, C, H, W]
        else:
            img = frame.unsqueeze(0).permute(0, 3, 1, 2)
        
        img = img.to(device)
        
        # Compute bounding box from mask or use full image
        if mask is not None:
            bbox = self._compute_bbox_from_mask(mask)
        else:
            # Try to detect person using model's built-in detection
            bbox = None
        
        # Run inference
        with torch.no_grad():
            # SAM3DBody expects specific input format
            # This may vary based on exact SAM3DBody version
            try:
                if hasattr(model, 'inference'):
                    output = model.inference(img, bbox=bbox)
                elif hasattr(model, 'forward'):
                    output = model(img)
                else:
                    output = model(img)
            except Exception as e:
                print(f"[IntrinsicsFromSAM3DBody] Model inference error: {e}")
                # Try alternative call patterns
                try:
                    output = model(img, bbox)
                except:
                    return None
        
        # Validate output
        if output is None:
            return None
        
        # Convert to dict if needed
        if isinstance(output, tuple):
            output = {"vertices": output[0], "focal_length": output[1] if len(output) > 1 else None}
        
        return output
    
    def _compute_bbox_from_mask(self, mask: torch.Tensor) -> Optional[list]:
        """Compute bounding box from mask."""
        mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)
        
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        
        # Find non-zero pixels
        rows = np.any(mask_np > 0.5, axis=1)
        cols = np.any(mask_np > 0.5, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Add padding
        H, W = mask_np.shape
        pad = 20
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(W, x_max + pad)
        y_max = min(H, y_max + pad)
        
        return [x_min, y_min, x_max, y_max]
    
    def _generate_overlay(
        self,
        frame: torch.Tensor,
        output: Dict,
        opacity: float,
        show_skeleton: bool,
        show_mesh: bool,
    ) -> torch.Tensor:
        """Generate debug overlay with mesh and skeleton."""
        
        # Convert frame to numpy
        frame_np = frame.cpu().numpy() if hasattr(frame, 'cpu') else np.array(frame)
        if frame_np.max() <= 1.0:
            frame_np = (frame_np * 255).astype(np.uint8)
        else:
            frame_np = frame_np.astype(np.uint8)
        
        overlay = frame_np.copy()
        H, W = frame_np.shape[:2]
        
        # Get vertices for projection
        vertices = output.get("vertices")
        if vertices is not None:
            if hasattr(vertices, 'cpu'):
                vertices = vertices.cpu().numpy()
            if vertices.ndim == 3:
                vertices = vertices[0]  # Remove batch dim
        
        # Get joints
        joints = output.get("joint_coords") or output.get("joints_3d") or output.get("keypoints_3d")
        if joints is not None:
            if hasattr(joints, 'cpu'):
                joints = joints.cpu().numpy()
            if joints.ndim == 3:
                joints = joints[0]
        
        # Get projection parameters
        focal_length = output.get("focal_length", W)
        if hasattr(focal_length, 'item'):
            focal_length = focal_length.item()
        
        pred_cam_t = output.get("pred_cam_t")
        if pred_cam_t is not None:
            if hasattr(pred_cam_t, 'cpu'):
                pred_cam_t = pred_cam_t.cpu().numpy()
            if pred_cam_t.ndim > 1:
                pred_cam_t = pred_cam_t[0]
        
        # Project and draw mesh
        if show_mesh and vertices is not None:
            overlay = self._draw_mesh_wireframe(
                overlay, vertices, focal_length, pred_cam_t, W, H, opacity
            )
        
        # Project and draw skeleton
        if show_skeleton and joints is not None:
            overlay = self._draw_skeleton(
                overlay, joints, focal_length, pred_cam_t, W, H
            )
        
        # Add info text
        focal_mm = focal_length * 36.0 / W  # Assume full frame
        cv2.putText(overlay, f"Focal: {focal_length:.0f}px ({focal_mm:.1f}mm)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(overlay, f"FOV: {2*np.degrees(np.arctan(W/(2*focal_length))):.1f} deg", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if pred_cam_t is not None:
            cv2.putText(overlay, f"cam_t: [{pred_cam_t[0]:.2f}, {pred_cam_t[1]:.2f}, {pred_cam_t[2]:.2f}]", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Convert back to tensor
        overlay_tensor = torch.from_numpy(overlay).float() / 255.0
        overlay_tensor = overlay_tensor.unsqueeze(0)  # Add batch dimension
        
        return overlay_tensor
    
    def _draw_mesh_wireframe(
        self,
        image: np.ndarray,
        vertices: np.ndarray,
        focal_length: float,
        pred_cam_t: Optional[np.ndarray],
        W: int, H: int,
        opacity: float,
    ) -> np.ndarray:
        """Draw mesh wireframe projection."""
        
        # Project vertices to 2D
        pts_2d = self._project_points(vertices, focal_length, pred_cam_t, W, H)
        
        if pts_2d is None:
            return image
        
        # Create mesh overlay
        overlay = image.copy()
        
        # Draw points (subsample for speed)
        for i in range(0, len(pts_2d), 10):
            x, y = int(pts_2d[i, 0]), int(pts_2d[i, 1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(overlay, (x, y), 1, (100, 255, 100), -1)
        
        # Blend
        image = cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0)
        
        return image
    
    def _draw_skeleton(
        self,
        image: np.ndarray,
        joints: np.ndarray,
        focal_length: float,
        pred_cam_t: Optional[np.ndarray],
        W: int, H: int,
    ) -> np.ndarray:
        """Draw skeleton joints and connections."""
        
        # Project joints to 2D
        pts_2d = self._project_points(joints, focal_length, pred_cam_t, W, H)
        
        if pts_2d is None:
            return image
        
        # SMPL joint connections (simplified)
        connections = [
            (0, 1), (0, 2), (1, 4), (2, 5), (4, 7), (5, 8),  # Legs
            (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),  # Spine to head
            (9, 13), (13, 16), (16, 18), (18, 20),  # Left arm
            (9, 14), (14, 17), (17, 19), (19, 21),  # Right arm
        ]
        
        # Draw connections
        for i, j in connections:
            if i < len(pts_2d) and j < len(pts_2d):
                pt1 = (int(pts_2d[i, 0]), int(pts_2d[i, 1]))
                pt2 = (int(pts_2d[j, 0]), int(pts_2d[j, 1]))
                if (0 <= pt1[0] < W and 0 <= pt1[1] < H and
                    0 <= pt2[0] < W and 0 <= pt2[1] < H):
                    cv2.line(image, pt1, pt2, (255, 255, 0), 2)
        
        # Draw joints
        for i, pt in enumerate(pts_2d):
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < W and 0 <= y < H:
                color = (0, 0, 255) if i in [0, 3, 6, 9, 12, 15] else (255, 0, 0)  # Spine=red, limbs=blue
                cv2.circle(image, (x, y), 5, color, -1)
                cv2.circle(image, (x, y), 5, (255, 255, 255), 1)
        
        return image
    
    def _project_points(
        self,
        points_3d: np.ndarray,
        focal_length: float,
        pred_cam_t: Optional[np.ndarray],
        W: int, H: int,
    ) -> Optional[np.ndarray]:
        """Project 3D points to 2D image coordinates."""
        
        if points_3d is None or len(points_3d) == 0:
            return None
        
        # Apply camera translation if available
        if pred_cam_t is not None:
            # pred_cam_t is [tx, ty, tz] - weak perspective camera
            # tx, ty are normalized offsets, tz is depth scale
            tx, ty, tz = pred_cam_t[0], pred_cam_t[1], pred_cam_t[2]
            
            # Scale and translate
            points = points_3d.copy()
            points[:, 0] = points[:, 0] + tx
            points[:, 1] = points[:, 1] + ty
            points[:, 2] = points[:, 2] + tz
        else:
            points = points_3d.copy()
            points[:, 2] = points[:, 2] + 5.0  # Default depth
        
        # Perspective projection
        z = points[:, 2:3]
        z = np.maximum(z, 0.1)  # Avoid division by zero
        
        pts_2d = np.zeros((len(points), 2))
        pts_2d[:, 0] = points[:, 0] * focal_length / z[:, 0] + W / 2
        pts_2d[:, 1] = points[:, 1] * focal_length / z[:, 0] + H / 2
        
        return pts_2d
    
    def _fallback_result(
        self,
        images: torch.Tensor,
        frame_idx: int,
        sensor_width_mm: float,
        error_msg: str,
    ) -> Tuple[Dict, torch.Tensor, float, str]:
        """Return fallback result when SAM3DBody fails."""
        
        H, W = images.shape[1], images.shape[2]
        focal_px = float(W)  # Fallback: focal = width
        focal_mm = focal_px * sensor_width_mm / W
        
        intrinsics = {
            "focal_px": focal_px,
            "focal_mm": focal_mm,
            "sensor_width_mm": sensor_width_mm,
            "cx": W / 2,
            "cy": H / 2,
            "width": W,
            "height": H,
            "fov_x_deg": 2 * np.degrees(np.arctan(W / (2 * focal_px))),
            "fov_y_deg": 2 * np.degrees(np.arctan(H / (2 * focal_px))),
            "aspect_ratio": W / H,
            "source": "fallback",
            "confidence": 0.3,
            "k_matrix": [
                [focal_px, 0.0, W / 2],
                [0.0, focal_px, H / 2],
                [0.0, 0.0, 1.0]
            ],
            "reference_frame": frame_idx,
            "error": error_msg,
            "per_frame": None,
            "is_variable": False,
        }
        
        # Return the reference frame as-is for debug
        debug_frame = images[frame_idx:frame_idx+1].clone()
        
        # Add error text
        frame_np = (debug_frame[0].cpu().numpy() * 255).astype(np.uint8)
        cv2.putText(frame_np, f"FALLBACK: {error_msg}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame_np, f"Using heuristic: {focal_mm:.1f}mm", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        debug_frame = torch.from_numpy(frame_np).float() / 255.0
        debug_frame = debug_frame.unsqueeze(0)
        
        status = f"FALLBACK: {focal_mm:.1f}mm (heuristic) - {error_msg}"
        
        return (intrinsics, debug_frame, focal_mm, status)


# Node registration
NODE_CLASS_MAPPINGS = {
    "IntrinsicsFromSAM3DBody": IntrinsicsFromSAM3DBody,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IntrinsicsFromSAM3DBody": "📷 Intrinsics from SAM3DBody (Single Frame)",
}
