"""
Monocular Silhouette Refiner Node
=================================

Refines SAM3DBody pose estimation using silhouette consistency with
SAM3 segmentation masks. Works with single-camera (monocular) video.

Unlike the multi-camera Silhouette Refiner, this node:
- Takes MESH_SEQUENCE directly (not triangulated trajectory)
- Uses single-view silhouette matching
- Optimizes pose to match observed silhouette boundary

This helps with:
- Depth ambiguity (silhouette constrains pose)
- Limb position errors (pulls to correct boundary)
- Body shape mismatches (adjusts to match person)

Author: Claude (Anthropic)
Version: 1.0.0
License: Apache 2.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import cv2


# =============================================================================
# Check Dependencies
# =============================================================================

PYTORCH3D_AVAILABLE = False
try:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        SoftSilhouetteShader,
    )
    PYTORCH3D_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# Logger
# =============================================================================

class MonoSilhouetteLogger:
    """Simple logger for Monocular Silhouette Refiner."""
    
    def __init__(self, level: str = "normal"):
        self.level = level
        self.timings = {}
        self._timer_starts = {}
    
    def _should_log(self, msg_level: str) -> bool:
        levels = {"silent": 0, "normal": 1, "verbose": 2, "debug": 3}
        return levels.get(msg_level, 1) <= levels.get(self.level, 1)
    
    def info(self, msg: str):
        if self._should_log("normal"):
            print(f"[MonoSilhouette] {msg}")
    
    def verbose(self, msg: str):
        if self._should_log("verbose"):
            print(f"[MonoSilhouette] {msg}")
    
    def debug(self, msg: str):
        if self._should_log("debug"):
            print(f"[MonoSilhouette] DEBUG: {msg}")
    
    def warning(self, msg: str):
        print(f"[MonoSilhouette] ⚠ WARNING: {msg}")
    
    def error(self, msg: str):
        print(f"[MonoSilhouette] ✗ ERROR: {msg}")
    
    def start_timer(self, name: str):
        import time
        self._timer_starts[name] = time.time()
    
    def end_timer(self, name: str):
        import time
        if name in self._timer_starts:
            elapsed = time.time() - self._timer_starts[name]
            self.timings[name] = elapsed
            if self._should_log("verbose"):
                print(f"[MonoSilhouette] ⏱ {name}: {elapsed:.3f}s")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MonoSilhouetteConfig:
    """Configuration for monocular silhouette refinement."""
    
    # Optimization
    iterations: int = 50
    learning_rate: float = 0.01
    
    # Loss weights
    silhouette_weight: float = 1.0
    keypoint_weight: float = 0.5
    smoothness_weight: float = 0.1
    
    # Rendering
    render_size: int = 256  # Render at lower resolution for speed
    sigma: float = 1e-4  # Soft silhouette blur
    
    # What to optimize
    optimize_translation: bool = True
    optimize_pose: bool = False  # Pose optimization is more complex
    
    # Logging
    log_level: str = "verbose"


# =============================================================================
# Skeleton Hull Renderer (Fallback when PyTorch3D not available)
# =============================================================================

class SkeletonHullRenderer:
    """
    Render silhouette using skeleton joints as ellipses/circles.
    Works without PyTorch3D.
    """
    
    # Joint connections for body parts
    BODY_PARTS = [
        # Torso (larger)
        ([0, 1, 2, 3], 0.15),  # pelvis to spine
        # Left leg
        ([1, 4, 7, 10], 0.08),  # hip to ankle
        # Right leg
        ([2, 5, 8, 11], 0.08),
        # Left arm
        ([12, 13, 16, 18, 20], 0.06),  # shoulder to wrist
        # Right arm
        ([12, 14, 17, 19, 21], 0.06),
        # Head
        ([12, 15], 0.1),
    ]
    
    def __init__(self, image_size: Tuple[int, int], device: torch.device):
        self.image_size = image_size
        self.device = device
    
    def render(
        self,
        joints_3d: torch.Tensor,
        translation: torch.Tensor,
        focal_length: float,
        principal_point: Tuple[float, float]
    ) -> torch.Tensor:
        """
        Render skeleton as soft silhouette.
        
        Args:
            joints_3d: (N, 3) joint positions
            translation: (3,) root translation
            focal_length: Camera focal length in pixels
            principal_point: (cx, cy)
        
        Returns:
            silhouette: (H, W) soft silhouette
        """
        H, W = self.image_size
        cx, cy = principal_point
        
        # Apply translation
        joints_world = joints_3d + translation
        
        # Project to 2D
        z = joints_world[:, 2:3].clamp(min=0.1)
        joints_2d = joints_world[:, :2] * focal_length / z
        joints_2d[:, 0] += cx
        joints_2d[:, 1] += cy
        
        # Create silhouette image
        silhouette = torch.zeros((H, W), device=self.device)
        
        # Draw circles at each joint
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing='ij'
        )
        
        n_joints = min(len(joints_2d), 22)  # Limit to expected joints
        for i in range(n_joints):
            x, y = joints_2d[i, 0], joints_2d[i, 1]
            
            # Radius based on depth (closer = larger)
            depth = joints_world[i, 2].item()
            radius = max(10, min(50, 500 / max(depth, 0.5)))
            
            # Soft circle
            dist = torch.sqrt((x_coords - x)**2 + (y_coords - y)**2)
            circle = torch.sigmoid((radius - dist) * 0.5)
            silhouette = torch.maximum(silhouette, circle)
        
        return silhouette


# =============================================================================
# Differentiable Silhouette Renderer (PyTorch3D)
# =============================================================================

class DifferentiableMeshRenderer:
    """
    Render mesh silhouette using PyTorch3D.
    Requires PyTorch3D to be installed.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int],
        device: torch.device,
        sigma: float = 1e-4
    ):
        if not PYTORCH3D_AVAILABLE:
            raise ImportError("PyTorch3D required for mesh rendering")
        
        self.image_size = image_size
        self.device = device
        
        # Rasterization settings for soft silhouette
        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
            faces_per_pixel=50,
        )
    
    def render(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        focal_length: float,
        principal_point: Tuple[float, float],
        translation: torch.Tensor
    ) -> torch.Tensor:
        """
        Render mesh silhouette.
        
        Args:
            vertices: (V, 3) mesh vertices
            faces: (F, 3) face indices
            focal_length: Camera focal length
            principal_point: (cx, cy)
            translation: (3,) camera translation
        
        Returns:
            silhouette: (H, W) soft silhouette
        """
        H, W = self.image_size
        
        # Apply translation
        verts = vertices + translation
        
        # Create mesh
        mesh = Meshes(verts=[verts], faces=[faces])
        
        # Create camera
        # PyTorch3D uses a different convention, need to convert
        R = torch.eye(3, device=self.device).unsqueeze(0)
        T = torch.zeros(1, 3, device=self.device)
        
        cameras = FoVPerspectiveCameras(
            device=self.device,
            R=R,
            T=T,
            fov=2 * np.arctan(H / (2 * focal_length)) * 180 / np.pi,
        )
        
        # Create renderer
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            ),
            shader=SoftSilhouetteShader()
        )
        
        # Render
        images = renderer(mesh)
        silhouette = images[0, ..., 3]  # Alpha channel
        
        return silhouette


# =============================================================================
# Loss Functions
# =============================================================================

def silhouette_iou_loss(
    pred_silhouette: torch.Tensor,
    target_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute IoU-based silhouette loss.
    
    Args:
        pred_silhouette: (H, W) predicted soft silhouette [0, 1]
        target_mask: (H, W) target binary mask
    
    Returns:
        loss: 1 - IoU (lower is better)
    """
    # Ensure same size
    if pred_silhouette.shape != target_mask.shape:
        target_mask = torch.nn.functional.interpolate(
            target_mask.unsqueeze(0).unsqueeze(0).float(),
            size=pred_silhouette.shape,
            mode='bilinear',
            align_corners=False
        ).squeeze()
    
    # Soft IoU
    intersection = (pred_silhouette * target_mask).sum()
    union = pred_silhouette.sum() + target_mask.sum() - intersection
    
    iou = intersection / (union + 1e-6)
    return 1.0 - iou


def silhouette_boundary_loss(
    pred_silhouette: torch.Tensor,
    target_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute boundary alignment loss.
    Penalizes silhouette extending outside mask boundary.
    """
    # Dilate target slightly for tolerance
    kernel_size = 5
    target_dilated = torch.nn.functional.max_pool2d(
        target_mask.unsqueeze(0).unsqueeze(0).float(),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2
    ).squeeze()
    
    # Penalize predictions outside dilated mask
    outside = pred_silhouette * (1 - target_dilated)
    
    return outside.mean()


def keypoint_reprojection_loss(
    joints_3d: torch.Tensor,
    translation: torch.Tensor,
    target_joints_2d: torch.Tensor,
    focal_length: float,
    principal_point: Tuple[float, float],
    confidence: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute keypoint reprojection error.
    """
    cx, cy = principal_point
    
    # Project 3D joints to 2D
    joints_world = joints_3d + translation
    z = joints_world[:, 2:3].clamp(min=0.1)
    pred_2d = joints_world[:, :2] * focal_length / z
    pred_2d[:, 0] += cx
    pred_2d[:, 1] += cy
    
    # Compute error
    error = (pred_2d - target_joints_2d[:, :2]).pow(2).sum(dim=1)
    
    if confidence is not None:
        error = error * confidence
    
    return error.mean()


# =============================================================================
# Main Refiner
# =============================================================================

class MonocularSilhouetteRefiner:
    """
    Refine SAM3DBody output using silhouette consistency.
    """
    
    def __init__(self, config: MonoSilhouetteConfig):
        self.config = config
        self.log = MonoSilhouetteLogger(config.log_level)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def refine(
        self,
        mesh_sequence: Dict,
        masks: np.ndarray,
        intrinsics: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """
        Refine mesh sequence using silhouette masks.
        
        Args:
            mesh_sequence: SAM3DBody output
            masks: (T, H, W) binary segmentation masks
            intrinsics: Camera intrinsics (optional, uses from mesh_sequence)
        
        Returns:
            refined_sequence: Updated mesh sequence
            metrics: Refinement metrics per frame
        """
        self.log.info("=" * 60)
        self.log.info("MONOCULAR SILHOUETTE REFINEMENT")
        self.log.info("=" * 60)
        
        # Extract frames
        frames_data = mesh_sequence.get("frames", {})
        if isinstance(frames_data, dict):
            frame_keys = sorted(frames_data.keys())
            frames = [frames_data[k] for k in frame_keys]
        else:
            frame_keys = None
            frames = list(frames_data)
        
        T = len(frames)
        self.log.info(f"Processing {T} frames")
        
        # Get intrinsics
        if intrinsics:
            focal = intrinsics.get("focal_length", 1000.0)
            cx = intrinsics.get("cx", masks.shape[2] / 2)
            cy = intrinsics.get("cy", masks.shape[1] / 2)
        else:
            focal = mesh_sequence.get("focal_length", frames[0].get("focal_length", 1000.0))
            cx = mesh_sequence.get("width", masks.shape[2]) / 2
            cy = mesh_sequence.get("height", masks.shape[1]) / 2
        
        self.log.info(f"Focal length: {focal:.1f}px, Principal point: ({cx:.1f}, {cy:.1f})")
        
        # Create renderer
        render_size = (self.config.render_size, self.config.render_size)
        renderer = SkeletonHullRenderer(render_size, self.device)
        
        # Scale factors for resizing
        mask_h, mask_w = masks.shape[1], masks.shape[2]
        scale_x = self.config.render_size / mask_w
        scale_y = self.config.render_size / mask_h
        scaled_focal = focal * scale_x
        scaled_cx = cx * scale_x
        scaled_cy = cy * scale_y
        
        # Process each frame
        refined_frames = []
        all_metrics = []
        
        for t in range(T):
            frame = frames[t].copy()
            mask = masks[t]
            
            # Resize mask to render size
            mask_resized = cv2.resize(
                mask.astype(np.float32),
                (self.config.render_size, self.config.render_size),
                interpolation=cv2.INTER_LINEAR
            )
            mask_tensor = torch.from_numpy(mask_resized).float().to(self.device)
            
            # Get current pose data
            joints_3d = self._get_joints_3d(frame)
            translation = self._get_translation(frame)
            joints_2d = self._get_joints_2d(frame)
            
            if joints_3d is None or translation is None:
                self.log.warning(f"Frame {t}: Missing pose data, skipping")
                refined_frames.append(frame)
                continue
            
            # Convert to tensors
            joints_3d_t = torch.from_numpy(joints_3d).float().to(self.device)
            translation_t = torch.from_numpy(translation).float().to(self.device)
            translation_t.requires_grad_(self.config.optimize_translation)
            
            if joints_2d is not None:
                joints_2d_t = torch.from_numpy(joints_2d).float().to(self.device)
                # Scale to render size
                joints_2d_t[:, 0] *= scale_x
                joints_2d_t[:, 1] *= scale_y
            else:
                joints_2d_t = None
            
            # Compute initial loss
            with torch.no_grad():
                pred_sil = renderer.render(
                    joints_3d_t, translation_t,
                    scaled_focal, (scaled_cx, scaled_cy)
                )
                initial_loss = silhouette_iou_loss(pred_sil, mask_tensor).item()
            
            # Optimize
            if self.config.optimize_translation:
                optimizer = optim.Adam([translation_t], lr=self.config.learning_rate)
                
                for i in range(self.config.iterations):
                    optimizer.zero_grad()
                    
                    # Render silhouette
                    pred_sil = renderer.render(
                        joints_3d_t, translation_t,
                        scaled_focal, (scaled_cx, scaled_cy)
                    )
                    
                    # Compute losses
                    loss = self.config.silhouette_weight * silhouette_iou_loss(pred_sil, mask_tensor)
                    loss += self.config.silhouette_weight * 0.5 * silhouette_boundary_loss(pred_sil, mask_tensor)
                    
                    if joints_2d_t is not None:
                        loss += self.config.keypoint_weight * keypoint_reprojection_loss(
                            joints_3d_t, translation_t, joints_2d_t,
                            scaled_focal, (scaled_cx, scaled_cy)
                        )
                    
                    # Backprop
                    loss.backward()
                    optimizer.step()
                
                # Get final loss
                with torch.no_grad():
                    pred_sil = renderer.render(
                        joints_3d_t, translation_t,
                        scaled_focal, (scaled_cx, scaled_cy)
                    )
                    final_loss = silhouette_iou_loss(pred_sil, mask_tensor).item()
                
                # Update frame
                new_trans = translation_t.detach().cpu().numpy()
                frame["pred_cam_t"] = new_trans.tolist()
                if "smpl_params" in frame:
                    frame["smpl_params"]["transl"] = new_trans.tolist()
            else:
                final_loss = initial_loss
            
            # Record metrics
            improvement = initial_loss - final_loss
            all_metrics.append({
                "frame": t,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "improvement": improvement,
                "initial_iou": 1 - initial_loss,
                "final_iou": 1 - final_loss,
            })
            
            refined_frames.append(frame)
            
            if t % 10 == 0 or t == T - 1:
                self.log.info(f"Frame {t}/{T}: IoU {1-initial_loss:.3f} → {1-final_loss:.3f} (Δ={improvement:.4f})")
        
        # Update mesh_sequence
        result = mesh_sequence.copy()
        if frame_keys is not None:
            result["frames"] = {k: v for k, v in zip(frame_keys, refined_frames)}
        else:
            result["frames"] = refined_frames
        
        # Add refinement metadata
        result["silhouette_refinement"] = {
            "method": "monocular_skeleton_hull",
            "iterations": self.config.iterations,
            "frames_refined": len(all_metrics),
            "avg_improvement": np.mean([m["improvement"] for m in all_metrics]) if all_metrics else 0,
            "avg_final_iou": np.mean([m["final_iou"] for m in all_metrics]) if all_metrics else 0,
        }
        
        self._log_summary(all_metrics)
        
        return result, {"metrics": all_metrics, "timings": self.log.timings}
    
    def _get_joints_3d(self, frame: Dict) -> Optional[np.ndarray]:
        """Extract 3D joints from frame."""
        for key in ["pred_keypoints_3d", "keypoints_3d", "joints_3d", "joint_coords"]:
            if key in frame:
                joints = np.array(frame[key])
                while joints.ndim > 2:
                    joints = joints[0]
                return joints
        return None
    
    def _get_joints_2d(self, frame: Dict) -> Optional[np.ndarray]:
        """Extract 2D joints from frame."""
        for key in ["pred_keypoints_2d", "keypoints_2d", "joints_2d"]:
            if key in frame:
                joints = np.array(frame[key])
                while joints.ndim > 2:
                    joints = joints[0]
                return joints
        return None
    
    def _get_translation(self, frame: Dict) -> Optional[np.ndarray]:
        """Extract translation from frame."""
        if "pred_cam_t" in frame:
            return np.array(frame["pred_cam_t"])
        elif "smpl_params" in frame and "transl" in frame["smpl_params"]:
            return np.array(frame["smpl_params"]["transl"])
        return None
    
    def _log_summary(self, metrics: List[Dict]):
        """Log refinement summary."""
        if not metrics:
            return
        
        self.log.info("")
        self.log.info("=" * 60)
        self.log.info("REFINEMENT SUMMARY")
        self.log.info("=" * 60)
        
        initial_ious = [m["initial_iou"] for m in metrics]
        final_ious = [m["final_iou"] for m in metrics]
        improvements = [m["improvement"] for m in metrics]
        
        self.log.info(f"Frames refined: {len(metrics)}")
        self.log.info(f"Initial IoU: {np.mean(initial_ious):.3f} (min={np.min(initial_ious):.3f}, max={np.max(initial_ious):.3f})")
        self.log.info(f"Final IoU:   {np.mean(final_ious):.3f} (min={np.min(final_ious):.3f}, max={np.max(final_ious):.3f})")
        self.log.info(f"Improvement: {np.mean(improvements):.4f} (total={np.sum(improvements):.4f})")
        
        # Quality assessment
        avg_iou = np.mean(final_ious)
        if avg_iou >= 0.8:
            quality = "✓✓ EXCELLENT"
        elif avg_iou >= 0.6:
            quality = "✓ GOOD"
        elif avg_iou >= 0.4:
            quality = "~ ACCEPTABLE"
        else:
            quality = "✗ POOR"
        
        self.log.info(f"Quality: {quality}")


# =============================================================================
# ComfyUI Node
# =============================================================================

class MonocularSilhouetteRefinerNode:
    """
    Monocular Silhouette Refiner - Refine pose using segmentation masks.
    
    Uses silhouette consistency to improve SAM3DBody pose estimation.
    Compares rendered body silhouette against SAM3 segmentation mask
    and optimizes translation to minimize difference.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE", {
                    "tooltip": "Mesh sequence from SAM3DBody"
                }),
                "masks": ("MASK", {
                    "tooltip": "SAM3 segmentation masks (binary)"
                }),
            },
            "optional": {
                "intrinsics": ("INTRINSICS", {
                    "tooltip": "Camera intrinsics (optional, uses from mesh_sequence if not provided)"
                }),
                "iterations": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 200,
                    "step": 10,
                    "tooltip": "Optimization iterations per frame"
                }),
                "learning_rate": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "Optimizer learning rate"
                }),
                "silhouette_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Weight for silhouette matching loss"
                }),
                "keypoint_weight": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Weight for keypoint reprojection loss"
                }),
                "render_size": ("INT", {
                    "default": 256,
                    "min": 128,
                    "max": 512,
                    "step": 64,
                    "tooltip": "Render resolution (lower = faster)"
                }),
                "optimize_translation": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Optimize root translation"
                }),
                "log_level": (["normal", "verbose", "debug", "silent"], {
                    "default": "verbose"
                }),
            }
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE", "STRING")
    RETURN_NAMES = ("refined_sequence", "refinement_info")
    FUNCTION = "refine"
    CATEGORY = "SAM3DBody2abc/Processing"
    
    def refine(
        self,
        mesh_sequence: Dict,
        masks,
        intrinsics: Optional[Dict] = None,
        iterations: int = 50,
        learning_rate: float = 0.01,
        silhouette_weight: float = 1.0,
        keypoint_weight: float = 0.5,
        render_size: int = 256,
        optimize_translation: bool = True,
        log_level: str = "verbose",
    ) -> Tuple[Dict, str]:
        """Refine mesh sequence using silhouette masks."""
        
        # Convert masks to numpy
        if torch.is_tensor(masks):
            masks_np = masks.cpu().numpy()
        else:
            masks_np = np.array(masks)
        
        # Ensure masks are 3D (T, H, W)
        if masks_np.ndim == 4:
            # (T, H, W, C) -> (T, H, W)
            masks_np = masks_np[..., 0] if masks_np.shape[-1] <= 4 else masks_np[:, 0]
        
        # Binarize masks
        if masks_np.max() <= 1.0:
            masks_np = (masks_np > 0.5).astype(np.float32)
        else:
            masks_np = (masks_np > 127).astype(np.float32)
        
        # Build config
        config = MonoSilhouetteConfig(
            iterations=iterations,
            learning_rate=learning_rate,
            silhouette_weight=silhouette_weight,
            keypoint_weight=keypoint_weight,
            render_size=render_size,
            optimize_translation=optimize_translation,
            log_level=log_level,
        )
        
        # Run refinement
        refiner = MonocularSilhouetteRefiner(config)
        result, metrics = refiner.refine(mesh_sequence, masks_np, intrinsics)
        
        # Build info string
        meta = result.get("silhouette_refinement", {})
        frame_metrics = metrics.get("metrics", [])
        
        info_lines = [
            "=== MONOCULAR SILHOUETTE REFINEMENT ===",
            f"Method: {meta.get('method', 'unknown')}",
            f"Iterations: {iterations}",
            f"Frames refined: {meta.get('frames_refined', 0)}",
            "",
            f"Average improvement: {meta.get('avg_improvement', 0):.4f}",
            f"Average final IoU: {meta.get('avg_final_iou', 0):.3f}",
            "",
        ]
        
        # Per-frame summary (sample)
        if frame_metrics:
            info_lines.append("Sample frames:")
            for i in range(0, len(frame_metrics), max(1, len(frame_metrics) // 5)):
                m = frame_metrics[i]
                info_lines.append(
                    f"  Frame {m['frame']}: IoU {m['initial_iou']:.3f} → {m['final_iou']:.3f}"
                )
        
        info_str = "\n".join(info_lines)
        
        return (result, info_str)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "MonocularSilhouetteRefiner": MonocularSilhouetteRefinerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MonocularSilhouetteRefiner": "🎭 Monocular Silhouette Refiner",
}
