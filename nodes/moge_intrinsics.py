"""
MoGe2 Camera Intrinsics Estimation Node

Uses Microsoft's MoGe2 model to estimate accurate camera intrinsics from single images.
MoGe2 provides superior focal length and principal point estimation compared to simple heuristics.

This is useful for:
1. When camera metadata is unavailable
2. When COLMAP fails to get good intrinsics
3. Improving accuracy of body-to-camera alignment in exports

Installation:
    pip install moge

Usage:
    Connect images -> MoGe2 Intrinsics -> CAMERA_DATA output can be used with other nodes
"""

import numpy as np
import torch
from typing import Dict, Tuple, Any, Optional, List


# Check for MoGe availability
MOGE_AVAILABLE = False
try:
    from moge.model.v2 import MoGeModel
    MOGE_AVAILABLE = True
    print("[MoGeIntrinsics] MoGe2 available")
except ImportError:
    print("[MoGeIntrinsics] MoGe2 not installed. Install with: pip install moge")
except Exception as e:
    print(f"[MoGeIntrinsics] MoGe2 import error: {e}")


class MoGe2IntrinsicsEstimator:
    """
    Estimate camera intrinsics using MoGe2 monocular geometry estimation.
    
    MoGe2 predicts:
    - Normalized 3x3 intrinsics matrix (focal length + principal point)
    - Metric depth map
    - Point map in camera space
    
    This node extracts the intrinsics for use with camera tracking and export.
    """
    
    MODELS = [
        "Ruicheng/moge-2-vitl-normal",
        "Ruicheng/moge-2-vitl",
        "Ruicheng/moge-vitl"  # MoGe-1 fallback
    ]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "model_name": (cls.MODELS, {
                    "default": "Ruicheng/moge-2-vitl-normal",
                    "tooltip": "MoGe model to use. moge-2-vitl-normal is recommended for best quality."
                }),
                "sample_frames": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 30,
                    "tooltip": "Number of frames to sample for intrinsics estimation (averaged for stability)"
                }),
                "sensor_width_mm": ("FLOAT", {
                    "default": 36.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Assumed sensor width in mm for focal length conversion"
                }),
            }
        }
    
    RETURN_TYPES = ("CAMERA_INTRINSICS", "FLOAT", "STRING")
    RETURN_NAMES = ("intrinsics", "focal_length_mm", "status")
    FUNCTION = "estimate_intrinsics"
    CATEGORY = "SAM3DBody2abc/Camera"
    
    def __init__(self):
        self.model = None
        self.model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self, model_name: str):
        """Load MoGe model."""
        if self.model is not None and self.model_name == model_name:
            return True
        
        if not MOGE_AVAILABLE:
            print("[MoGeIntrinsics] MoGe2 not available")
            return False
        
        try:
            print(f"[MoGeIntrinsics] Loading {model_name}...")
            self.model = MoGeModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.model_name = model_name
            print(f"[MoGeIntrinsics] Model loaded on {self.device}")
            return True
        except Exception as e:
            print(f"[MoGeIntrinsics] Failed to load model: {e}")
            return False
    
    def estimate_intrinsics(
        self,
        images: torch.Tensor,
        model_name: str = "Ruicheng/moge-2-vitl-normal",
        sample_frames: int = 5,
        sensor_width_mm: float = 36.0,
    ) -> Tuple[Dict, float, str]:
        """Estimate camera intrinsics from images using MoGe2."""
        
        if not MOGE_AVAILABLE:
            return self._fallback_intrinsics(images, sensor_width_mm)
        
        if not self.load_model(model_name):
            return self._fallback_intrinsics(images, sensor_width_mm)
        
        # images: (N, H, W, C) in range [0, 1]
        num_frames = images.shape[0]
        H, W = images.shape[1], images.shape[2]
        
        # Sample frames evenly
        if num_frames <= sample_frames:
            frame_indices = list(range(num_frames))
        else:
            step = num_frames / sample_frames
            frame_indices = [int(i * step) for i in range(sample_frames)]
        
        print(f"[MoGeIntrinsics] Estimating intrinsics from {len(frame_indices)} frames (of {num_frames})...")
        
        all_intrinsics = []
        
        with torch.no_grad():
            for idx in frame_indices:
                # Get frame and convert to (C, H, W) RGB
                frame = images[idx]  # (H, W, C)
                frame_tensor = frame.permute(2, 0, 1).to(self.device)  # (C, H, W)
                
                # Run MoGe inference
                try:
                    output = self.model.infer(frame_tensor)
                    intrinsics = output["intrinsics"].cpu().numpy()  # (3, 3)
                    all_intrinsics.append(intrinsics)
                except Exception as e:
                    print(f"[MoGeIntrinsics] Frame {idx} inference error: {e}")
                    continue
        
        if not all_intrinsics:
            print("[MoGeIntrinsics] No successful inferences, using fallback")
            return self._fallback_intrinsics(images, sensor_width_mm)
        
        # Average intrinsics for stability
        avg_intrinsics = np.mean(all_intrinsics, axis=0)
        
        # Extract parameters from normalized intrinsics
        # MoGe returns normalized intrinsics where fx, fy are normalized by image dimensions
        fx_norm = avg_intrinsics[0, 0]
        fy_norm = avg_intrinsics[1, 1]
        cx_norm = avg_intrinsics[0, 2]
        cy_norm = avg_intrinsics[1, 2]
        
        # Denormalize (MoGe normalizes by image size)
        # The normalization is typically: fx_norm = fx / W, fy_norm = fy / H
        fx_px = fx_norm * W
        fy_px = fy_norm * H
        cx_px = cx_norm * W
        cy_px = cy_norm * H
        
        # Convert to mm using sensor width
        focal_mm = fx_px * sensor_width_mm / W
        
        # Compute FOV for info
        fov_x_deg = 2 * np.degrees(np.arctan(W / (2 * fx_px)))
        fov_y_deg = 2 * np.degrees(np.arctan(H / (2 * fy_px)))
        
        print(f"[MoGeIntrinsics] Results:")
        print(f"  Focal length: {fx_px:.1f}px = {focal_mm:.1f}mm")
        print(f"  Principal point: ({cx_px:.1f}, {cy_px:.1f})")
        print(f"  FOV: {fov_x_deg:.1f}° x {fov_y_deg:.1f}°")
        
        # Build output
        intrinsics_data = {
            "focal_length_px": float(fx_px),
            "focal_length_x_px": float(fx_px),
            "focal_length_y_px": float(fy_px),
            "focal_length_mm": float(focal_mm),
            "principal_point_x": float(cx_px),
            "principal_point_y": float(cy_px),
            "image_width": int(W),
            "image_height": int(H),
            "sensor_width_mm": float(sensor_width_mm),
            "fov_x_deg": float(fov_x_deg),
            "fov_y_deg": float(fov_y_deg),
            "estimation_method": "MoGe2",
            "model": model_name,
            "frames_sampled": len(frame_indices),
            "intrinsics_matrix": avg_intrinsics.tolist(),
        }
        
        status = f"MoGe2: {focal_mm:.1f}mm ({fov_x_deg:.1f}° FOV) from {len(frame_indices)} frames"
        
        return (intrinsics_data, focal_mm, status)
    
    def _fallback_intrinsics(
        self,
        images: torch.Tensor,
        sensor_width_mm: float,
    ) -> Tuple[Dict, float, str]:
        """Fallback intrinsics estimation using simple heuristics."""
        
        H, W = images.shape[1], images.shape[2]
        
        # Common heuristic: focal length ≈ 0.7-1.2x image width
        # Use 1.0x as default (equivalent to ~53° horizontal FOV)
        fx_px = float(W)
        fy_px = float(H) * (W / H)  # Maintain aspect ratio
        cx_px = W / 2.0
        cy_px = H / 2.0
        
        focal_mm = fx_px * sensor_width_mm / W
        fov_x_deg = 2 * np.degrees(np.arctan(W / (2 * fx_px)))
        
        intrinsics_data = {
            "focal_length_px": float(fx_px),
            "focal_length_x_px": float(fx_px),
            "focal_length_y_px": float(fy_px),
            "focal_length_mm": float(focal_mm),
            "principal_point_x": float(cx_px),
            "principal_point_y": float(cy_px),
            "image_width": int(W),
            "image_height": int(H),
            "sensor_width_mm": float(sensor_width_mm),
            "fov_x_deg": float(fov_x_deg),
            "fov_y_deg": float(fov_x_deg),  # Assume square pixels
            "estimation_method": "Fallback (heuristic)",
            "model": None,
            "frames_sampled": 0,
        }
        
        status = f"Fallback: {focal_mm:.1f}mm (heuristic, MoGe2 unavailable)"
        
        return (intrinsics_data, focal_mm, status)


class ApplyIntrinsicsToMeshSequence:
    """
    Apply estimated camera intrinsics to a mesh sequence.
    
    This updates the focal length in the mesh sequence frames to use
    the more accurate MoGe2-estimated intrinsics instead of SAM3DBody's
    default focal length assumption.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE",),
                "intrinsics": ("CAMERA_INTRINSICS",),
            },
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE",)
    RETURN_NAMES = ("mesh_sequence",)
    FUNCTION = "apply_intrinsics"
    CATEGORY = "SAM3DBody2abc/Camera"
    
    def apply_intrinsics(
        self,
        mesh_sequence: Dict,
        intrinsics: Dict,
    ) -> Tuple[Dict]:
        """Apply intrinsics to mesh sequence frames."""
        import copy
        
        # Deep copy to avoid modifying original
        output = copy.deepcopy(mesh_sequence)
        
        focal_px = intrinsics.get("focal_length_px", 1000.0)
        
        frames = output.get("frames", {})
        updated_count = 0
        
        for frame_idx in frames:
            frame = frames[frame_idx]
            old_focal = frame.get("focal_length")
            frame["focal_length"] = focal_px
            frame["focal_length_source"] = intrinsics.get("estimation_method", "unknown")
            updated_count += 1
        
        print(f"[ApplyIntrinsics] Updated focal length in {updated_count} frames: {focal_px:.1f}px")
        
        return (output,)


class ApplyIntrinsicsToCameraData:
    """
    Apply estimated camera intrinsics to COLMAP camera data.
    
    This can improve camera alignment when COLMAP's intrinsics estimation
    is less accurate than MoGe2's monocular estimation.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "camera_data": ("CAMERA_DATA",),
                "intrinsics": ("CAMERA_INTRINSICS",),
            },
            "optional": {
                "override_mode": (["Replace", "Average", "WeightedAverage"], {
                    "default": "Average",
                    "tooltip": "How to combine MoGe2 intrinsics with existing. Replace: use MoGe2 only. Average: simple average. WeightedAverage: favor COLMAP for multi-view."
                }),
            }
        }
    
    RETURN_TYPES = ("CAMERA_DATA",)
    RETURN_NAMES = ("camera_data",)
    FUNCTION = "apply_intrinsics"
    CATEGORY = "SAM3DBody2abc/Camera"
    
    def apply_intrinsics(
        self,
        camera_data: Any,
        intrinsics: Dict,
        override_mode: str = "Average",
    ) -> Tuple[Any]:
        """Apply intrinsics to camera data."""
        
        moge_focal = intrinsics.get("focal_length_px", 1000.0)
        
        if hasattr(camera_data, 'intrinsics') and camera_data.intrinsics:
            colmap_focal = camera_data.intrinsics.focal_length_x
            
            if override_mode == "Replace":
                new_focal = moge_focal
            elif override_mode == "Average":
                new_focal = (moge_focal + colmap_focal) / 2
            else:  # WeightedAverage - favor COLMAP for multi-view
                weight = min(len(camera_data.extrinsics) / 10, 0.8)  # Max 80% weight to COLMAP
                new_focal = weight * colmap_focal + (1 - weight) * moge_focal
            
            camera_data.intrinsics.focal_length_x = new_focal
            camera_data.intrinsics.focal_length_y = new_focal
            
            print(f"[ApplyIntrinsics] COLMAP: {colmap_focal:.1f}px, MoGe2: {moge_focal:.1f}px -> {new_focal:.1f}px ({override_mode})")
        else:
            print(f"[ApplyIntrinsics] No existing intrinsics, using MoGe2: {moge_focal:.1f}px")
        
        return (camera_data,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "MoGe2IntrinsicsEstimator": MoGe2IntrinsicsEstimator,
    "ApplyIntrinsicsToMeshSequence": ApplyIntrinsicsToMeshSequence,
    "ApplyIntrinsicsToCameraData": ApplyIntrinsicsToCameraData,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoGe2IntrinsicsEstimator": "📷 MoGe2 Intrinsics Estimator",
    "ApplyIntrinsicsToMeshSequence": "📷 Apply Intrinsics to Mesh",
    "ApplyIntrinsicsToCameraData": "📷 Apply Intrinsics to Camera",
}
