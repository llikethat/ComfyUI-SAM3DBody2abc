"""
Video Stabilizer Node - Phase 2 of v5.0 Pipeline

This node applies inverse camera transforms to stabilize video footage.
The result is video where the camera appears static, allowing SAM3DBody
to estimate true world-space character motion without camera effects.

Version: 5.0.0 - Using torch grid_sample instead of OpenCV warpPerspective
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Any, Optional, List

def log_info(msg: str):
    print(f"[VideoStabilizer] {msg}")

def log_warning(msg: str):
    print(f"[VideoStabilizer] ⚠️ {msg}")

def log_error(msg: str):
    print(f"[VideoStabilizer] ❌ {msg}")


class VideoStabilizer:
    """
    Apply inverse camera transform to create stabilized footage.
    
    Uses PyTorch grid_sample for warping (avoids OpenCV segfault issues).
    """
    
    CATEGORY = "SAM3DBody2abc/Camera"
    FUNCTION = "stabilize"
    
    BORDER_MODES = ["replicate", "reflect", "zeros"]
    REFERENCE_FRAMES = ["first", "middle", "last", "custom"]
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "STABILIZE_INFO")
    RETURN_NAMES = ("stabilized", "comparison", "stabilize_info")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input video frames [N, H, W, 3]"
                }),
                "camera_matrices": ("CAMERA_MATRICES", {
                    "tooltip": "Camera matrices from CameraSolverV2"
                }),
                "intrinsics": ("INTRINSICS", {
                    "tooltip": "Camera intrinsics for correct warping"
                }),
            },
            "optional": {
                "reference_frame": (cls.REFERENCE_FRAMES, {
                    "default": "first",
                    "tooltip": "Which frame's camera pose to use as reference"
                }),
                "custom_reference": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Custom reference frame index (when reference_frame='custom')"
                }),
                "border_mode": (cls.BORDER_MODES, {
                    "default": "replicate",
                    "tooltip": "How to fill areas outside original frame"
                }),
                "smoothing": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Blend between full stabilization (0) and original (1)"
                }),
                "generate_comparison": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Generate side-by-side comparison video"
                }),
            }
        }
    
    def stabilize(
        self,
        images: torch.Tensor,
        camera_matrices: Dict,
        intrinsics: Dict,
        reference_frame: str = "first",
        custom_reference: int = 0,
        border_mode: str = "replicate",
        smoothing: float = 0.0,
        generate_comparison: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Stabilize video by applying inverse camera transforms."""
        
        log_info(f"{'='*60}")
        log_info(f"Stabilizing {images.shape[0]} frames")
        log_info(f"Reference: {reference_frame}, Border: {border_mode}")
        log_info(f"{'='*60}")
        
        N, H, W, C = images.shape
        device = images.device
        
        # Safety checks
        if H <= 0 or W <= 0 or N <= 0:
            log_error(f"Invalid image dimensions: {images.shape}")
            return self._return_original(images, generate_comparison)
        
        # Extract camera matrices
        matrices = camera_matrices.get("matrices", [])
        if len(matrices) == 0:
            log_error("No camera matrices provided")
            return self._return_original(images, generate_comparison)
        
        if len(matrices) != N:
            log_warning(f"Matrix count ({len(matrices)}) != frame count ({N})")
            matrices = self._interpolate_matrices(matrices, N)
        
        # Get intrinsics matrix K
        try:
            K = self._get_intrinsics_matrix(intrinsics, W, H)
            K_inv = np.linalg.inv(K)
            log_info(f"K matrix focal: {K[0,0]:.1f}px")
        except Exception as e:
            log_error(f"Failed to compute K matrix: {e}")
            return self._return_original(images, generate_comparison)
        
        # Get reference frame index
        ref_idx = self._get_reference_index(reference_frame, custom_reference, N)
        log_info(f"Reference frame: {ref_idx}")
        
        # Get reference rotation
        R_ref = self._extract_rotation(matrices[ref_idx])
        
        # Compute all homographies first
        log_info("Computing homographies...")
        homographies = []
        
        for i in range(N):
            try:
                R_curr = self._extract_rotation(matrices[i])
                R_rel = R_ref @ R_curr.T
                
                # H = K @ R @ K^-1
                H = K @ R_rel @ K_inv
                H = H / (H[2, 2] + 1e-10)
                
                # Validate
                if np.any(np.isnan(H)) or np.any(np.isinf(H)):
                    H = np.eye(3)
                
                # Apply smoothing
                if smoothing > 0:
                    H = (1 - smoothing) * H + smoothing * np.eye(3)
                    H = H / (H[2, 2] + 1e-10)
                
                homographies.append(torch.from_numpy(H).float().to(device))
                
            except Exception as e:
                log_warning(f"Frame {i}: Error computing H - {e}")
                homographies.append(torch.eye(3).float().to(device))
        
        log_info(f"Computed {len(homographies)} homographies")
        
        # Apply warps using torch
        log_info("Applying warps with torch grid_sample...")
        
        # Convert images to NCHW format for grid_sample
        images_nchw = images.permute(0, 3, 1, 2)  # [N, C, H, W]
        
        # Map border mode
        padding_mode_map = {
            "replicate": "border",
            "reflect": "reflection",
            "zeros": "zeros",
        }
        padding_mode = padding_mode_map.get(border_mode, "border")
        
        stabilized_frames = []
        
        for i in range(N):
            try:
                # Get single frame [1, C, H, W]
                frame = images_nchw[i:i+1]
                H_mat = homographies[i]
                
                # Create sampling grid
                grid = self._homography_to_grid(H_mat, H, W, device)
                
                # Apply warp
                warped = F.grid_sample(
                    frame,
                    grid,
                    mode='bilinear',
                    padding_mode=padding_mode,
                    align_corners=True
                )
                
                stabilized_frames.append(warped)
                
                if i % 10 == 0:
                    log_info(f"Frame {i}/{N} warped")
                    
            except Exception as e:
                log_error(f"Frame {i}: Warp error - {e}")
                stabilized_frames.append(images_nchw[i:i+1])
        
        # Stack and convert back to NHWC
        stabilized = torch.cat(stabilized_frames, dim=0)  # [N, C, H, W]
        stabilized = stabilized.permute(0, 2, 3, 1)  # [N, H, W, C]
        
        log_info(f"Stabilized shape: {stabilized.shape}")
        
        # Generate comparison
        if generate_comparison:
            comparison = self._generate_comparison(images, stabilized)
        else:
            comparison = stabilized
        
        # Build info dict
        total_rotation = self._total_rotation(matrices)
        stabilize_info = {
            "num_frames": N,
            "reference_frame": ref_idx,
            "border_mode": border_mode,
            "smoothing": smoothing,
            "total_rotation_deg": total_rotation,
            "intrinsics_used": {
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
            },
        }
        
        log_info(f"✅ Stabilization complete")
        log_info(f"Total camera rotation: {total_rotation:.1f}°")
        
        return (stabilized, comparison, stabilize_info)
    
    def _homography_to_grid(
        self,
        H: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Convert homography to sampling grid for grid_sample.
        
        grid_sample expects coordinates in [-1, 1] range.
        """
        
        # Create coordinate grid
        y_coords = torch.linspace(-1, 1, height, device=device)
        x_coords = torch.linspace(-1, 1, width, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Stack to [H, W, 3] homogeneous coordinates
        ones = torch.ones_like(xx)
        
        # Convert from [-1,1] to pixel coordinates for homography
        xx_px = (xx + 1) / 2 * (width - 1)
        yy_px = (yy + 1) / 2 * (height - 1)
        
        # Apply inverse homography (we want to find source pixels)
        H_inv = torch.inverse(H)
        
        # Flatten for matrix multiplication
        coords = torch.stack([xx_px.flatten(), yy_px.flatten(), ones.flatten()], dim=0)  # [3, H*W]
        
        # Transform
        transformed = H_inv @ coords  # [3, H*W]
        
        # Normalize homogeneous coordinates
        transformed = transformed / (transformed[2:3, :] + 1e-10)
        
        # Convert back to [-1, 1] range
        x_new = transformed[0, :].reshape(height, width)
        y_new = transformed[1, :].reshape(height, width)
        
        x_new = (x_new / (width - 1)) * 2 - 1
        y_new = (y_new / (height - 1)) * 2 - 1
        
        # Stack to grid [1, H, W, 2]
        grid = torch.stack([x_new, y_new], dim=-1).unsqueeze(0)
        
        return grid
    
    def _return_original(
        self,
        images: torch.Tensor,
        generate_comparison: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Return original images when stabilization fails."""
        
        if generate_comparison:
            comparison = self._generate_comparison(images, images)
        else:
            comparison = images
        
        return (images, comparison, {"error": "Stabilization failed"})
    
    def _get_intrinsics_matrix(self, intrinsics: Dict, W: int, H: int) -> np.ndarray:
        """Extract or build K matrix from intrinsics."""
        
        if "k_matrix" in intrinsics:
            return np.array(intrinsics["k_matrix"], dtype=np.float64)
        
        fx = intrinsics.get("focal_px", W)
        fy = intrinsics.get("focal_px", W)
        cx = intrinsics.get("cx", W / 2)
        cy = intrinsics.get("cy", H / 2)
        
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=np.float64)
    
    def _get_reference_index(self, mode: str, custom: int, N: int) -> int:
        """Get reference frame index based on mode."""
        
        if mode == "first":
            return 0
        elif mode == "middle":
            return N // 2
        elif mode == "last":
            return N - 1
        elif mode == "custom":
            return min(max(0, custom), N - 1)
        return 0
    
    def _extract_rotation(self, matrix_data: Dict) -> np.ndarray:
        """Extract 3x3 rotation matrix from camera matrix data."""
        
        if "rotation" in matrix_data:
            R = np.array(matrix_data["rotation"], dtype=np.float64)
        elif "matrix" in matrix_data:
            M = np.array(matrix_data["matrix"], dtype=np.float64)
            R = M[:3, :3]
        else:
            return np.eye(3, dtype=np.float64)
        
        # Orthonormalize
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = -R
        
        return R
    
    def _total_rotation(self, matrices: List[Dict]) -> float:
        """Calculate total rotation from first to last frame."""
        
        if len(matrices) < 2:
            return 0.0
        
        R_first = self._extract_rotation(matrices[0])
        R_last = self._extract_rotation(matrices[-1])
        R_total = R_last @ R_first.T
        
        trace = np.trace(R_total)
        cos_angle = np.clip((trace - 1) / 2, -1, 1)
        return np.degrees(np.arccos(cos_angle))
    
    def _interpolate_matrices(self, matrices: List[Dict], target_count: int) -> List[Dict]:
        """Interpolate matrices to match frame count."""
        
        if len(matrices) == 0:
            return [{"rotation": np.eye(3).tolist()} for _ in range(target_count)]
        
        if len(matrices) >= target_count:
            return matrices[:target_count]
        
        result = []
        for i in range(target_count):
            src_idx = int(i * len(matrices) / target_count)
            result.append(matrices[src_idx])
        return result
    
    def _generate_comparison(
        self,
        original: torch.Tensor,
        stabilized: torch.Tensor,
    ) -> torch.Tensor:
        """Generate side-by-side comparison video."""
        
        N, H, W, C = original.shape
        comparison = torch.zeros((N, H, W * 2, C), dtype=original.dtype, device=original.device)
        comparison[:, :, :W, :] = original
        comparison[:, :, W:, :] = stabilized
        comparison[:, :, W-1:W+1, :] = 1.0  # White separator
        return comparison


class StabilizationInfo:
    """Display stabilization info."""
    
    CATEGORY = "SAM3DBody2abc/Camera"
    FUNCTION = "display"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info_text",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stabilize_info": ("STABILIZE_INFO",),
            }
        }
    
    def display(self, stabilize_info: Dict) -> Tuple[str]:
        lines = [
            "═" * 50,
            "VIDEO STABILIZATION INFO",
            "═" * 50,
            f"Frames: {stabilize_info.get('num_frames', 'N/A')}",
            f"Reference frame: {stabilize_info.get('reference_frame', 'N/A')}",
            f"Border mode: {stabilize_info.get('border_mode', 'N/A')}",
            f"Smoothing: {stabilize_info.get('smoothing', 0):.1%}",
            f"Total rotation: {stabilize_info.get('total_rotation_deg', 0):.1f}°",
            "═" * 50,
        ]
        text = "\n".join(lines)
        print(text)
        return (text,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "VideoStabilizer": VideoStabilizer,
    "StabilizationInfo": StabilizationInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoStabilizer": "🎬 Video Stabilizer",
    "StabilizationInfo": "🎬 Stabilization Info",
}
