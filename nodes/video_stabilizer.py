"""
Video Stabilizer Node - Phase 2 of v5.0 Pipeline

Uses PyTorch grid_sample for warping with temporal smoothing filters
to produce smooth, jitter-free stabilization.

Smoothing Options:
- Gaussian: Simple blur, good for general smoothing
- Savitzky-Golay: Preserves motion details while removing noise
- Butterworth: Frequency-domain filter, removes high-frequency jitter

Version: 5.0.1
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp
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
    Uses PyTorch grid_sample with temporal smoothing for jitter-free results.
    """
    
    CATEGORY = "SAM3DBody2abc/Camera"
    FUNCTION = "stabilize"
    
    BORDER_MODES = ["replicate", "reflect", "zeros", "crop"]
    REFERENCE_FRAMES = ["first", "middle", "last", "custom"]
    SMOOTHING_FILTERS = ["none", "gaussian", "savgol", "butterworth"]
    
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
                
                # === Temporal Smoothing ===
                "smoothing_filter": (cls.SMOOTHING_FILTERS, {
                    "default": "gaussian",
                    "tooltip": "Filter type for temporal smoothing of camera motion"
                }),
                "smoothing_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Smoothing strength: 0=no smoothing, 1=maximum smoothing"
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
        smoothing_filter: str = "gaussian",
        smoothing_strength: float = 0.5,
        generate_comparison: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Stabilize video by applying inverse camera transforms with temporal smoothing."""
        
        log_info(f"{'='*60}")
        log_info(f"Stabilizing {images.shape[0]} frames")
        log_info(f"Reference: {reference_frame}, Border: {border_mode}")
        log_info(f"Smoothing: {smoothing_filter} (strength={smoothing_strength:.2f})")
        log_info(f"{'='*60}")
        
        # Get dimensions as Python ints
        N = int(images.shape[0])
        H = int(images.shape[1])
        W = int(images.shape[2])
        C = int(images.shape[3])
        device = images.device
        
        log_info(f"Dimensions: N={N}, H={H}, W={W}, C={C}, device={device}")
        
        # Safety checks
        if H <= 0 or W <= 0 or N <= 0:
            log_error(f"Invalid image dimensions")
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
        
        # Extract all rotations
        log_info("Extracting rotations...")
        rotations = []
        for i in range(N):
            R = self._extract_rotation(matrices[i])
            rotations.append(R)
        
        # Get reference rotation
        R_ref = rotations[ref_idx]
        
        # Compute relative rotations (current -> reference)
        log_info("Computing relative rotations...")
        relative_rotations = []
        for i in range(N):
            R_rel = R_ref @ rotations[i].T
            relative_rotations.append(R_rel)
        
        # Apply temporal smoothing to rotations
        if smoothing_filter != "none" and smoothing_strength > 0:
            log_info(f"Applying {smoothing_filter} smoothing...")
            relative_rotations = self._smooth_rotations(
                relative_rotations, 
                smoothing_filter, 
                smoothing_strength,
                N
            )
        
        # Convert images to NCHW format
        images_nchw = images.permute(0, 3, 1, 2).contiguous()
        
        # Pre-compute base grid
        log_info("Creating base sampling grid...")
        y_coords = torch.arange(H, dtype=torch.float32, device=device)
        x_coords = torch.arange(W, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        ones = torch.ones_like(xx)
        base_coords = torch.stack([xx, yy, ones], dim=-1)
        
        # Compute homographies from smoothed rotations
        log_info("Computing homographies from smoothed rotations...")
        homographies = []
        for i in range(N):
            R_rel = relative_rotations[i]
            H_mat = K @ R_rel @ K_inv
            H_mat = H_mat / (H_mat[2, 2] + 1e-10)
            
            # We need inverse for grid_sample
            H_inv = np.linalg.inv(H_mat)
            homographies.append(torch.from_numpy(H_inv).float().to(device))
        
        # Map border mode
        padding_mode_map = {
            "replicate": "border",
            "reflect": "reflection",
            "zeros": "zeros",
            "crop": "zeros",
        }
        padding_mode = padding_mode_map.get(border_mode, "border")
        
        # Apply warps
        log_info("Applying warps...")
        stabilized_frames = []
        
        for i in range(N):
            try:
                frame = images_nchw[i:i+1]
                H_inv = homographies[i]
                
                # Apply homography to get source coordinates
                coords_flat = base_coords.reshape(-1, 3)
                transformed = (H_inv @ coords_flat.T).T
                w = transformed[:, 2:3].clamp(min=1e-8)
                xy = transformed[:, :2] / w
                xy = xy.reshape(H, W, 2)
                
                # Normalize to [-1, 1]
                grid_x = (xy[..., 0] / (W - 1)) * 2 - 1
                grid_y = (xy[..., 1] / (H - 1)) * 2 - 1
                grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                
                # Warp
                warped = F.grid_sample(
                    frame, grid,
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
        stabilized = torch.cat(stabilized_frames, dim=0)
        stabilized = stabilized.permute(0, 2, 3, 1).clamp(0, 1)
        
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
            "smoothing_filter": smoothing_filter,
            "smoothing_strength": smoothing_strength,
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
    
    def _smooth_rotations(
        self,
        rotations: List[np.ndarray],
        filter_type: str,
        strength: float,
        N: int,
    ) -> List[np.ndarray]:
        """
        Apply temporal smoothing to rotation sequence.
        
        Works in axis-angle space for proper rotation interpolation.
        """
        
        # Convert rotations to axis-angle representation
        axis_angles = []
        for R in rotations:
            r = Rotation.from_matrix(R)
            axis_angles.append(r.as_rotvec())
        
        axis_angles = np.array(axis_angles)  # [N, 3]
        
        # Calculate filter parameters based on strength
        # strength 0 = no smoothing, strength 1 = heavy smoothing
        
        if filter_type == "gaussian":
            # Sigma ranges from 0.5 to 5.0 based on strength
            sigma = 0.5 + strength * 4.5
            log_info(f"  Gaussian sigma: {sigma:.2f}")
            
            smoothed = np.zeros_like(axis_angles)
            for i in range(3):
                smoothed[:, i] = gaussian_filter1d(axis_angles[:, i], sigma, mode='nearest')
        
        elif filter_type == "savgol":
            # Window size: 5 to 21 (must be odd)
            window = int(5 + strength * 16)
            if window % 2 == 0:
                window += 1
            window = min(window, N - 1)
            if window % 2 == 0:
                window -= 1
            window = max(window, 5)
            
            # Polynomial order: 2 or 3
            polyorder = min(3, window - 1)
            log_info(f"  Savitzky-Golay window: {window}, order: {polyorder}")
            
            smoothed = np.zeros_like(axis_angles)
            for i in range(3):
                smoothed[:, i] = savgol_filter(axis_angles[:, i], window, polyorder)
        
        elif filter_type == "butterworth":
            # Cutoff frequency: 0.5 to 0.05 (lower = more smoothing)
            cutoff = 0.5 - strength * 0.45
            cutoff = max(cutoff, 0.05)
            log_info(f"  Butterworth cutoff: {cutoff:.3f}")
            
            # Design filter (2nd order)
            b, a = butter(2, cutoff, btype='low')
            
            smoothed = np.zeros_like(axis_angles)
            for i in range(3):
                # filtfilt applies filter forward and backward (zero phase)
                # Pad signal to avoid edge effects
                padlen = min(3 * max(len(a), len(b)), N - 1)
                smoothed[:, i] = filtfilt(b, a, axis_angles[:, i], padlen=padlen)
        
        else:
            return rotations
        
        # Convert back to rotation matrices
        smoothed_rotations = []
        for rotvec in smoothed:
            R = Rotation.from_rotvec(rotvec).as_matrix()
            smoothed_rotations.append(R)
        
        return smoothed_rotations
    
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
        
        fx = float(intrinsics.get("focal_px", W))
        fy = float(intrinsics.get("focal_px", W))
        cx = float(intrinsics.get("cx", W / 2))
        cy = float(intrinsics.get("cy", H / 2))
        
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
        
        # Orthonormalize via SVD
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
        return float(np.degrees(np.arccos(cos_angle)))
    
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
        comparison[:, :, W-1:W+1, :] = 1.0
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
            f"Smoothing: {stabilize_info.get('smoothing_filter', 'none')} "
            f"(strength={stabilize_info.get('smoothing_strength', 0):.0%})",
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
