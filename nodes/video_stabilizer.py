"""
Video Stabilizer Node - Phase 2 of v5.0 Pipeline

This node applies inverse camera transforms to stabilize video footage.
The result is video where the camera appears static, allowing SAM3DBody
to estimate true world-space character motion without camera effects.

Pipeline:
    Original Video → CameraSolverV2 → Camera Matrices
                  ↓                        ↓
            VideoStabilizer ←──────────────┘
                  ↓
            Stabilized Video → SAM3DBody → Clean Pose

Technical Approach:
    For rotation-only shots: Apply inverse rotation via homography
    For translation shots: Apply inverse transform (future COLMAP integration)

Version: 5.0.0
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
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
    
    For rotation-only camera motion (pan/tilt), this applies a homography
    transform to each frame that cancels out the camera rotation, making
    the background appear static.
    
    The stabilized output allows SAM3DBody to estimate true world-space
    character motion without camera effects baked in.
    """
    
    CATEGORY = "SAM3DBody2abc/Camera"
    FUNCTION = "stabilize"
    
    BORDER_MODES = ["crop", "replicate", "reflect", "constant"]
    OUTPUT_SIZES = ["same", "expanded", "crop_valid"]
    REFERENCE_FRAMES = ["first", "middle", "last", "smoothed"]
    
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
                # === Reference Frame ===
                "reference_frame": (cls.REFERENCE_FRAMES, {
                    "default": "first",
                    "tooltip": "Which frame's camera pose to use as reference (stabilize TO this pose)"
                }),
                "custom_reference": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Custom reference frame index (if reference_frame not used)"
                }),
                
                # === Border Handling ===
                "border_mode": (cls.BORDER_MODES, {
                    "default": "replicate",
                    "tooltip": "How to fill areas outside original frame"
                }),
                
                # === Output Size ===
                "output_size": (cls.OUTPUT_SIZES, {
                    "default": "same",
                    "tooltip": "same: keep original size. expanded: include all warped content. crop_valid: only valid pixels"
                }),
                
                # === Smoothing ===
                "smoothing": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Blend between full stabilization (0) and original (1)"
                }),
                
                # === Debug ===
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
        output_size: str = "same",
        smoothing: float = 0.0,
        generate_comparison: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Stabilize video by applying inverse camera transforms.
        
        Args:
            images: Input frames [N, H, W, 3]
            camera_matrices: Dict with 'matrices' list of per-frame transforms
            intrinsics: Camera intrinsics with K matrix
            reference_frame: Which frame to stabilize to
            custom_reference: Custom reference frame index
            border_mode: How to handle borders
            output_size: Output frame size mode
            smoothing: Blend factor (0=full stabilization, 1=original)
            generate_comparison: Create side-by-side output
            
        Returns:
            stabilized: Stabilized video frames
            comparison: Side-by-side comparison (original | stabilized)
            stabilize_info: Dict with stabilization metadata
        """
        
        log_info(f"{'='*60}")
        log_info(f"Stabilizing {images.shape[0]} frames")
        log_info(f"Reference: {reference_frame}, Border: {border_mode}")
        log_info(f"{'='*60}")
        
        N, H, W, C = images.shape
        device = images.device
        
        # Extract camera matrices
        matrices = camera_matrices.get("matrices", [])
        if len(matrices) != N:
            log_warning(f"Matrix count ({len(matrices)}) != frame count ({N})")
            matrices = self._interpolate_matrices(matrices, N)
        
        # Get intrinsics matrix K
        K = self._get_intrinsics_matrix(intrinsics, W, H)
        K_inv = np.linalg.inv(K)
        
        log_info(f"K matrix:\n{K}")
        
        # Determine reference frame index
        ref_idx = self._get_reference_index(reference_frame, custom_reference, N)
        log_info(f"Reference frame: {ref_idx}")
        
        # Get reference rotation matrix
        R_ref = self._extract_rotation(matrices[ref_idx])
        
        # Process each frame
        stabilized_frames = []
        homographies = []
        
        for i in range(N):
            # Get current frame's rotation
            R_curr = self._extract_rotation(matrices[i])
            
            # Compute relative rotation (current → reference)
            # R_rel = R_ref @ R_curr.T  (rotate from current to reference)
            R_rel = R_ref @ R_curr.T
            
            # Convert rotation to homography: H = K @ R @ K^-1
            H = K @ R_rel @ K_inv
            
            # Normalize homography
            H = H / H[2, 2]
            
            homographies.append(H)
            
            # Apply homography to frame
            frame_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
            
            # Apply smoothing (blend between stabilized and original)
            if smoothing > 0:
                H_smooth = self._blend_homography(H, np.eye(3), smoothing)
            else:
                H_smooth = H
            
            # Warp frame
            warped = self._warp_frame(frame_np, H_smooth, W, H, border_mode)
            
            stabilized_frames.append(warped)
            
            if i % 10 == 0:
                log_info(f"Frame {i}/{N}: R_rel angle = {self._rotation_angle(R_rel):.2f}°")
        
        # Stack frames
        stabilized = np.stack(stabilized_frames, axis=0)
        stabilized = torch.from_numpy(stabilized).float() / 255.0
        stabilized = stabilized.to(device)
        
        # Generate comparison if requested
        if generate_comparison:
            comparison = self._generate_comparison(images, stabilized)
        else:
            comparison = stabilized
        
        # Build info dict
        stabilize_info = {
            "num_frames": N,
            "reference_frame": ref_idx,
            "border_mode": border_mode,
            "output_size": output_size,
            "smoothing": smoothing,
            "total_rotation_deg": self._total_rotation(matrices),
            "max_homography_scale": max(np.linalg.norm(H[:2, :2]) for H in homographies),
            "intrinsics_used": {
                "fx": K[0, 0],
                "fy": K[1, 1],
                "cx": K[0, 2],
                "cy": K[1, 2],
            },
        }
        
        log_info(f"✅ Stabilization complete")
        log_info(f"Total camera rotation: {stabilize_info['total_rotation_deg']:.1f}°")
        
        return (stabilized, comparison, stabilize_info)
    
    def _get_intrinsics_matrix(self, intrinsics: Dict, W: int, H: int) -> np.ndarray:
        """Extract or build K matrix from intrinsics."""
        
        if "k_matrix" in intrinsics:
            return np.array(intrinsics["k_matrix"])
        
        fx = intrinsics.get("focal_px", W)
        fy = intrinsics.get("focal_px", W)  # Assume square pixels
        cx = intrinsics.get("cx", W / 2)
        cy = intrinsics.get("cy", H / 2)
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=np.float64)
        
        return K
    
    def _get_reference_index(self, mode: str, custom: int, N: int) -> int:
        """Get reference frame index based on mode."""
        
        if mode == "first":
            return 0
        elif mode == "middle":
            return N // 2
        elif mode == "last":
            return N - 1
        else:  # custom or smoothed
            return min(custom, N - 1)
    
    def _extract_rotation(self, matrix_data: Dict) -> np.ndarray:
        """Extract 3x3 rotation matrix from camera matrix data."""
        
        if "rotation" in matrix_data:
            return np.array(matrix_data["rotation"])
        elif "matrix" in matrix_data:
            M = np.array(matrix_data["matrix"])
            return M[:3, :3]
        else:
            log_warning("No rotation found in matrix data, using identity")
            return np.eye(3)
    
    def _rotation_angle(self, R: np.ndarray) -> float:
        """Get rotation angle in degrees from rotation matrix."""
        
        # trace(R) = 1 + 2*cos(theta)
        trace = np.trace(R)
        cos_angle = (trace - 1) / 2
        cos_angle = np.clip(cos_angle, -1, 1)
        angle_rad = np.arccos(cos_angle)
        return np.degrees(angle_rad)
    
    def _total_rotation(self, matrices: List[Dict]) -> float:
        """Calculate total rotation from first to last frame."""
        
        if len(matrices) < 2:
            return 0.0
        
        R_first = self._extract_rotation(matrices[0])
        R_last = self._extract_rotation(matrices[-1])
        R_total = R_last @ R_first.T
        
        return self._rotation_angle(R_total)
    
    def _blend_homography(self, H1: np.ndarray, H2: np.ndarray, alpha: float) -> np.ndarray:
        """Blend two homographies. alpha=0 gives H1, alpha=1 gives H2."""
        
        # Simple linear blend (not geometrically correct but works for small differences)
        H_blend = (1 - alpha) * H1 + alpha * H2
        return H_blend / H_blend[2, 2]
    
    def _warp_frame(
        self,
        frame: np.ndarray,
        H: np.ndarray,
        W: int,
        H_img: int,
        border_mode: str,
    ) -> np.ndarray:
        """Apply homography to frame with specified border handling."""
        
        # Border mode mapping
        border_map = {
            "crop": cv2.BORDER_CONSTANT,
            "constant": cv2.BORDER_CONSTANT,
            "replicate": cv2.BORDER_REPLICATE,
            "reflect": cv2.BORDER_REFLECT,
        }
        
        border_value = border_map.get(border_mode, cv2.BORDER_REPLICATE)
        
        # Apply warp
        warped = cv2.warpPerspective(
            frame,
            H,
            (W, H_img),
            flags=cv2.INTER_LINEAR,
            borderMode=border_value,
            borderValue=(0, 0, 0),
        )
        
        return warped
    
    def _generate_comparison(
        self,
        original: torch.Tensor,
        stabilized: torch.Tensor,
    ) -> torch.Tensor:
        """Generate side-by-side comparison video."""
        
        N, H, W, C = original.shape
        
        # Create side-by-side frames
        comparison = torch.zeros((N, H, W * 2, C), dtype=original.dtype, device=original.device)
        comparison[:, :, :W, :] = original
        comparison[:, :, W:, :] = stabilized
        
        # Add separator line
        comparison[:, :, W-1:W+1, :] = 1.0  # White line
        
        return comparison
    
    def _interpolate_matrices(self, matrices: List[Dict], target_count: int) -> List[Dict]:
        """Interpolate matrices to match frame count."""
        
        if len(matrices) == 0:
            # Return identity matrices
            return [{"rotation": np.eye(3).tolist(), "matrix": np.eye(4).tolist()} 
                    for _ in range(target_count)]
        
        if len(matrices) >= target_count:
            return matrices[:target_count]
        
        # Simple nearest-neighbor interpolation
        result = []
        for i in range(target_count):
            src_idx = int(i * len(matrices) / target_count)
            result.append(matrices[src_idx])
        
        return result


class StabilizationInfo:
    """
    Display stabilization info and diagnostics.
    """
    
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
        """Format stabilization info for display."""
        
        lines = [
            "═" * 50,
            "VIDEO STABILIZATION INFO",
            "═" * 50,
            f"Frames: {stabilize_info.get('num_frames', 'N/A')}",
            f"Reference frame: {stabilize_info.get('reference_frame', 'N/A')}",
            f"Border mode: {stabilize_info.get('border_mode', 'N/A')}",
            f"Smoothing: {stabilize_info.get('smoothing', 0):.1%}",
            "",
            f"Total camera rotation: {stabilize_info.get('total_rotation_deg', 0):.1f}°",
            f"Max homography scale: {stabilize_info.get('max_homography_scale', 1):.3f}",
            "",
            "Intrinsics used:",
        ]
        
        intrinsics = stabilize_info.get("intrinsics_used", {})
        lines.append(f"  fx: {intrinsics.get('fx', 'N/A'):.1f}")
        lines.append(f"  fy: {intrinsics.get('fy', 'N/A'):.1f}")
        lines.append(f"  cx: {intrinsics.get('cx', 'N/A'):.1f}")
        lines.append(f"  cy: {intrinsics.get('cy', 'N/A'):.1f}")
        
        lines.append("═" * 50)
        
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
