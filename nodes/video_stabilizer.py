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
    
    BORDER_MODES = ["replicate", "reflect", "constant", "crop"]
    OUTPUT_SIZES = ["same", "expanded", "crop_valid"]
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
                # === Reference Frame ===
                "reference_frame": (cls.REFERENCE_FRAMES, {
                    "default": "first",
                    "tooltip": "Which frame's camera pose to use as reference (stabilize TO this pose)"
                }),
                "custom_reference": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Custom reference frame index (used when reference_frame='custom')"
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
        """
        
        log_info(f"{'='*60}")
        log_info(f"Stabilizing {images.shape[0]} frames")
        log_info(f"Reference: {reference_frame}, Border: {border_mode}")
        log_info(f"{'='*60}")
        
        N, H, W, C = images.shape
        device = images.device
        
        # Safety check - ensure valid dimensions
        if H <= 0 or W <= 0 or N <= 0:
            log_error(f"Invalid image dimensions: {images.shape}")
            return (images, images, {"error": "Invalid dimensions"})
        
        # Extract camera matrices
        matrices = camera_matrices.get("matrices", [])
        if len(matrices) == 0:
            log_error("No camera matrices provided")
            return (images, images, {"error": "No matrices"})
            
        if len(matrices) != N:
            log_warning(f"Matrix count ({len(matrices)}) != frame count ({N})")
            matrices = self._interpolate_matrices(matrices, N)
        
        # Get intrinsics matrix K
        try:
            K = self._get_intrinsics_matrix(intrinsics, W, H)
            
            # Ensure K is float64 and check for validity
            K = np.array(K, dtype=np.float64)
            
            # Check K is valid
            if np.any(np.isnan(K)) or np.any(np.isinf(K)):
                log_error("K matrix contains NaN or Inf")
                return (images, images, {"error": "Invalid K matrix"})
            
            # Check K is invertible
            det_K = np.linalg.det(K)
            if abs(det_K) < 1e-10:
                log_error(f"K matrix is singular (det={det_K})")
                return (images, images, {"error": "Singular K matrix"})
            
            K_inv = np.linalg.inv(K)
            log_info(f"K matrix:\n{K}")
            log_info(f"K determinant: {det_K:.6f}")
            
        except Exception as e:
            log_error(f"Failed to compute K matrix: {e}")
            import traceback
            traceback.print_exc()
            return (images, images, {"error": f"K matrix error: {e}"})
        
        # Determine reference frame index
        ref_idx = self._get_reference_index(reference_frame, custom_reference, N)
        log_info(f"Reference frame: {ref_idx}")
        
        # Get reference rotation matrix
        R_ref = self._extract_rotation(matrices[ref_idx])
        log_info(f"R_ref det: {np.linalg.det(R_ref):.6f}")
        
        # Process each frame
        stabilized_frames = []
        homographies = []
        
        # Convert all images to numpy first
        log_info("Converting images to numpy...")
        try:
            images_np = (images.cpu().numpy() * 255).astype(np.uint8)
            log_info(f"Images numpy shape: {images_np.shape}, dtype: {images_np.dtype}")
        except Exception as e:
            log_error(f"Failed to convert images: {e}")
            return (images, images, {"error": f"Image conversion error: {e}"})
        
        for i in range(N):
            try:
                # Get current frame's rotation
                R_curr = self._extract_rotation(matrices[i])
                
                # Compute relative rotation (current → reference)
                R_rel = R_ref @ R_curr.T
                
                # Verify R_rel is valid rotation matrix
                det_R = np.linalg.det(R_rel)
                if abs(det_R - 1.0) > 0.1:
                    log_warning(f"Frame {i}: R_rel not orthonormal (det={det_R:.6f})")
                    # Re-orthonormalize using SVD
                    U, _, Vt = np.linalg.svd(R_rel)
                    R_rel = U @ Vt
                
                # Convert rotation to homography: H = K @ R @ K^-1
                H = K @ R_rel @ K_inv
                
                # Normalize homography
                if abs(H[2, 2]) < 1e-10:
                    log_warning(f"Frame {i}: H[2,2] near zero, using identity")
                    H = np.eye(3, dtype=np.float64)
                else:
                    H = H / H[2, 2]
                
                # Check for degenerate homography
                det_H = np.linalg.det(H)
                if abs(det_H) < 1e-6 or abs(det_H) > 1e6:
                    log_warning(f"Frame {i}: Degenerate homography (det={det_H:.6f}), using identity")
                    H = np.eye(3, dtype=np.float64)
                
                # Check for NaN/Inf
                if np.any(np.isnan(H)) or np.any(np.isinf(H)):
                    log_warning(f"Frame {i}: H contains NaN/Inf, using identity")
                    H = np.eye(3, dtype=np.float64)
                
                homographies.append(H.copy())
                
                # Get frame - ensure contiguous
                frame_np = np.ascontiguousarray(images_np[i])
                
                # Apply smoothing
                if smoothing > 0:
                    H_final = self._blend_homography(H, np.eye(3, dtype=np.float64), smoothing)
                else:
                    H_final = H
                
                # Ensure H is float64 for OpenCV
                H_final = np.ascontiguousarray(H_final.astype(np.float64))
                
                # Warp frame with error handling
                warped = self._warp_frame_safe(frame_np, H_final, W, H, border_mode)
                
                if warped is None:
                    log_warning(f"Frame {i}: Warp failed, using original")
                    warped = frame_np.copy()
                
                stabilized_frames.append(warped)
                
                if i % 10 == 0:
                    angle = self._rotation_angle(R_rel)
                    log_info(f"Frame {i}/{N}: R_rel angle = {angle:.2f}°, H det = {det_H:.4f}")
                    
            except Exception as e:
                log_error(f"Frame {i}: Error - {e}")
                import traceback
                traceback.print_exc()
                # Use original frame on error
                stabilized_frames.append(images_np[i].copy())
        
        # Stack frames
        log_info("Stacking stabilized frames...")
        try:
            stabilized = np.stack(stabilized_frames, axis=0)
            stabilized = torch.from_numpy(stabilized).float() / 255.0
            stabilized = stabilized.to(device)
            log_info(f"Stabilized shape: {stabilized.shape}")
        except Exception as e:
            log_error(f"Failed to stack frames: {e}")
            return (images, images, {"error": f"Stack error: {e}"})
        
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
            "max_homography_scale": max(np.linalg.norm(H[:2, :2]) for H in homographies) if homographies else 1.0,
            "intrinsics_used": {
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
            },
        }
        
        log_info(f"✅ Stabilization complete")
        log_info(f"Total camera rotation: {stabilize_info['total_rotation_deg']:.1f}°")
        
        return (stabilized, comparison, stabilize_info)
    
    def _get_intrinsics_matrix(self, intrinsics: Dict, W: int, H: int) -> np.ndarray:
        """Extract or build K matrix from intrinsics."""
        
        if "k_matrix" in intrinsics:
            return np.array(intrinsics["k_matrix"], dtype=np.float64)
        
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
        elif mode == "custom":
            return min(max(0, custom), N - 1)
        else:
            return 0
    
    def _extract_rotation(self, matrix_data: Dict) -> np.ndarray:
        """Extract 3x3 rotation matrix from camera matrix data."""
        
        if "rotation" in matrix_data:
            R = np.array(matrix_data["rotation"], dtype=np.float64)
        elif "matrix" in matrix_data:
            M = np.array(matrix_data["matrix"], dtype=np.float64)
            R = M[:3, :3]
        else:
            log_warning("No rotation found in matrix data, using identity")
            R = np.eye(3, dtype=np.float64)
        
        # Ensure it's a valid rotation matrix (orthonormal)
        # Use SVD to find closest orthonormal matrix
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        
        # Ensure proper rotation (det = +1, not -1)
        if np.linalg.det(R) < 0:
            R = -R
        
        return R
    
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
        
        # Simple linear blend
        H_blend = (1 - alpha) * H1 + alpha * H2
        if abs(H_blend[2, 2]) > 1e-10:
            H_blend = H_blend / H_blend[2, 2]
        return H_blend
    
    def _warp_frame_safe(
        self,
        frame: np.ndarray,
        H: np.ndarray,
        W: int,
        H_img: int,
        border_mode: str,
    ) -> Optional[np.ndarray]:
        """Apply homography to frame with safety checks."""
        
        try:
            # Validate inputs
            if frame is None or frame.size == 0:
                log_warning("Empty frame")
                return None
            
            if H is None:
                log_warning("Null homography")
                return None
            
            # Ensure frame is correct format
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
            # Ensure H is correct format
            H = np.ascontiguousarray(H.astype(np.float64))
            
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
            
        except Exception as e:
            log_error(f"Warp error: {e}")
            return None
    
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
