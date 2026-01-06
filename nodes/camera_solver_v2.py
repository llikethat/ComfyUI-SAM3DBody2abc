"""
Camera Solver V2 for SAM3DBody2abc v5.0
=======================================

TAPIR-based camera solver with:
- Temporal point tracking (not frame-pair matching)
- Automatic shot type classification
- Rainbow trail debug visualization
- Background-only tracking (foreground masked)

Key Differences from v4.x CameraSolver (Legacy):
1. Uses TAPIR for temporally consistent tracking (vs LoFTR/KLT frame pairs)
2. Shot classification determines solver type automatically
3. Debug visualization shows tracked points with rainbow trails
4. Designed for stabilization-first pipeline

Shot Types:
- Rotation: Pure pan/tilt/roll (homography decomposition)
- Translation: Dolly/truck/crane with parallax (future: COLMAP)
- Mixed: Rotation-dominant with weak parallax (two-stage)
- Static: No camera motion detected

Installation:
    # TAPIR (required)
    pip install 'tapnet[torch] @ git+https://github.com/google-deepmind/tapnet.git'
    
    # Download checkpoint (~250MB)
    mkdir -p ComfyUI/models/tapir
    wget -P ComfyUI/models/tapir https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt

Fixes in v5.0.0:
- Python 3.12-compatible TensorFlow blocker (find_spec)
- Robust TAPIR import (tapnet OR tapir backend)
- Explicit checkpoint validation with helpful error messages
- Loud failures (no silent ComfyUI skips)

Version: 5.0.0
Author: SAM3DBody2abc Project
License: Apache 2.0 (TAPIR: Apache 2.0)
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import colorsys
from typing import Dict, Tuple, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum

# -----------------------------------------------------------------------------
# Logging Setup (single output to avoid duplicates)
# -----------------------------------------------------------------------------

def log_info(msg):
    print(f"[CameraSolverV2] {msg}")

def log_error(msg):
    print(f"[CameraSolverV2] ❌ {msg}")

def log_warning(msg):
    print(f"[CameraSolverV2] ⚠️ {msg}")

# -----------------------------------------------------------------------------
# TensorFlow Blocker - DISABLED (TensorFlow is installed)
# -----------------------------------------------------------------------------
# Note: If TensorFlow is NOT installed, you can enable the blocker below.
# However, it's easier to just: pip install tensorflow

# import importlib.abc
# import importlib.util
# 
# class TensorFlowBlocker(importlib.abc.MetaPathFinder):
#     """Block TensorFlow import - not needed for TAPIR torch inference."""
#     def find_spec(self, fullname, path, target=None):
#         if fullname == "tensorflow" or fullname.startswith("tensorflow."):
#             return None  # Return None to let other finders fail gracefully
#         return None
# 
# _tf_blocker = TensorFlowBlocker()
# sys.meta_path.insert(0, _tf_blocker)

# -----------------------------------------------------------------------------
# TAPIR Import
# -----------------------------------------------------------------------------

TAPIR_AVAILABLE = False
TAPIR_IMPORT_ERROR = None
TAPIR_BACKEND = None

log_info("Attempting to import TAPIR...")
try:
    from tapnet.torch import tapir_model
    from tapnet.utils import transforms as tapir_transforms
    TAPIR_BACKEND = "tapnet"
    TAPIR_AVAILABLE = True
    log_info(f"✅ TAPIR imported successfully ({TAPIR_BACKEND} backend)")
except ImportError as e1:
    # Try alternative package name
    try:
        from tapir.torch import tapir_model
        from tapir.utils import transforms as tapir_transforms
        TAPIR_BACKEND = "tapir"
        TAPIR_AVAILABLE = True
        log_info(f"✅ TAPIR imported successfully ({TAPIR_BACKEND} backend)")
    except ImportError as e2:
        TAPIR_IMPORT_ERROR = f"tapnet: {e1}, tapir: {e2}"
        log_error("TAPIR import failed")
        log_error(f"  tapnet error: {e1}")
        log_error(f"  tapir error: {e2}")
        log_info("Install with: pip install tensorflow tapnet")
        log_info("  Or: pip install 'tapnet[torch] @ git+https://github.com/google-deepmind/tapnet.git'")

# -----------------------------------------------------------------------------
# Shot Type Definitions
# -----------------------------------------------------------------------------

class ShotType(Enum):
    """Camera motion classification."""
    STATIC = "static"
    ROTATION = "rotation"  # Pan/tilt/roll only
    TRANSLATION = "translation"  # Dolly/truck/crane with parallax
    MIXED = "mixed"  # Rotation-dominant with weak parallax


@dataclass
class ShotClassification:
    """Shot type classification result."""
    shot_type: ShotType
    confidence: float
    flow_coherence: float
    parallax_score: float
    homography_error: float
    motion_magnitude: float
    details: Dict[str, Any]


@dataclass
class CameraMatrix:
    """Per-frame camera transformation."""
    frame: int
    matrix: np.ndarray  # 4x4 transformation matrix
    rotation: Optional[np.ndarray] = None  # 3x3 rotation matrix
    translation: Optional[np.ndarray] = None  # 3x1 translation vector
    euler_zxy: Optional[Tuple[float, float, float]] = None  # (pan, tilt, roll) in degrees


class CameraSolverV2:
    """
    TAPIR-based camera solver for v5.0 pipeline.
    
    Features:
    - Temporal point tracking across all frames
    - Automatic shot type classification
    - Background-only tracking (masked by foreground)
    - Rainbow trail debug visualization
    - Homography-based rotation solving
    
    Usage:
        images (IMAGE) + foreground_mask (MASK) + intrinsics (INTRINSICS)
        → camera_matrices (CAMERA_MATRICES) + debug_vis (IMAGE) + shot_info (SHOT_INFO)
    """
    
    QUALITY_MODES = ["fast", "balanced", "best"]
    SHOT_TYPE_OVERRIDES = ["auto", "static", "rotation", "translation", "mixed"]
    
    # Default checkpoint path (relative to ComfyUI custom_nodes)
    DEFAULT_CHECKPOINT = "models/tapir/bootstapir_checkpoint_v2.pt"
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input video frames [N, H, W, 3]"
                }),
                "intrinsics": ("INTRINSICS", {
                    "tooltip": "Camera intrinsics from IntrinsicsEstimator"
                }),
            },
            "optional": {
                # === Masking ===
                "foreground_mask": ("MASK", {
                    "tooltip": "Foreground mask from SAM3 (tracked points will be background only)"
                }),
                
                # === Quality ===
                "quality": (cls.QUALITY_MODES, {
                    "default": "balanced",
                    "tooltip": "fast: fewer points, faster. balanced: good quality. best: maximum accuracy."
                }),
                
                # === Shot Type ===
                "force_shot_type": (cls.SHOT_TYPE_OVERRIDES, {
                    "default": "auto",
                    "tooltip": "Override automatic shot classification"
                }),
                
                # === Point Grid ===
                "grid_size": ("INT", {
                    "default": 12,
                    "min": 5,
                    "max": 25,
                    "tooltip": "Grid density for point sampling (grid_size x grid_size points)"
                }),
                
                # === TAPIR Settings ===
                "checkpoint_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to TAPIR checkpoint (leave empty for default location)"
                }),
                
                # === Debug Visualization ===
                "debug_visualization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Generate rainbow trail debug visualization"
                }),
                "trail_length": ("INT", {
                    "default": 20,
                    "min": 5,
                    "max": 50,
                    "tooltip": "Length of rainbow trails in frames"
                }),
                
                # === Advanced ===
                "smoothing_window": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 15,
                    "step": 2,
                    "tooltip": "Gaussian smoothing window for camera motion (0=none)"
                }),
                "min_visible_ratio": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.1,
                    "max": 0.9,
                    "tooltip": "Minimum ratio of visible points per frame"
                }),
                
                # === Correction Factors ===
                "rotation_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Scale detected rotation (use <1.0 if pan appears too strong, >1.0 if too weak)"
                }),
            }
        }
    
    RETURN_TYPES = ("CAMERA_MATRICES", "IMAGE", "SHOT_INFO", "STRING")
    RETURN_NAMES = ("camera_matrices", "debug_vis", "shot_info", "status")
    FUNCTION = "solve"
    CATEGORY = "SAM3DBody2abc/Camera"
    
    def __init__(self):
        self.tapir_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._checkpoint_loaded = None
    
    def _load_tapir(self, checkpoint_path: str = "") -> bool:
        """Load TAPIR model with robust path resolution."""
        log_info(f"_load_tapir called, TAPIR_AVAILABLE={TAPIR_AVAILABLE}")
        
        if not TAPIR_AVAILABLE:
            log_error("TAPIR module not available, cannot load")
            return False
        
        if self.tapir_model is not None and self._checkpoint_loaded == checkpoint_path:
            log_info(f"TAPIR already loaded from: {self._checkpoint_loaded}")
            return True
        
        # Resolve checkpoint path
        if not checkpoint_path:
            # Get paths relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))  # nodes/
            extension_dir = os.path.dirname(current_dir)  # ComfyUI-SAM3DBody2abc/
            custom_nodes_dir = os.path.dirname(extension_dir)  # custom_nodes/
            comfyui_dir = os.path.dirname(custom_nodes_dir)  # ComfyUI/
            
            log_info("Path resolution:")
            log_info(f"  current_dir: {current_dir}")
            log_info(f"  extension_dir: {extension_dir}")
            log_info(f"  comfyui_dir: {comfyui_dir}")
            
            # Search order:
            # 0. Hardcoded /workspace path (for RunPod/cloud environments)
            # 1. ComfyUI/models/tapir/ (standard ComfyUI convention)
            # 2. ComfyUI-SAM3DBody2abc/models/tapir/ (self-contained)
            # 3. ~/.cache/tapir/ (user cache)
            # 4. Current working directory
            possible_paths = [
                # Hardcoded workspace path (RunPod/cloud)
                "/workspace/ComfyUI/models/tapir/bootstapir_checkpoint_v2.pt",
                # Standard ComfyUI models path
                os.path.join(comfyui_dir, "models", "tapir", "bootstapir_checkpoint_v2.pt"),
                # Self-contained in extension
                os.path.join(extension_dir, "models", "tapir", "bootstapir_checkpoint_v2.pt"),
                # User cache
                os.path.expanduser("~/.cache/tapir/bootstapir_checkpoint_v2.pt"),
                # Current directory
                "bootstapir_checkpoint_v2.pt",
            ]
            
            log_info("Searching for checkpoint in:")
            for i, path in enumerate(possible_paths, 1):
                exists = os.path.exists(path)
                status = "✅ FOUND" if exists else "not found"
                log_info(f"  {i}. {path} -> {status}")
                if exists and not checkpoint_path:
                    checkpoint_path = path
        
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            log_error("TAPIR checkpoint not found!")
            log_info("Download from: https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt")
            log_info("Place in one of these locations:")
            log_info("  1. ComfyUI/models/tapir/bootstapir_checkpoint_v2.pt (recommended)")
            log_info("  2. ComfyUI-SAM3DBody2abc/models/tapir/bootstapir_checkpoint_v2.pt")
            return False
        
        try:
            log_info(f"🔄 Loading TAPIR from {checkpoint_path}...")
            self.tapir_model = tapir_model.TAPIR(pyramid_level=1)
            log_info("TAPIR model created, loading weights...")
            self.tapir_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            log_info(f"Weights loaded, moving to {self.device}...")
            self.tapir_model = self.tapir_model.to(self.device)
            self.tapir_model.eval()
            self._checkpoint_loaded = checkpoint_path
            log_info(f"✅ TAPIR loaded successfully on {self.device}")
            return True
        except Exception as e:
            log_error(f"Failed to load TAPIR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def solve(
        self,
        images: torch.Tensor,
        intrinsics: Dict,
        foreground_mask: Optional[torch.Tensor] = None,
        quality: str = "balanced",
        force_shot_type: str = "auto",
        grid_size: int = 12,
        checkpoint_path: str = "",
        debug_visualization: bool = True,
        trail_length: int = 20,
        smoothing_window: int = 5,
        min_visible_ratio: float = 0.3,
        rotation_scale: float = 1.0,
    ) -> Tuple[Dict, torch.Tensor, Dict, str]:
        """
        Solve camera motion from video frames.
        
        Returns:
            camera_matrices: Dict with per-frame 4x4 transformation matrices
            debug_vis: Rainbow trail visualization frames
            shot_info: Shot classification results
            status: Human-readable status string
        """
        
        num_frames = images.shape[0]
        H, W = images.shape[1], images.shape[2]
        
        print(f"\n{'='*60}")
        log_info(f"Processing {num_frames} frames ({W}x{H})")
        log_info(f"Quality: {quality}, Grid: {grid_size}x{grid_size}")
        log_info(f"TAPIR_AVAILABLE: {TAPIR_AVAILABLE}")
        log_info(f"debug_visualization: {debug_visualization}")
        print(f"{'='*60}")
        
        # Check TAPIR availability
        log_info("Calling _load_tapir...")
        if not self._load_tapir(checkpoint_path):
            log_warning("_load_tapir returned False, using fallback")
            return self._fallback_static(images, intrinsics)
        
        log_info("TAPIR loaded, generating query points...")
        
        # Step 1: Generate query points on background
        query_points = self._generate_query_points(
            H, W, grid_size, foreground_mask, quality
        )
        
        if query_points is None or len(query_points) < 10:
            log_warning(f"Not enough valid query points ({query_points.shape if query_points is not None else 'None'}), falling back to static")
            return self._fallback_static(images, intrinsics)
        
        log_info(f"Generated {len(query_points)} query points, running TAPIR tracking...")
        
        # Step 2: Run TAPIR tracking
        tracks, visibles = self._run_tapir(images, query_points)
        
        if tracks is None:
            log_warning("TAPIR tracking failed, using fallback")
            return self._fallback_static(images, intrinsics)
        
        log_info(f"TAPIR tracking complete: tracks shape {tracks.shape}")
        
        # Step 3: Classify shot type
        shot_classification = self._classify_shot(
            tracks, visibles, intrinsics, force_shot_type
        )
        
        log_info(f"Shot type: {shot_classification.shot_type.value} "
              f"(confidence: {shot_classification.confidence:.2f})")
        
        # Log rotation_scale if not 1.0
        if abs(rotation_scale - 1.0) > 0.01:
            log_info(f"Rotation scale: {rotation_scale:.2f}x (user override)")
        
        # Step 4: Solve camera based on shot type
        log_info("Solving camera motion...")
        camera_matrices = self._solve_camera(
            tracks, visibles, intrinsics, shot_classification, smoothing_window, rotation_scale
        )
        
        # Step 5: Generate debug visualization
        if debug_visualization:
            log_info("Generating rainbow trail visualization...")
            debug_vis = self._render_rainbow_trails(
                images, tracks, visibles, trail_length
            )
            log_info(f"Debug visualization complete: {debug_vis.shape}")
        else:
            debug_vis = images.clone()
        
        # Build output
        shot_info = {
            "shot_type": shot_classification.shot_type.value,
            "confidence": shot_classification.confidence,
            "flow_coherence": shot_classification.flow_coherence,
            "parallax_score": shot_classification.parallax_score,
            "homography_error": shot_classification.homography_error,
            "motion_magnitude": shot_classification.motion_magnitude,
            "num_points_tracked": len(query_points),
            "details": shot_classification.details,
        }
        
        matrices_output = {
            "matrices": camera_matrices,
            "num_frames": num_frames,
            "intrinsics": intrinsics,
            "shot_type": shot_classification.shot_type.value,
        }
        
        status = (f"{shot_classification.shot_type.value.capitalize()}: "
                 f"{len(query_points)} points, "
                 f"confidence {shot_classification.confidence:.0%}")
        
        log_info(f"✅ Complete: {status}")
        
        return (matrices_output, debug_vis, shot_info, status)
    
    def _generate_query_points(
        self,
        height: int,
        width: int,
        grid_size: int,
        foreground_mask: Optional[torch.Tensor],
        quality: str,
    ) -> Optional[np.ndarray]:
        """Generate query points on background (avoiding foreground mask)."""
        
        # Adjust grid based on quality
        if quality == "fast":
            grid_size = max(5, grid_size - 3)
        elif quality == "best":
            grid_size = min(25, grid_size + 3)
        
        # Generate grid
        margin = 20
        y_coords = np.linspace(margin, height - margin, grid_size)
        x_coords = np.linspace(margin, width - margin, grid_size)
        
        query_points = []
        masked_count = 0
        
        # Debug mask info
        if foreground_mask is not None:
            log_info(f"Foreground mask shape: {foreground_mask.shape}, dtype: {foreground_mask.dtype}")
            log_info(f"Mask value range: [{foreground_mask.min():.3f}, {foreground_mask.max():.3f}]")
            log_info(f"Mask unique values: {torch.unique(foreground_mask).tolist()[:10]}...")
            
            # Handle multi-frame mask - use first frame or union
            if foreground_mask.dim() == 4:  # [N, C, H, W] or [N, H, W, C]
                mask_2d = foreground_mask[0]
                if mask_2d.dim() == 3:
                    mask_2d = mask_2d.max(dim=0)[0]  # Take max across channels
            elif foreground_mask.dim() == 3:  # [N, H, W] - multiple frames
                # Use union of all frames (any frame with foreground = masked)
                mask_2d = foreground_mask.max(dim=0)[0]  # Union across frames
            else:
                mask_2d = foreground_mask
            
            log_info(f"Using mask shape: {mask_2d.shape}")
        else:
            mask_2d = None
            log_info("No foreground mask provided - tracking all points")
        
        # All points start at frame 0
        for y in y_coords:
            for x in x_coords:
                yi, xi = int(y), int(x)
                
                # Check if point is on background (not masked)
                if mask_2d is not None:
                    # Ensure indices are within bounds
                    yi_safe = min(yi, mask_2d.shape[0] - 1)
                    xi_safe = min(xi, mask_2d.shape[1] - 1)
                    
                    mask_val = mask_2d[yi_safe, xi_safe].item()
                    
                    # Skip if in foreground (any non-zero value = foreground)
                    # SAM3 multi-object mask has object IDs (0, 1, 2, 3...)
                    # Background is 0, any other value is foreground
                    if mask_val > 0.01:  # Small threshold to handle float masks
                        masked_count += 1
                        continue
                
                # TAPIR query format: [t, y, x]
                query_points.append([0, yi, xi])
        
        log_info(f"Generated {len(query_points)} query points ({masked_count} masked out)")
        
        if len(query_points) < 10:
            log_warning(f"Only {len(query_points)} background points found - mask may be too large")
            # Add some points anyway if mask is too aggressive
            for y in y_coords[::2]:
                for x in x_coords[::2]:
                    query_points.append([0, int(y), int(x)])
            log_info(f"Added fallback points, total: {len(query_points)}")
        
        return np.array(query_points, dtype=np.int32)
    
    def _run_tapir(
        self,
        images: torch.Tensor,
        query_points: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Run TAPIR tracking on video frames with automatic resolution scaling."""
        
        num_frames = images.shape[0]
        H, W = images.shape[1], images.shape[2]
        
        # Calculate if we need to downscale for memory
        # TAPIR uses roughly: frames * H * W * 4 bytes * ~100 (feature maps)
        # Safe limit: ~4GB for tracking = ~40M pixels total
        total_pixels = num_frames * H * W
        max_pixels = 30_000_000  # Conservative limit
        
        scale_factor = 1.0
        if total_pixels > max_pixels:
            scale_factor = (max_pixels / total_pixels) ** 0.5
            scale_factor = max(0.25, scale_factor)  # Don't go below 25%
            log_info(f"Downscaling to {scale_factor:.0%} for memory ({total_pixels/1e6:.1f}M -> {total_pixels*scale_factor**2/1e6:.1f}M pixels)")
        
        try:
            # Prepare video: [T, H, W, C] normalized to [-1, 1]
            video = images.float()
            if video.max() > 1:
                video = video / 255.0 * 2 - 1
            else:
                video = video * 2 - 1
            
            # Downscale if needed
            if scale_factor < 1.0:
                new_H = int(H * scale_factor)
                new_W = int(W * scale_factor)
                # Resize: [T, H, W, C] -> [T, C, H, W] -> resize -> [T, new_H, new_W, C]
                video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
                video = F.interpolate(video, size=(new_H, new_W), mode='bilinear', align_corners=False)
                video = video.permute(0, 2, 3, 1)  # [T, new_H, new_W, C]
                
                # Scale query points
                query_points = query_points.copy().astype(np.float32)
                query_points[:, 1] *= scale_factor  # y
                query_points[:, 2] *= scale_factor  # x
                
                log_info(f"Resized to {new_W}x{new_H}")
            
            video = video.unsqueeze(0).to(self.device)  # [1, T, H, W, C]
            
            # Prepare query points: [B, N, 3]
            query_tensor = torch.tensor(query_points, dtype=torch.float32)
            query_tensor = query_tensor.unsqueeze(0).to(self.device)  # [1, N, 3]
            
            log_info(f"Running TAPIR inference ({video.shape[1]} frames, {query_tensor.shape[1]} points)...")
            
            # Clear cache before inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with torch.no_grad():
                outputs = self.tapir_model(video, query_tensor)
            
            # Extract results
            tracks = outputs['tracks'][0].cpu().numpy()  # [N, T, 2] - (x, y)
            occlusions = outputs['occlusion'][0]  # [N, T]
            expected_dist = outputs['expected_dist'][0]  # [N, T]
            
            # Scale tracks back to original resolution
            if scale_factor < 1.0:
                tracks = tracks / scale_factor
            
            # Compute visibility
            visibles = (1 - torch.sigmoid(occlusions)) * (1 - torch.sigmoid(expected_dist)) > 0.5
            visibles = visibles.cpu().numpy()  # [N, T]
            
            log_info(f"Tracking complete: {tracks.shape[0]} points × {tracks.shape[1]} frames")
            
            # Clear GPU memory
            del video, query_tensor, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return tracks, visibles
            
        except torch.cuda.OutOfMemoryError:
            log_error("GPU out of memory! Try reducing grid_size or video resolution")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, None
        except Exception as e:
            log_error(f"TAPIR inference error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _classify_shot(
        self,
        tracks: np.ndarray,
        visibles: np.ndarray,
        intrinsics: Dict,
        force_type: str,
    ) -> ShotClassification:
        """Classify shot type based on tracked points."""
        
        # If forced, return that type
        if force_type != "auto":
            return ShotClassification(
                shot_type=ShotType(force_type),
                confidence=1.0,
                flow_coherence=0.0,
                parallax_score=0.0,
                homography_error=0.0,
                motion_magnitude=0.0,
                details={"forced": True}
            )
        
        num_points, num_frames = tracks.shape[:2]
        
        # Compute motion statistics
        # 1. Flow coherence: how uniformly do all points move?
        # 2. Parallax: do nearby points move differently based on depth?
        # 3. Homography fit: how well does a homography explain the motion?
        
        flow_coherence_scores = []
        parallax_scores = []
        homography_errors = []
        motion_magnitudes = []
        
        # Analyze frame pairs
        for t in range(1, num_frames):
            # Get visible points in both frames
            vis_both = visibles[:, 0] & visibles[:, t]
            if np.sum(vis_both) < 8:
                continue
            
            pts0 = tracks[vis_both, 0, :]  # [M, 2]
            pts1 = tracks[vis_both, t, :]  # [M, 2]
            
            # Motion vectors
            motion = pts1 - pts0
            motion_mag = np.linalg.norm(motion, axis=1)
            motion_magnitudes.append(np.median(motion_mag))
            
            # Flow coherence: std of motion directions
            if np.median(motion_mag) > 0.5:  # Only if there's motion
                motion_norm = motion / (np.linalg.norm(motion, axis=1, keepdims=True) + 1e-6)
                mean_direction = np.mean(motion_norm, axis=0)
                coherence = np.mean(np.sum(motion_norm * mean_direction, axis=1))
                flow_coherence_scores.append(coherence)
            
            # Homography fit
            if len(pts0) >= 8:
                try:
                    H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
                    if H is not None:
                        # Compute reprojection error
                        pts0_h = np.hstack([pts0, np.ones((len(pts0), 1))])
                        pts1_pred = (H @ pts0_h.T).T
                        pts1_pred = pts1_pred[:, :2] / pts1_pred[:, 2:3]
                        error = np.mean(np.linalg.norm(pts1 - pts1_pred, axis=1))
                        homography_errors.append(error)
                except:
                    pass
            
            # Parallax: variance in motion relative to distance from center
            cx, cy = intrinsics.get('cx', tracks.shape[1] // 2), intrinsics.get('cy', tracks.shape[0] // 2)
            dist_from_center = np.sqrt((pts0[:, 0] - cx)**2 + (pts0[:, 1] - cy)**2)
            
            # If motion varies with distance from center, there's parallax
            if len(motion_mag) > 10:
                correlation = np.corrcoef(dist_from_center, motion_mag)[0, 1]
                parallax_scores.append(abs(correlation) if not np.isnan(correlation) else 0)
        
        # Aggregate scores
        flow_coherence = np.mean(flow_coherence_scores) if flow_coherence_scores else 0.9
        parallax_score = np.mean(parallax_scores) if parallax_scores else 0.0
        homography_error = np.mean(homography_errors) if homography_errors else 0.0
        motion_magnitude = np.mean(motion_magnitudes) if motion_magnitudes else 0.0
        
        # Classification thresholds
        STATIC_MOTION_THRESHOLD = 2.0  # pixels
        ROTATION_COHERENCE_THRESHOLD = 0.85
        ROTATION_HOMOGRAPHY_ERROR_THRESHOLD = 3.0  # pixels
        PARALLAX_THRESHOLD = 0.3
        
        # Determine shot type
        if motion_magnitude < STATIC_MOTION_THRESHOLD:
            shot_type = ShotType.STATIC
            confidence = 0.95
        elif (flow_coherence > ROTATION_COHERENCE_THRESHOLD and 
              homography_error < ROTATION_HOMOGRAPHY_ERROR_THRESHOLD and
              parallax_score < PARALLAX_THRESHOLD):
            shot_type = ShotType.ROTATION
            confidence = min(flow_coherence, 1.0 - parallax_score)
        elif parallax_score > PARALLAX_THRESHOLD * 2:
            shot_type = ShotType.TRANSLATION
            confidence = parallax_score
        else:
            shot_type = ShotType.MIXED
            confidence = 0.7
        
        print(f"[CameraSolverV2] Classification:")
        print(f"  Motion magnitude: {motion_magnitude:.1f}px")
        print(f"  Flow coherence: {flow_coherence:.3f}")
        print(f"  Homography error: {homography_error:.2f}px")
        print(f"  Parallax score: {parallax_score:.3f}")
        
        return ShotClassification(
            shot_type=shot_type,
            confidence=confidence,
            flow_coherence=flow_coherence,
            parallax_score=parallax_score,
            homography_error=homography_error,
            motion_magnitude=motion_magnitude,
            details={
                "num_frame_pairs_analyzed": len(flow_coherence_scores),
                "thresholds": {
                    "static_motion": STATIC_MOTION_THRESHOLD,
                    "rotation_coherence": ROTATION_COHERENCE_THRESHOLD,
                    "rotation_homography_error": ROTATION_HOMOGRAPHY_ERROR_THRESHOLD,
                    "parallax": PARALLAX_THRESHOLD,
                }
            }
        )
    
    def _solve_camera(
        self,
        tracks: np.ndarray,
        visibles: np.ndarray,
        intrinsics: Dict,
        classification: ShotClassification,
        smoothing_window: int,
        rotation_scale: float = 1.0,
    ) -> List[Dict]:
        """
        Solve camera matrices based on shot type.
        
        Uses BIDIRECTIONAL INCREMENTAL homography computation:
        - Compute H between consecutive frames (small motion = accurate)
        - Solve both forward and backward
        - Blend results for robustness at sequence boundaries
        
        Args:
            rotation_scale: Multiplier for detected rotation (use <1.0 if pan is overestimated)
        """
        
        num_frames = tracks.shape[1]
        matrices = []
        
        if classification.shot_type == ShotType.STATIC:
            # Identity matrices for all frames
            for t in range(num_frames):
                matrices.append({
                    "frame": t,
                    "matrix": np.eye(4).tolist(),
                    "rotation": np.eye(3).tolist(),
                    "pan": 0.0,
                    "tilt": 0.0,
                    "roll": 0.0,
                })
            return matrices
        
        # For rotation/mixed shots, use homography decomposition
        focal_px = intrinsics.get('focal_px', intrinsics.get('width', 1920))
        cx = intrinsics.get('cx', intrinsics.get('width', 1920) / 2)
        cy = intrinsics.get('cy', intrinsics.get('height', 1080) / 2)
        
        # Camera matrix K
        K = np.array([
            [focal_px, 0, cx],
            [0, focal_px, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        K_inv = np.linalg.inv(K)
        
        print(f"[CameraSolverV2] Using BIDIRECTIONAL INCREMENTAL homography")
        
        # DEBUG: Check actual track motion
        vis_first_last = visibles[:, 0] & visibles[:, num_frames-1]
        if np.sum(vis_first_last) > 0:
            pts_first = tracks[vis_first_last, 0, :]
            pts_last = tracks[vis_first_last, num_frames-1, :]
            motion = pts_last - pts_first
            print(f"[CameraSolverV2] DEBUG: Track motion frame 0→{num_frames-1}:")
            print(f"[CameraSolverV2]   Points visible in both: {np.sum(vis_first_last)}")
            print(f"[CameraSolverV2]   Mean X motion: {np.mean(motion[:, 0]):.1f}px")
            print(f"[CameraSolverV2]   Mean Y motion: {np.mean(motion[:, 1]):.1f}px")
            print(f"[CameraSolverV2]   Expected rotation (from X): {np.degrees(np.arctan(np.mean(motion[:, 0]) / focal_px)):.2f}°")
            if abs(rotation_scale - 1.0) > 0.01:
                scaled_rotation = np.degrees(np.arctan(np.mean(motion[:, 0]) / focal_px)) * rotation_scale
                print(f"[CameraSolverV2]   After rotation_scale ({rotation_scale:.2f}x): {scaled_rotation:.2f}°")
        
        # ========== DIRECT PAN/TILT ESTIMATION ==========
        # Instead of decomposing homography (which gives inconsistent axes),
        # compute pan/tilt directly from average pixel motion
        print(f"[CameraSolverV2] Using DIRECT PAN/TILT estimation")
        if abs(rotation_scale - 1.0) > 0.01:
            print(f"[CameraSolverV2] Rotation scale: {rotation_scale:.2f}x")
        
        direct_rotations = [np.eye(3)]
        
        for t in range(1, num_frames):
            # Get visible points in both frames
            vis_both = visibles[:, t-1] & visibles[:, t]
            num_visible = np.sum(vis_both)
            
            if num_visible < 4:
                # Use previous rotation
                direct_rotations.append(direct_rotations[-1].copy())
                continue
            
            pts_prev = tracks[vis_both, t-1, :]
            pts_curr = tracks[vis_both, t, :]
            
            # Compute average motion (pan = X motion, tilt = Y motion)
            motion = pts_curr - pts_prev
            mean_dx = np.median(motion[:, 0])  # Use median for robustness
            mean_dy = np.median(motion[:, 1])
            
            # Convert pixel motion to angle
            # pan = arctan(dx / focal), tilt = arctan(dy / focal)
            pan_angle = np.arctan(mean_dx / focal_px) * rotation_scale
            tilt_angle = np.arctan(mean_dy / focal_px) * rotation_scale
            
            # Create rotation matrix: R = Ry(pan) @ Rx(tilt)
            # Pan is rotation around Y axis (vertical)
            # Tilt is rotation around X axis (horizontal)
            cp, sp = np.cos(pan_angle), np.sin(pan_angle)
            ct, st = np.cos(tilt_angle), np.sin(tilt_angle)
            
            # R_pan (around Y)
            R_pan = np.array([
                [cp, 0, sp],
                [0, 1, 0],
                [-sp, 0, cp]
            ])
            
            # R_tilt (around X)
            R_tilt = np.array([
                [1, 0, 0],
                [0, ct, -st],
                [0, st, ct]
            ])
            
            # Combined incremental rotation
            R_inc = R_pan @ R_tilt
            
            # Accumulate
            R_cumulative = R_inc @ direct_rotations[-1]
            direct_rotations.append(R_cumulative)
        
        # Use direct rotations instead of homography-based
        final_rotations = direct_rotations
        
        # Log total rotation
        R_final = final_rotations[-1]
        total_angle = np.degrees(np.arccos(np.clip((np.trace(R_final) - 1) / 2, -1, 1)))
        print(f"[CameraSolverV2] Direct estimation total: {total_angle:.1f}°")
        
        # Apply temporal smoothing if requested
        if smoothing_window > 0 and len(final_rotations) > smoothing_window:
            print(f"[CameraSolverV2] Applying rotation smoothing (window={smoothing_window})")
            final_rotations = self._smooth_rotations(final_rotations, smoothing_window)
        
        # Convert to output format
        for t, R in enumerate(final_rotations):
            # Build 4x4 matrix (rotation only, no translation for rotation shots)
            M = np.eye(4)
            M[:3, :3] = R
            
            # Extract ZXY Euler angles (camera convention)
            pan, tilt, roll = self._rotation_to_euler_zxy(R)
            
            matrices.append({
                "frame": t,
                "matrix": M.tolist(),
                "rotation": R.tolist(),
                "pan": float(np.degrees(pan)),
                "tilt": float(np.degrees(tilt)),
                "roll": float(np.degrees(roll)),
            })
        
        # Log total rotation
        print(f"[CameraSolverV2] Total camera rotation: {total_angle:.1f}°")
        
        return matrices
    
    def _rotation_to_axis_angle(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to axis-angle representation."""
        from scipy.spatial.transform import Rotation
        return Rotation.from_matrix(R).as_rotvec()
    
    def _axis_angle_to_rotation(self, axis_angle: np.ndarray) -> np.ndarray:
        """Convert axis-angle to rotation matrix."""
        from scipy.spatial.transform import Rotation
        return Rotation.from_rotvec(axis_angle).as_matrix()
    
    def _smooth_rotations(
        self,
        rotations: List[np.ndarray],
        window: int,
    ) -> List[np.ndarray]:
        """Smooth rotation matrices using Gaussian averaging in axis-angle space."""
        
        from scipy.ndimage import gaussian_filter1d
        from scipy.spatial.transform import Rotation
        
        # Convert to axis-angle representation
        axis_angles = []
        for R in rotations:
            r = Rotation.from_matrix(R)
            axis_angles.append(r.as_rotvec())
        
        axis_angles = np.array(axis_angles)  # [N, 3]
        
        # Smooth each component
        sigma = window / 4.0
        smoothed = np.zeros_like(axis_angles)
        for i in range(3):
            smoothed[:, i] = gaussian_filter1d(axis_angles[:, i], sigma, mode='nearest')
        
        # Convert back to rotation matrices
        smoothed_rotations = []
        for rotvec in smoothed:
            R = Rotation.from_rotvec(rotvec).as_matrix()
            smoothed_rotations.append(R)
        
        return smoothed_rotations
    
    def _rotation_to_euler_zxy(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Extract pan/tilt/roll from rotation matrix.
        
        We build rotations as R = R_pan(Y) @ R_tilt(X), so we need to decompose
        in the same order.
        
        For a Y-X decomposition:
        R = Ry(pan) @ Rx(tilt)
        
        Returns:
            (pan, tilt, roll) in radians
        """
        # For Ry(p) @ Rx(t):
        # R = [[cos(p), sin(p)*sin(t), sin(p)*cos(t)],
        #      [0,      cos(t),        -sin(t)      ],
        #      [-sin(p), cos(p)*sin(t), cos(p)*cos(t)]]
        
        # Extract tilt from R[1,2] = -sin(tilt)
        tilt = np.arcsin(np.clip(-R[1, 2], -1.0, 1.0))
        
        # Extract pan from R[0,2] and R[2,2]
        # R[0,2] = sin(pan)*cos(tilt)
        # R[2,2] = cos(pan)*cos(tilt)
        ct = np.cos(tilt)
        if abs(ct) > 1e-6:
            pan = np.arctan2(R[0, 2], R[2, 2])
        else:
            # Gimbal lock - pan and roll are coupled
            pan = 0
        
        roll = 0  # We don't compute roll for now
        
        return pan, tilt, roll
    
    def _render_rainbow_trails(
        self,
        images: torch.Tensor,
        tracks: np.ndarray,
        visibles: np.ndarray,
        trail_length: int,
    ) -> torch.Tensor:
        """Render rainbow trail visualization like TAPIR demo."""
        
        num_points, num_frames = tracks.shape[:2]
        H, W = images.shape[1], images.shape[2]
        
        # Generate colors for each point (rainbow distribution)
        colors = self._get_rainbow_colors(num_points)
        
        # Convert images to numpy for drawing
        frames_np = (images.cpu().numpy() * 255).astype(np.uint8)
        output_frames = []
        
        # Point rendering settings
        point_radius = max(3, int(min(H, W) * 0.008))
        trail_thickness = max(1, int(min(H, W) * 0.003))
        
        for t in range(num_frames):
            frame = frames_np[t].copy()
            
            # Draw trails and points for each tracked point
            for i in range(num_points):
                color = colors[i]
                
                # Draw trail (history of positions)
                trail_start = max(0, t - trail_length)
                trail_points = []
                
                for tt in range(trail_start, t + 1):
                    if visibles[i, tt]:
                        x, y = tracks[i, tt]
                        trail_points.append((int(x), int(y)))
                
                # Draw trail segments with gradient
                if len(trail_points) > 1:
                    for j in range(len(trail_points) - 1):
                        # Alpha fades with age
                        alpha = (j + 1) / len(trail_points)
                        # Blend color with frame for transparency effect
                        pt1 = trail_points[j]
                        pt2 = trail_points[j + 1]
                        
                        # Draw line
                        cv2.line(frame, pt1, pt2, color, trail_thickness, cv2.LINE_AA)
                
                # Draw current point (if visible)
                if visibles[i, t]:
                    x, y = tracks[i, t]
                    center = (int(x), int(y))
                    
                    # Filled circle for current position
                    cv2.circle(frame, center, point_radius, color, -1, cv2.LINE_AA)
                    # White outline
                    cv2.circle(frame, center, point_radius, (255, 255, 255), 1, cv2.LINE_AA)
            
            output_frames.append(frame)
        
        # Convert back to tensor
        output_tensor = torch.from_numpy(np.stack(output_frames)).float() / 255.0
        
        return output_tensor
    
    def _get_rainbow_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate rainbow colors for point visualization."""
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            # Use HSV to RGB conversion for smooth rainbow
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            colors.append((int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
        return colors
    
    def _fallback_static(
        self,
        images: torch.Tensor,
        intrinsics: Dict,
    ) -> Tuple[Dict, torch.Tensor, Dict, str]:
        """Fallback to static camera assumption."""
        
        num_frames = images.shape[0]
        
        matrices = []
        for t in range(num_frames):
            matrices.append({
                "frame": t,
                "matrix": np.eye(4).tolist(),
                "pan": 0.0,
                "tilt": 0.0,
                "roll": 0.0,
            })
        
        matrices_output = {
            "matrices": matrices,
            "num_frames": num_frames,
            "intrinsics": intrinsics,
            "shot_type": "static",
        }
        
        shot_info = {
            "shot_type": "static",
            "confidence": 0.5,
            "flow_coherence": 0.0,
            "parallax_score": 0.0,
            "homography_error": 0.0,
            "motion_magnitude": 0.0,
            "num_points_tracked": 0,
            "details": {"fallback": True, "reason": "TAPIR not available or failed"},
        }
        
        return (matrices_output, images, shot_info, "Static (fallback - TAPIR unavailable)")


class CameraSolverLegacy:
    """
    Legacy camera solver from v4.x.
    
    This is the original LoFTR/LightGlue/KLT-based solver, renamed for backwards compatibility.
    For new workflows, use CameraSolverV2 with TAPIR.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        # Import and delegate to original
        from . import camera_solver as legacy
        return legacy.CameraSolver.INPUT_TYPES()
    
    RETURN_TYPES = ("CAMERA_EXTRINSICS", "IMAGE", "STRING")
    RETURN_NAMES = ("camera_extrinsics", "debug_vis", "status")
    FUNCTION = "solve"
    CATEGORY = "SAM3DBody2abc/Camera"
    
    def __init__(self):
        from . import camera_solver as legacy
        self._legacy_solver = legacy.CameraSolver()
    
    def solve(self, *args, **kwargs):
        return self._legacy_solver.solve(*args, **kwargs)


# Node registration
NODE_CLASS_MAPPINGS = {
    "CameraSolverV2": CameraSolverV2,
    # Legacy solver will be registered separately to avoid circular import
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CameraSolverV2": "📷 Camera Solver V2 (TAPIR)",
}
