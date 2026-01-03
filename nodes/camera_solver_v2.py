"""
Camera Solver V2 for SAM3DBody2abc v5.0

TAPIR-based camera solver with:
- Temporal point tracking (not frame-pair matching)
- Automatic shot type classification
- Rainbow trail debug visualization
- Background-only tracking (foreground masked)

Key Differences from v4.x CameraSolver:
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
    
    # Download checkpoint
    wget https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt

Version: 5.0.0
Author: SAM3DBody2abc Project
License: Apache 2.0 (TAPIR: Apache 2.0)
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import os
import colorsys
from typing import Dict, Tuple, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum

# Check for TAPIR availability
TAPIR_AVAILABLE = False
TAPIR_IMPORT_ERROR = None
print("[CameraSolverV2] Attempting to import TAPIR...")
try:
    # Import only the torch components we need, avoiding tensorflow dependency
    import sys
    
    # Block tensorflow import to avoid the dependency issue
    class TensorFlowBlocker:
        def find_module(self, name, path=None):
            if name == 'tensorflow' or name.startswith('tensorflow.'):
                return self
            return None
        def load_module(self, name):
            raise ImportError(f"TensorFlow import blocked - not needed for TAPIR torch inference")
    
    # Temporarily block tensorflow
    tf_blocker = TensorFlowBlocker()
    sys.meta_path.insert(0, tf_blocker)
    
    try:
        from tapnet.torch import tapir_model
        from tapnet.utils import transforms as tapir_transforms
        TAPIR_AVAILABLE = True
        print("[CameraSolverV2] ✅ TAPIR module imported successfully (TensorFlow blocked)")
    finally:
        # Remove the blocker
        if tf_blocker in sys.meta_path:
            sys.meta_path.remove(tf_blocker)
            
except ImportError as e:
    TAPIR_IMPORT_ERROR = str(e)
    print(f"[CameraSolverV2] ❌ TAPIR ImportError: {e}")
    print("[CameraSolverV2] Try: pip install tensorflow")
    print("[CameraSolverV2] Or: pip install 'tapnet[torch] @ git+https://github.com/google-deepmind/tapnet.git'")
except Exception as e:
    TAPIR_IMPORT_ERROR = str(e)
    print(f"[CameraSolverV2] ❌ TAPIR import failed with unexpected error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()


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
        """Load TAPIR model."""
        print(f"[CameraSolverV2] _load_tapir called, TAPIR_AVAILABLE={TAPIR_AVAILABLE}")
        
        if not TAPIR_AVAILABLE:
            print("[CameraSolverV2] TAPIR module not available, cannot load")
            return False
        
        if self.tapir_model is not None and self._checkpoint_loaded == checkpoint_path:
            print(f"[CameraSolverV2] TAPIR already loaded from: {self._checkpoint_loaded}")
            return True
        
        # Resolve checkpoint path
        if not checkpoint_path:
            # Get paths relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))  # nodes/
            extension_dir = os.path.dirname(current_dir)  # ComfyUI-SAM3DBody2abc/
            custom_nodes_dir = os.path.dirname(extension_dir)  # custom_nodes/
            comfyui_dir = os.path.dirname(custom_nodes_dir)  # ComfyUI/
            
            print(f"[CameraSolverV2] Path resolution:")
            print(f"[CameraSolverV2]   current_dir: {current_dir}")
            print(f"[CameraSolverV2]   extension_dir: {extension_dir}")
            print(f"[CameraSolverV2]   custom_nodes_dir: {custom_nodes_dir}")
            print(f"[CameraSolverV2]   comfyui_dir: {comfyui_dir}")
            
            # Search order:
            # 1. ComfyUI/models/tapir/ (standard ComfyUI convention)
            # 2. ComfyUI-SAM3DBody2abc/models/tapir/ (self-contained)
            # 3. ~/.cache/tapir/ (user cache)
            # 4. Current working directory
            possible_paths = [
                # Standard ComfyUI models path
                os.path.join(comfyui_dir, "models", "tapir", "bootstapir_checkpoint_v2.pt"),
                # Self-contained in extension (FIXED PATH)
                os.path.join(extension_dir, "models", "tapir", "bootstapir_checkpoint_v2.pt"),
                # User cache
                os.path.expanduser("~/.cache/tapir/bootstapir_checkpoint_v2.pt"),
                # Current directory
                "bootstapir_checkpoint_v2.pt",
            ]
            
            print(f"[CameraSolverV2] Searching for checkpoint in:")
            for i, path in enumerate(possible_paths, 1):
                exists = os.path.exists(path)
                print(f"[CameraSolverV2]   {i}. {path} -> {'FOUND' if exists else 'not found'}")
                if exists and not checkpoint_path:
                    checkpoint_path = path
        
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"[CameraSolverV2] ❌ TAPIR checkpoint not found!")
            print(f"[CameraSolverV2] Download from: https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt")
            print(f"[CameraSolverV2] Place in one of these locations:")
            print(f"[CameraSolverV2]   1. ComfyUI/models/tapir/bootstapir_checkpoint_v2.pt (recommended)")
            print(f"[CameraSolverV2]   2. ComfyUI-SAM3DBody2abc/models/tapir/bootstapir_checkpoint_v2.pt")
            return False
        
        try:
            print(f"[CameraSolverV2] Loading TAPIR from {checkpoint_path}...")
            self.tapir_model = tapir_model.TAPIR(pyramid_level=1)
            print(f"[CameraSolverV2] TAPIR model created, loading weights...")
            self.tapir_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"[CameraSolverV2] Weights loaded, moving to {self.device}...")
            self.tapir_model = self.tapir_model.to(self.device)
            self.tapir_model.eval()
            self._checkpoint_loaded = checkpoint_path
            print(f"[CameraSolverV2] ✅ TAPIR loaded successfully on {self.device}")
            return True
        except Exception as e:
            print(f"[CameraSolverV2] ❌ Failed to load TAPIR: {e}")
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
        print(f"[CameraSolverV2] Processing {num_frames} frames ({W}x{H})")
        print(f"[CameraSolverV2] Quality: {quality}, Grid: {grid_size}x{grid_size}")
        print(f"[CameraSolverV2] TAPIR_AVAILABLE: {TAPIR_AVAILABLE}")
        print(f"[CameraSolverV2] debug_visualization: {debug_visualization}")
        print(f"{'='*60}")
        
        # Check TAPIR availability
        print(f"[CameraSolverV2] Calling _load_tapir...")
        if not self._load_tapir(checkpoint_path):
            print(f"[CameraSolverV2] _load_tapir returned False, using fallback")
            return self._fallback_static(images, intrinsics)
        
        print(f"[CameraSolverV2] TAPIR loaded, generating query points...")
        
        # Step 1: Generate query points on background
        query_points = self._generate_query_points(
            H, W, grid_size, foreground_mask, quality
        )
        
        if query_points is None or len(query_points) < 10:
            print(f"[CameraSolverV2] Not enough valid query points ({query_points.shape if query_points is not None else 'None'}), falling back to static")
            return self._fallback_static(images, intrinsics)
        
        print(f"[CameraSolverV2] Generated {len(query_points)} query points, running TAPIR tracking...")
        
        # Step 2: Run TAPIR tracking
        tracks, visibles = self._run_tapir(images, query_points)
        
        if tracks is None:
            print(f"[CameraSolverV2] TAPIR tracking failed, using fallback")
            return self._fallback_static(images, intrinsics)
        
        print(f"[CameraSolverV2] TAPIR tracking complete: tracks shape {tracks.shape}")
        
        # Step 3: Classify shot type
        shot_classification = self._classify_shot(
            tracks, visibles, intrinsics, force_shot_type
        )
        
        print(f"[CameraSolverV2] Shot type: {shot_classification.shot_type.value} "
              f"(confidence: {shot_classification.confidence:.2f})")
        
        # Step 4: Solve camera based on shot type
        print(f"[CameraSolverV2] Solving camera motion...")
        camera_matrices = self._solve_camera(
            tracks, visibles, intrinsics, shot_classification, smoothing_window
        )
        
        # Step 5: Generate debug visualization
        if debug_visualization:
            print(f"[CameraSolverV2] Generating rainbow trail visualization...")
            debug_vis = self._render_rainbow_trails(
                images, tracks, visibles, trail_length
            )
            print(f"[CameraSolverV2] Debug visualization complete: {debug_vis.shape}")
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
        
        print(f"[CameraSolverV2] ✅ Complete: {status}")
        
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
        
        # All points start at frame 0
        for y in y_coords:
            for x in x_coords:
                yi, xi = int(y), int(x)
                
                # Check if point is on background (not masked)
                if foreground_mask is not None:
                    # Mask is [N, H, W] or [H, W]
                    if foreground_mask.dim() == 3:
                        mask_val = foreground_mask[0, yi, xi].item()
                    else:
                        mask_val = foreground_mask[yi, xi].item()
                    
                    # Skip if in foreground (mask > 0.5)
                    if mask_val > 0.5:
                        continue
                
                # TAPIR query format: [t, y, x]
                query_points.append([0, yi, xi])
        
        if len(query_points) < 10:
            print(f"[CameraSolverV2] Warning: Only {len(query_points)} background points found")
            # Add some points anyway if mask is too aggressive
            for y in y_coords[::2]:
                for x in x_coords[::2]:
                    query_points.append([0, int(y), int(x)])
        
        print(f"[CameraSolverV2] Generated {len(query_points)} query points")
        return np.array(query_points, dtype=np.int32)
    
    def _run_tapir(
        self,
        images: torch.Tensor,
        query_points: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Run TAPIR tracking on video frames."""
        
        try:
            # Prepare video: [B, T, H, W, C] normalized to [-1, 1]
            video = images.float()  # [T, H, W, C]
            video = video / 255.0 * 2 - 1 if video.max() > 1 else video * 2 - 1
            video = video.unsqueeze(0).to(self.device)  # [1, T, H, W, C]
            
            # Prepare query points: [B, N, 3]
            query_tensor = torch.tensor(query_points, dtype=torch.float32)
            query_tensor = query_tensor.unsqueeze(0).to(self.device)  # [1, N, 3]
            
            print(f"[CameraSolverV2] Running TAPIR inference...")
            
            with torch.no_grad():
                outputs = self.tapir_model(video, query_tensor)
            
            # Extract results
            tracks = outputs['tracks'][0].cpu().numpy()  # [N, T, 2] - (x, y)
            occlusions = outputs['occlusion'][0]  # [N, T]
            expected_dist = outputs['expected_dist'][0]  # [N, T]
            
            # Compute visibility
            visibles = (1 - torch.sigmoid(occlusions)) * (1 - torch.sigmoid(expected_dist)) > 0.5
            visibles = visibles.cpu().numpy()  # [N, T]
            
            print(f"[CameraSolverV2] Tracking complete: {tracks.shape[0]} points x {tracks.shape[1]} frames")
            
            return tracks, visibles
            
        except Exception as e:
            print(f"[CameraSolverV2] TAPIR inference error: {e}")
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
    ) -> List[Dict]:
        """Solve camera matrices based on shot type."""
        
        num_frames = tracks.shape[1]
        matrices = []
        
        if classification.shot_type == ShotType.STATIC:
            # Identity matrices for all frames
            for t in range(num_frames):
                matrices.append({
                    "frame": t,
                    "matrix": np.eye(4).tolist(),
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
        
        # Reference frame is frame 0
        ref_frame = 0
        
        all_rotations = []
        
        for t in range(num_frames):
            if t == ref_frame:
                R = np.eye(3)
            else:
                # Get visible points in both frames
                vis_both = visibles[:, ref_frame] & visibles[:, t]
                if np.sum(vis_both) < 8:
                    # Not enough points, use identity
                    R = np.eye(3)
                else:
                    pts0 = tracks[vis_both, ref_frame, :].astype(np.float64)
                    pts1 = tracks[vis_both, t, :].astype(np.float64)
                    
                    # Find homography
                    H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)
                    
                    if H is not None:
                        # Decompose homography to get rotation
                        # For pure rotation: H = K * R * K^-1
                        # Therefore: R = K^-1 * H * K
                        R = K_inv @ H @ K
                        
                        # Ensure R is a proper rotation matrix via SVD
                        U, _, Vt = np.linalg.svd(R)
                        R = U @ Vt
                        
                        # Ensure det(R) = 1
                        if np.linalg.det(R) < 0:
                            R = -R
                    else:
                        R = np.eye(3)
            
            all_rotations.append(R)
        
        # Smooth rotations if requested
        if smoothing_window > 0 and len(all_rotations) > smoothing_window:
            all_rotations = self._smooth_rotations(all_rotations, smoothing_window)
        
        # Convert to 4x4 matrices and extract Euler angles
        for t, R in enumerate(all_rotations):
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
        
        return matrices
    
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
        """Extract ZXY Euler angles from rotation matrix (camera convention)."""
        # ZXY decomposition for camera: Z=roll, X=tilt, Y=pan
        # This matches Maya's camera rotation order
        
        # Clamp to avoid numerical issues
        sy = np.clip(R[0, 2], -1.0, 1.0)
        
        if abs(sy) < 0.99999:
            tilt = np.arcsin(sy)
            pan = np.arctan2(-R[0, 1], R[0, 0])
            roll = np.arctan2(-R[1, 2], R[2, 2])
        else:
            # Gimbal lock
            tilt = np.pi / 2 * np.sign(sy)
            pan = np.arctan2(R[1, 0], R[1, 1])
            roll = 0
        
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
