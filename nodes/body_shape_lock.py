"""
Body Shape Lock for SAM3DBody2abc
==================================

Ensures consistent body shape (SMPL beta parameters) across all frames.

The Problem:
    SAM3DBody estimates body shape (beta) independently per frame.
    This causes the body to flicker in size/proportions between frames,
    even though a person's body shape doesn't actually change.

The Solution:
    1. Collect beta parameters from all frames
    2. Compute a single consistent shape (median or mean)
    3. Apply that shape to all frames
    4. Optionally: use confidence weighting to trust better detections more

SMPL Beta Parameters:
    - 10 parameters controlling body shape
    - beta[0]: Overall body size
    - beta[1]: Body mass/weight
    - beta[2-9]: Other shape variations (height, limb length, etc.)

Usage:
    from body_shape_lock import lock_body_shape, BodyShapeLock
    
    # Quick usage
    locked_sequence = lock_body_shape(mesh_sequence)
    
    # With options
    locked_sequence = lock_body_shape(
        mesh_sequence,
        method="median",
        confidence_weighted=True
    )

Author: SAM3DBody2abc
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
from enum import Enum


class AggregationMethod(Enum):
    """Methods for aggregating beta parameters across frames."""
    MEAN = "mean"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"
    FIRST_FRAME = "first_frame"
    BEST_FRAME = "best_frame"  # Frame with highest confidence
    WEIGHTED_MEAN = "weighted_mean"  # Confidence-weighted mean


@dataclass
class BodyShapeConfig:
    """Configuration for body shape locking."""
    
    method: AggregationMethod = AggregationMethod.MEDIAN
    
    # For trimmed mean
    trim_percent: float = 10.0      # Percent to trim from each end
    
    # For confidence weighting
    use_confidence: bool = False
    min_confidence: float = 0.5     # Ignore frames below this confidence
    
    # Outlier rejection
    reject_outliers: bool = True
    outlier_std: float = 2.0        # Reject if > N std from mean
    
    # Smoothing (for gradual shape changes, rarely needed)
    allow_gradual_change: bool = False
    change_window: int = 30         # Frames for gradual change


class BodyShapeLock:
    """
    Locks body shape parameters to be consistent across a video.
    
    Processes MESH_SEQUENCE to ensure uniform body proportions.
    """
    
    def __init__(self, config: Optional[BodyShapeConfig] = None):
        """
        Initialize body shape locker.
        
        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or BodyShapeConfig()
    
    def process(
        self,
        mesh_sequence: Dict,
        verbose: bool = True
    ) -> Dict:
        """
        Lock body shape across all frames.
        
        Args:
            mesh_sequence: MESH_SEQUENCE from SAM3DBody
            verbose: Print progress information
        
        Returns:
            Modified mesh_sequence with consistent body shape
        """
        frames_data = mesh_sequence.get("frames", {})
        
        # Handle both dict and list formats
        frames_is_dict = isinstance(frames_data, dict)
        if frames_is_dict:
            # Sort keys to ensure consistent frame ordering
            frame_keys = sorted(frames_data.keys())
            frames = [frames_data[k] for k in frame_keys]
        else:
            frames = frames_data
            frame_keys = None
        
        if not frames:
            return mesh_sequence
        
        # 1. Extract beta parameters from all frames
        betas, confidences, valid_indices = self._extract_betas(frames)
        
        if len(betas) == 0:
            if verbose:
                print("[Body Shape] No beta parameters found in sequence")
            return mesh_sequence
        
        if verbose:
            print(f"[Body Shape] Found betas in {len(betas)}/{len(frames)} frames")
        
        # 2. Compute consistent shape
        locked_beta = self._compute_locked_shape(betas, confidences, verbose)
        
        if verbose:
            # Report shape change
            original_mean = np.mean(betas, axis=0)
            change = np.linalg.norm(locked_beta - original_mean)
            print(f"[Body Shape] Locked shape (change from mean: {change:.4f})")
            print(f"[Body Shape] First 3 betas: [{locked_beta[0]:.3f}, {locked_beta[1]:.3f}, {locked_beta[2]:.3f}]")
        
        # 3. Apply locked shape to all frames
        result = self._apply_locked_shape(mesh_sequence, locked_beta, verbose, frames_is_dict, frame_keys)
        
        # 4. Store metadata
        result["body_shape_locked"] = {
            "method": self.config.method.value,
            "locked_beta": locked_beta.tolist(),
            "original_variance": float(np.mean(np.var(betas, axis=0))),
            "frames_used": len(betas),
        }
        
        return result
    
    def _extract_betas(
        self,
        frames: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Extract beta parameters from frames.
        
        Returns:
            betas: [N, 10] array of beta parameters
            confidences: [N] array of confidence scores
            valid_indices: List of frame indices with valid betas
        """
        betas = []
        confidences = []
        valid_indices = []
        
        for i, frame in enumerate(frames):
            beta = None
            confidence = 1.0
            
            # Try different locations for beta
            if "pose_params" in frame and frame["pose_params"]:
                beta = frame["pose_params"].get("shape")
                beta = beta if beta is None else frame["pose_params"].get("betas", beta)
            
            if beta is None:
                beta = frame.get("betas")
            
            if beta is None:
                beta = frame.get("shape")
            
            if beta is not None:
                beta = np.array(beta).flatten()
                
                # Ensure 10 parameters
                if len(beta) < 10:
                    beta = np.pad(beta, (0, 10 - len(beta)))
                elif len(beta) > 10:
                    beta = beta[:10]
                
                # Get confidence if available
                if "pose_params" in frame and frame["pose_params"]:
                    confidence = frame["pose_params"].get("confidence", 1.0)
                if "confidence" in frame:
                    confidence = frame["confidence"]
                
                betas.append(beta)
                confidences.append(confidence)
                valid_indices.append(i)
        
        if len(betas) == 0:
            return np.array([]), np.array([]), []
        
        return np.array(betas), np.array(confidences), valid_indices
    
    def _compute_locked_shape(
        self,
        betas: np.ndarray,
        confidences: np.ndarray,
        verbose: bool
    ) -> np.ndarray:
        """
        Compute the locked (consistent) shape parameters.
        
        Args:
            betas: [N, 10] array of beta parameters
            confidences: [N] array of confidence scores
        
        Returns:
            locked_beta: [10] array of locked beta parameters
        """
        cfg = self.config
        
        # Filter by minimum confidence
        if cfg.use_confidence and cfg.min_confidence > 0:
            mask = confidences >= cfg.min_confidence
            if np.sum(mask) > 0:
                betas = betas[mask]
                confidences = confidences[mask]
        
        # Reject outliers
        if cfg.reject_outliers and len(betas) > 3:
            mean = np.mean(betas, axis=0)
            std = np.std(betas, axis=0)
            
            # Mark outliers (any beta > N std from mean)
            outlier_mask = np.any(np.abs(betas - mean) > cfg.outlier_std * std, axis=1)
            inlier_mask = ~outlier_mask
            
            if np.sum(inlier_mask) > 0:
                if verbose and np.sum(outlier_mask) > 0:
                    print(f"[Body Shape] Rejected {np.sum(outlier_mask)} outlier frames")
                betas = betas[inlier_mask]
                confidences = confidences[inlier_mask]
        
        # Compute locked shape based on method
        method = cfg.method
        
        if method == AggregationMethod.MEAN:
            locked_beta = np.mean(betas, axis=0)
        
        elif method == AggregationMethod.MEDIAN:
            locked_beta = np.median(betas, axis=0)
        
        elif method == AggregationMethod.TRIMMED_MEAN:
            trim = int(len(betas) * cfg.trim_percent / 100)
            if trim > 0 and len(betas) > 2 * trim:
                # Sort and trim each beta independently
                locked_beta = np.zeros(10)
                for i in range(10):
                    sorted_vals = np.sort(betas[:, i])
                    locked_beta[i] = np.mean(sorted_vals[trim:-trim])
            else:
                locked_beta = np.mean(betas, axis=0)
        
        elif method == AggregationMethod.FIRST_FRAME:
            locked_beta = betas[0]
        
        elif method == AggregationMethod.BEST_FRAME:
            best_idx = np.argmax(confidences)
            locked_beta = betas[best_idx]
        
        elif method == AggregationMethod.WEIGHTED_MEAN:
            # Confidence-weighted mean
            weights = confidences / np.sum(confidences)
            locked_beta = np.sum(betas * weights[:, np.newaxis], axis=0)
        
        else:
            locked_beta = np.median(betas, axis=0)
        
        return locked_beta
    
    def _apply_locked_shape(
        self,
        mesh_sequence: Dict,
        locked_beta: np.ndarray,
        verbose: bool,
        frames_is_dict: bool = False,
        frame_keys: list = None
    ) -> Dict:
        """
        Apply locked shape to all frames.
        
        Args:
            mesh_sequence: Original mesh sequence
            locked_beta: [10] locked beta parameters
            frames_is_dict: Whether frames are stored as dict
            frame_keys: Original dict keys if frames were dict
        
        Returns:
            Modified mesh sequence
        """
        result = mesh_sequence.copy()
        frames_data = mesh_sequence.get("frames", {})
        
        # Handle both dict and list formats
        if frames_is_dict or isinstance(frames_data, dict):
            # Process dict format
            new_frames = {}
            for key, frame in frames_data.items():
                new_frame = frame.copy() if isinstance(frame, dict) else frame
                if isinstance(new_frame, dict):
                    # Update pose_params.shape
                    if "pose_params" in new_frame:
                        if new_frame["pose_params"] is None:
                            new_frame["pose_params"] = {}
                        new_frame["pose_params"]["shape"] = locked_beta.tolist()
                        new_frame["pose_params"]["betas"] = locked_beta.tolist()
                    
                    # Update direct beta fields if present
                    if "betas" in new_frame:
                        new_frame["betas"] = locked_beta.tolist()
                    if "shape" in new_frame:
                        new_frame["shape"] = locked_beta.tolist()
                
                new_frames[key] = new_frame
            result["frames"] = new_frames
            
            if verbose:
                print(f"[Body Shape] Applied locked shape to {len(new_frames)} frames")
        else:
            # Process list format
            frames = [f.copy() if isinstance(f, dict) else f for f in frames_data]
            
            for frame in frames:
                if not isinstance(frame, dict):
                    continue
                # Update pose_params.shape
                if "pose_params" in frame:
                    if frame["pose_params"] is None:
                        frame["pose_params"] = {}
                    frame["pose_params"]["shape"] = locked_beta.tolist()
                    frame["pose_params"]["betas"] = locked_beta.tolist()
                
                # Update direct beta fields if present
                if "betas" in frame:
                    frame["betas"] = locked_beta.tolist()
                if "shape" in frame:
                    frame["shape"] = locked_beta.tolist()
            
            result["frames"] = frames
            
            if verbose:
                print(f"[Body Shape] Applied locked shape to {len(frames)} frames")
        
        return result


def lock_body_shape(
    mesh_sequence: Dict,
    method: str = "median",
    confidence_weighted: bool = False,
    reject_outliers: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Convenience function to lock body shape in a mesh sequence.
    
    Args:
        mesh_sequence: MESH_SEQUENCE from SAM3DBody
        method: Aggregation method ("mean", "median", "trimmed_mean", 
                "first_frame", "best_frame", "weighted_mean")
        confidence_weighted: Use confidence scores for weighting
        reject_outliers: Reject frames with outlier betas
        verbose: Print progress
    
    Returns:
        Modified mesh_sequence with consistent body shape
    """
    # Map string to enum
    method_map = {
        "mean": AggregationMethod.MEAN,
        "median": AggregationMethod.MEDIAN,
        "trimmed_mean": AggregationMethod.TRIMMED_MEAN,
        "first_frame": AggregationMethod.FIRST_FRAME,
        "best_frame": AggregationMethod.BEST_FRAME,
        "weighted_mean": AggregationMethod.WEIGHTED_MEAN,
    }
    
    config = BodyShapeConfig(
        method=method_map.get(method, AggregationMethod.MEDIAN),
        use_confidence=confidence_weighted,
        reject_outliers=reject_outliers,
    )
    
    locker = BodyShapeLock(config)
    return locker.process(mesh_sequence, verbose=verbose)


def analyze_shape_variance(mesh_sequence: Dict) -> Dict:
    """
    Analyze the variance in body shape across frames.
    
    Useful for diagnosing shape flickering issues.
    
    Args:
        mesh_sequence: MESH_SEQUENCE from SAM3DBody
    
    Returns:
        Analysis dict with variance statistics
    """
    frames_data = mesh_sequence.get("frames", {})
    
    # Handle both dict and list formats for frames
    if isinstance(frames_data, dict):
        # Sort keys to ensure consistent frame ordering
        frame_keys = sorted(frames_data.keys())
        frames = [frames_data[k] for k in frame_keys]
    else:
        frames = frames_data
    
    # Extract betas
    betas = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        beta = None
        if "pose_params" in frame and frame["pose_params"]:
            beta = frame["pose_params"].get("shape")
        if beta is None:
            beta = frame.get("betas")
        if beta is not None:
            betas.append(np.array(beta).flatten()[:10])
    
    if len(betas) == 0:
        return {"error": "No betas found"}
    
    betas = np.array(betas)
    
    return {
        "num_frames": len(betas),
        "mean_beta": betas.mean(axis=0).tolist(),
        "std_beta": betas.std(axis=0).tolist(),
        "min_beta": betas.min(axis=0).tolist(),
        "max_beta": betas.max(axis=0).tolist(),
        "total_variance": float(np.sum(np.var(betas, axis=0))),
        "per_component_variance": np.var(betas, axis=0).tolist(),
        "range_per_component": (betas.max(axis=0) - betas.min(axis=0)).tolist(),
    }


# =============================================================================
# ComfyUI Node
# =============================================================================

class BodyShapeLockNode:
    """
    ComfyUI node for locking body shape across frames.
    
    Ensures consistent body proportions by fixing SMPL beta parameters.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE",),
            },
            "optional": {
                "method": ([
                    "Median (Recommended)",
                    "Mean",
                    "Trimmed Mean",
                    "First Frame",
                    "Best Frame (Confidence)",
                    "Weighted Mean (Confidence)"
                ], {
                    "default": "Median (Recommended)",
                    "tooltip": "How to compute the locked shape. Median is robust to outliers."
                }),
                "reject_outliers": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Reject frames with unusual body shapes."
                }),
                "outlier_threshold": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.5,
                    "tooltip": "Standard deviations for outlier rejection."
                }),
                "log_level": (["Normal", "Verbose", "Silent"], {
                    "default": "Normal",
                }),
            }
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE", "STRING")
    RETURN_NAMES = ("mesh_sequence", "status")
    FUNCTION = "lock_shape"
    CATEGORY = "SAM3DBody2abc/Processing"
    
    def lock_shape(
        self,
        mesh_sequence: Dict,
        method: str = "Median (Recommended)",
        reject_outliers: bool = True,
        outlier_threshold: float = 2.0,
        log_level: str = "Normal",
    ) -> Tuple[Dict, str]:
        """Lock body shape across all frames."""
        verbose = log_level != "Silent"
        
        # Map method name
        method_map = {
            "Median (Recommended)": "median",
            "Mean": "mean",
            "Trimmed Mean": "trimmed_mean",
            "First Frame": "first_frame",
            "Best Frame (Confidence)": "best_frame",
            "Weighted Mean (Confidence)": "weighted_mean",
        }
        method_key = method_map.get(method, "median")
        
        # Check original variance
        if verbose:
            analysis = analyze_shape_variance(mesh_sequence)
            if "total_variance" in analysis:
                print(f"[Body Shape] Original variance: {analysis['total_variance']:.4f}")
        
        # Configure and process
        config = BodyShapeConfig(
            method=AggregationMethod(method_key),
            reject_outliers=reject_outliers,
            outlier_std=outlier_threshold,
        )
        
        locker = BodyShapeLock(config)
        result = locker.process(mesh_sequence, verbose=verbose)
        
        # Build status
        locked_info = result.get("body_shape_locked", {})
        frames_used = locked_info.get("frames_used", 0)
        orig_var = locked_info.get("original_variance", 0)
        
        status = f"Locked shape using {method_key} ({frames_used} frames, orig variance: {orig_var:.4f})"
        
        return (result, status)


NODE_CLASS_MAPPINGS = {
    "BodyShapeLock": BodyShapeLockNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BodyShapeLock": "📐 Body Shape Lock",
}
