"""
Temporal Smoothing for SAM3DBody2abc

Provides temporal filtering to reduce jitter in:
- Trajectory (pred_cam_t) - camera-relative position (tx, ty, tz)
- Tracked depth (tracked_depth) - from depth estimation
- Optionally mesh vertices - for shape key smoothing

NOTE: Joint positions (joint_coords) are NOT smoothed because they represent
accurate pose relative to pelvis. The jitter comes from trajectory/depth estimation.

Smoothing methods:
- Gaussian: Symmetric smoothing, good for general jitter reduction
- EMA (Bidirectional): Exponential moving average, preserves sharp movements better
- Savitzky-Golay: Polynomial fitting, best for preserving peaks/valleys
"""

import numpy as np
import torch
import copy
from typing import Dict, Tuple, List, Optional
from scipy.ndimage import gaussian_filter1d

# Try to import Savitzky-Golay filter
try:
    from scipy.signal import savgol_filter
    HAS_SAVGOL = True
except ImportError:
    HAS_SAVGOL = False

# Try to import logger
try:
    from .lib.logger import get_logger
    log = get_logger("TemporalSmoothing")
except ImportError:
    class FallbackLogger:
        def info(self, msg): print(f"[Temporal Smoothing] {msg}")
        def warning(self, msg): print(f"[Temporal Smoothing] WARNING: {msg}")
        def error(self, msg): print(f"[Temporal Smoothing] ERROR: {msg}")
        def debug(self, msg): pass
    log = FallbackLogger()


# ============================================================================
# Smoothing Functions
# ============================================================================

def ema_smooth_1d(signal: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Exponential Moving Average smoothing.
    
    Args:
        signal: 1D array to smooth
        alpha: Smoothing factor (0-1), lower = more smooth
    
    Returns:
        Smoothed signal
    """
    result = np.zeros_like(signal, dtype=np.float64)
    result[0] = signal[0]
    
    for i in range(1, len(signal)):
        result[i] = alpha * signal[i] + (1 - alpha) * result[i - 1]
    
    return result


def bidirectional_ema(signal: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Bidirectional EMA (forward + backward averaged) for zero-phase smoothing.
    """
    forward = ema_smooth_1d(signal, alpha)
    backward = ema_smooth_1d(signal[::-1], alpha)[::-1]
    return (forward + backward) / 2


def smooth_signal_1d(
    signal: np.ndarray,
    method: str = "gaussian",
    sigma: float = 1.0,
    alpha: float = 0.3,
    window_length: int = 5,
    polyorder: int = 2
) -> np.ndarray:
    """
    Smooth a 1D signal using specified method.
    
    Args:
        signal: 1D array to smooth
        method: "gaussian", "ema", or "savgol"
        sigma: Gaussian sigma (for gaussian method)
        alpha: EMA alpha (for ema method)
        window_length: Window size for Savitzky-Golay
        polyorder: Polynomial order for Savitzky-Golay
    
    Returns:
        Smoothed signal
    """
    if len(signal) < 3:
        return signal.copy()
    
    signal = np.asarray(signal, dtype=np.float64)
    
    if method == "gaussian":
        return gaussian_filter1d(signal, sigma=sigma)
    elif method == "ema":
        return bidirectional_ema(signal, alpha=alpha)
    elif method == "savgol" and HAS_SAVGOL:
        # Ensure window_length is odd and <= len(signal)
        wl = min(window_length, len(signal))
        if wl % 2 == 0:
            wl -= 1
        wl = max(3, wl)
        po = min(polyorder, wl - 1)
        return savgol_filter(signal, wl, po)
    else:
        return signal.copy()


def smooth_vertices_sequence(
    vertices_list: List[np.ndarray],
    method: str = "gaussian",
    sigma: float = 1.0,
    alpha: float = 0.3,
    **kwargs
) -> List[np.ndarray]:
    """
    Smooth a sequence of mesh vertices temporally.
    
    Args:
        vertices_list: List of [N, 3] vertex arrays
        method: "gaussian", "ema", or "savgol"
        sigma: Gaussian sigma
        alpha: EMA alpha
    
    Returns:
        List of smoothed vertex arrays
    """
    if len(vertices_list) < 3:
        return [v.copy() for v in vertices_list]
    
    # Stack into [T, N, 3]
    vertices_stack = np.stack(vertices_list, axis=0).astype(np.float64)
    T, N, D = vertices_stack.shape
    
    # Smooth along time axis for each vertex coordinate
    smoothed = np.zeros_like(vertices_stack)
    
    for v in range(N):
        for d in range(D):
            signal = vertices_stack[:, v, d]
            smoothed[:, v, d] = smooth_signal_1d(signal, method, sigma, alpha, **kwargs)
    
    return [smoothed[t].astype(np.float32) for t in range(T)]


# ============================================================================
# ComfyUI Node
# ============================================================================

class TemporalSmoothing:
    """
    Apply temporal smoothing to mesh sequence trajectory and depth.
    
    Reduces jitter in:
    - pred_cam_t (tx, ty, tz) - camera-relative position
    - tracked_depth - depth from estimation
    - Optionally: mesh vertices
    
    Does NOT smooth:
    - joint_coords - these are accurate pose data
    - keypoints_2d - these are detection results
    """
    
    METHODS = ["gaussian", "ema", "savgol"] if HAS_SAVGOL else ["gaussian", "ema"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE", {
                    "tooltip": "Mesh sequence from Video Batch Processor or Character Trajectory"
                }),
            },
            "optional": {
                "method": (cls.METHODS, {
                    "default": "gaussian",
                    "tooltip": "Smoothing method: gaussian (symmetric), ema (preserves sharp moves), savgol (preserves peaks)"
                }),
                "smoothing_strength": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Higher = more smoothing. Gaussian: sigma value. EMA: 1/alpha."
                }),
                "smooth_trajectory_xy": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Smooth pred_cam_t X and Y (screen position)"
                }),
                "smooth_trajectory_z": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Smooth pred_cam_t Z and tracked_depth (depth)"
                }),
                "smooth_vertices": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Smooth mesh vertices (for shape keys). Usually not needed."
                }),
                "preserve_endpoints": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep first and last frame values unchanged"
                }),
            }
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE", "STRING")
    RETURN_NAMES = ("smoothed_sequence", "smoothing_info")
    FUNCTION = "smooth"
    CATEGORY = "SAM3DBody2abc/Processing"
    
    def smooth(
        self,
        mesh_sequence: Dict,
        method: str = "gaussian",
        smoothing_strength: float = 1.5,
        smooth_trajectory_xy: bool = True,
        smooth_trajectory_z: bool = True,
        smooth_vertices: bool = False,
        preserve_endpoints: bool = True,
    ) -> Tuple[Dict, str]:
        """Apply temporal smoothing to mesh sequence."""
        
        log.info("=" * 60)
        log.info("TEMPORAL SMOOTHING")
        log.info("=" * 60)
        log.info(f"Method: {method}")
        log.info(f"Strength: {smoothing_strength}")
        log.info(f"Smooth XY: {smooth_trajectory_xy}")
        log.info(f"Smooth Z/Depth: {smooth_trajectory_z}")
        log.info(f"Smooth Vertices: {smooth_vertices}")
        
        # Deep copy to avoid modifying original
        result = copy.deepcopy(mesh_sequence)
        frames = result.get("frames", {})
        frame_indices = sorted(frames.keys())
        T = len(frame_indices)
        
        if T < 3:
            log.warning(f"Only {T} frames - need at least 3 for smoothing")
            return (result, f"Skipped: Only {T} frames (need 3+)")
        
        log.info(f"Processing {T} frames")
        
        # Convert strength to method parameters
        if method == "gaussian":
            sigma = smoothing_strength
            alpha = 0.3
            window_length = 5
        elif method == "ema":
            sigma = 1.0
            alpha = 1.0 / max(smoothing_strength, 0.1)
            alpha = max(0.05, min(0.9, alpha))
            window_length = 5
        else:  # savgol
            sigma = 1.0
            alpha = 0.3
            window_length = int(smoothing_strength * 2) * 2 + 1  # Ensure odd
            window_length = max(3, min(window_length, T if T % 2 == 1 else T - 1))
        
        stats = {
            "tx_range_before": 0, "tx_range_after": 0,
            "ty_range_before": 0, "ty_range_after": 0,
            "tz_range_before": 0, "tz_range_after": 0,
            "depth_range_before": 0, "depth_range_after": 0,
        }
        
        # Extract trajectory data
        tx_values = []
        ty_values = []
        tz_values = []
        depth_values = []
        
        for idx in frame_indices:
            frame = frames[idx]
            cam_t = frame.get("pred_cam_t")
            
            if cam_t is not None:
                if hasattr(cam_t, 'numpy'):
                    cam_t = cam_t.numpy()
                cam_t = np.array(cam_t).flatten()
                
                tx_values.append(float(cam_t[0]) if len(cam_t) > 0 else 0.0)
                ty_values.append(float(cam_t[1]) if len(cam_t) > 1 else 0.0)
                tz_values.append(float(cam_t[2]) if len(cam_t) > 2 else 5.0)
            else:
                tx_values.append(0.0)
                ty_values.append(0.0)
                tz_values.append(5.0)
            
            depth = frame.get("tracked_depth")
            if depth is not None:
                depth_values.append(float(depth))
            else:
                depth_values.append(tz_values[-1])
        
        tx_values = np.array(tx_values)
        ty_values = np.array(ty_values)
        tz_values = np.array(tz_values)
        depth_values = np.array(depth_values)
        
        # Record before stats
        stats["tx_range_before"] = float(tx_values.max() - tx_values.min())
        stats["ty_range_before"] = float(ty_values.max() - ty_values.min())
        stats["tz_range_before"] = float(tz_values.max() - tz_values.min())
        stats["depth_range_before"] = float(depth_values.max() - depth_values.min())
        
        # Smooth trajectory XY
        if smooth_trajectory_xy:
            tx_smoothed = smooth_signal_1d(tx_values, method, sigma, alpha, window_length)
            ty_smoothed = smooth_signal_1d(ty_values, method, sigma, alpha, window_length)
            
            if preserve_endpoints:
                tx_smoothed[0] = tx_values[0]
                tx_smoothed[-1] = tx_values[-1]
                ty_smoothed[0] = ty_values[0]
                ty_smoothed[-1] = ty_values[-1]
            
            log.info(f"Smoothed trajectory XY")
        else:
            tx_smoothed = tx_values
            ty_smoothed = ty_values
        
        # Smooth trajectory Z and depth
        if smooth_trajectory_z:
            tz_smoothed = smooth_signal_1d(tz_values, method, sigma, alpha, window_length)
            depth_smoothed = smooth_signal_1d(depth_values, method, sigma, alpha, window_length)
            
            if preserve_endpoints:
                tz_smoothed[0] = tz_values[0]
                tz_smoothed[-1] = tz_values[-1]
                depth_smoothed[0] = depth_values[0]
                depth_smoothed[-1] = depth_values[-1]
            
            log.info(f"Smoothed trajectory Z and tracked_depth")
        else:
            tz_smoothed = tz_values
            depth_smoothed = depth_values
        
        # Record after stats
        stats["tx_range_after"] = float(tx_smoothed.max() - tx_smoothed.min())
        stats["ty_range_after"] = float(ty_smoothed.max() - ty_smoothed.min())
        stats["tz_range_after"] = float(tz_smoothed.max() - tz_smoothed.min())
        stats["depth_range_after"] = float(depth_smoothed.max() - depth_smoothed.min())
        
        # Apply smoothed values back to frames
        for i, idx in enumerate(frame_indices):
            frame = frames[idx]
            
            # Update pred_cam_t
            frame["pred_cam_t"] = np.array([
                tx_smoothed[i],
                ty_smoothed[i],
                tz_smoothed[i]
            ], dtype=np.float32)
            
            # Update tracked_depth
            frame["tracked_depth"] = float(depth_smoothed[i])
            
            # Store original values for reference
            frame["pred_cam_t_original"] = np.array([
                tx_values[i],
                ty_values[i],
                tz_values[i]
            ], dtype=np.float32)
            frame["tracked_depth_original"] = float(depth_values[i])
        
        # Optionally smooth vertices
        vertices_smoothed = 0
        if smooth_vertices:
            # Check if vertices exist
            first_frame = frames[frame_indices[0]]
            if "vertices" in first_frame and first_frame["vertices"] is not None:
                vertices_list = []
                for idx in frame_indices:
                    v = frames[idx].get("vertices")
                    if v is not None:
                        if hasattr(v, 'numpy'):
                            v = v.numpy()
                        vertices_list.append(np.array(v))
                
                if len(vertices_list) == T:
                    smoothed_vertices = smooth_vertices_sequence(
                        vertices_list, method, sigma, alpha
                    )
                    
                    for i, idx in enumerate(frame_indices):
                        frames[idx]["vertices"] = smoothed_vertices[i]
                        frames[idx]["vertices_original"] = vertices_list[i]
                    
                    vertices_smoothed = T
                    log.info(f"Smoothed {vertices_smoothed} frames of vertices")
        
        # Add smoothing metadata
        result["temporal_smoothing"] = {
            "method": method,
            "strength": smoothing_strength,
            "smooth_xy": smooth_trajectory_xy,
            "smooth_z": smooth_trajectory_z,
            "smooth_vertices": smooth_vertices,
            "preserve_endpoints": preserve_endpoints,
            "stats": stats,
        }
        
        # Generate info string
        info_lines = [
            "=== TEMPORAL SMOOTHING RESULTS ===",
            f"Method: {method}",
            f"Strength: {smoothing_strength}",
            f"Frames: {T}",
            "",
            "=== TRAJECTORY X (horizontal) ===",
            f"  Range before: {stats['tx_range_before']:.4f}",
            f"  Range after:  {stats['tx_range_after']:.4f}",
            f"  Reduction:    {(1 - stats['tx_range_after']/max(stats['tx_range_before'], 0.0001))*100:.1f}%" if smooth_trajectory_xy else "  (not smoothed)",
            "",
            "=== TRAJECTORY Y (vertical) ===",
            f"  Range before: {stats['ty_range_before']:.4f}",
            f"  Range after:  {stats['ty_range_after']:.4f}",
            f"  Reduction:    {(1 - stats['ty_range_after']/max(stats['ty_range_before'], 0.0001))*100:.1f}%" if smooth_trajectory_xy else "  (not smoothed)",
            "",
            "=== TRAJECTORY Z (depth) ===",
            f"  Range before: {stats['tz_range_before']:.4f}",
            f"  Range after:  {stats['tz_range_after']:.4f}",
            f"  Reduction:    {(1 - stats['tz_range_after']/max(stats['tz_range_before'], 0.0001))*100:.1f}%" if smooth_trajectory_z else "  (not smoothed)",
            "",
            "=== TRACKED DEPTH ===",
            f"  Range before: {stats['depth_range_before']:.4f}m",
            f"  Range after:  {stats['depth_range_after']:.4f}m",
            f"  Reduction:    {(1 - stats['depth_range_after']/max(stats['depth_range_before'], 0.0001))*100:.1f}%" if smooth_trajectory_z else "  (not smoothed)",
        ]
        
        if smooth_vertices and vertices_smoothed > 0:
            info_lines.extend([
                "",
                f"=== VERTICES ===",
                f"  Smoothed {vertices_smoothed} frames",
            ])
        
        info_lines.extend([
            "",
            "NOTE: joint_coords NOT smoothed (pose data is accurate)",
        ])
        
        info = "\n".join(info_lines)
        
        log.info(f"Trajectory Z jitter reduced by {(1 - stats['tz_range_after']/max(stats['tz_range_before'], 0.0001))*100:.1f}%")
        log.info("=" * 60)
        
        return (result, info)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "TemporalSmoothing": TemporalSmoothing,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TemporalSmoothing": "🔄 Temporal Smoothing",
}
