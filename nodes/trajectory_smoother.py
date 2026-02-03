"""
Trajectory Smoother for SAM3DBody2abc

Dedicated noise reduction for body_world_3d trajectory data.
Works on SUBJECT_MOTION from Motion Analyzer.

This node applies sophisticated smoothing to remove jitter while
preserving the overall motion path shape.

Smoothing Methods:
- Savitzky-Golay: Best for preserving shape (recommended)
- Gaussian: Good general-purpose smoothing
- Moving Average: Simple but effective
- Spline: Fits smooth curve through trajectory
- Kalman: Physics-based smoothing (constant velocity model)
- Joint-Guided: Uses stable joint detection to guide trajectory smoothing (NEW in v4.8.8)

v4.8.8: Added Joint-Guided smoothing that uses stable 2D joint detection
        to correct noisy pred_cam_t values.
"""

import numpy as np
import copy
from typing import Dict, Tuple, List, Optional

# Try to import scipy filters
try:
    from scipy.signal import savgol_filter
    HAS_SAVGOL = True
except ImportError:
    HAS_SAVGOL = False

try:
    from scipy.ndimage import gaussian_filter1d
    HAS_GAUSSIAN = True
except ImportError:
    HAS_GAUSSIAN = False

try:
    from scipy.interpolate import UnivariateSpline
    HAS_SPLINE = True
except ImportError:
    HAS_SPLINE = False

# Try to import logger
try:
    from ..lib.logger import log, set_module
    set_module("Trajectory Smoother")
except ImportError:
    class FallbackLogger:
        def info(self, msg): print(f"[Trajectory Smoother] {msg}")
        def warning(self, msg): print(f"[Trajectory Smoother] WARNING: {msg}")
        def warn(self, msg): print(f"[Trajectory Smoother] WARNING: {msg}")
        def error(self, msg): print(f"[Trajectory Smoother] ERROR: {msg}")
        def debug(self, msg): pass
    log = FallbackLogger()


# ============================================================================
# Joint Definitions
# ============================================================================

# COCO 17-joint format (used by keypoints_2d/3d)
COCO_JOINT_NAMES = [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
    "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"
]

# SMPL-H 22 body joints (from joint_coords)
SMPLH_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"
]


# ============================================================================
# Smoothing Functions
# ============================================================================

def moving_average(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average filter."""
    if window < 2:
        return signal
    
    # Pad signal to handle edges
    pad_width = window // 2
    padded = np.pad(signal, pad_width, mode='edge')
    
    # Compute moving average
    kernel = np.ones(window) / window
    smoothed = np.convolve(padded, kernel, mode='valid')
    
    return smoothed[:len(signal)]


def gaussian_smooth(signal: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gaussian smoothing filter."""
    if not HAS_GAUSSIAN:
        log.warning("scipy.ndimage not available, using moving average fallback")
        window = int(sigma * 4) | 1  # Make odd
        return moving_average(signal, window)
    
    return gaussian_filter1d(signal, sigma=sigma, mode='nearest')


def savgol_smooth(signal: np.ndarray, window: int = 11, polyorder: int = 3) -> np.ndarray:
    """
    Savitzky-Golay filter - best for preserving peaks and valleys.
    
    Args:
        signal: Input signal
        window: Window size (must be odd, larger = smoother)
        polyorder: Polynomial order (must be < window)
    """
    if not HAS_SAVGOL:
        log.warning("scipy.signal.savgol_filter not available, using Gaussian fallback")
        return gaussian_smooth(signal, sigma=window / 4)
    
    # Ensure window is odd and >= polyorder + 2
    window = max(window, polyorder + 2)
    if window % 2 == 0:
        window += 1
    
    # Ensure we have enough data
    if len(signal) < window:
        window = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
        if window < polyorder + 2:
            return signal  # Not enough data to smooth
    
    return savgol_filter(signal, window, polyorder, mode='nearest')


def spline_smooth(signal: np.ndarray, smoothing_factor: float = 0.5) -> np.ndarray:
    """
    Spline-based smoothing - fits a smooth curve through the data.
    
    Args:
        signal: Input signal
        smoothing_factor: 0 = interpolate exactly, 1 = very smooth
    """
    if not HAS_SPLINE:
        log.warning("scipy.interpolate not available, using Savitzky-Golay fallback")
        return savgol_smooth(signal, window=11)
    
    n = len(signal)
    if n < 4:
        return signal
    
    x = np.arange(n)
    
    # Compute smoothing parameter (s) based on data variance
    variance = np.var(signal)
    s = smoothing_factor * n * variance
    
    try:
        spline = UnivariateSpline(x, signal, s=s)
        return spline(x)
    except Exception as e:
        log.warning(f"Spline fitting failed: {e}, using Savitzky-Golay fallback")
        return savgol_smooth(signal, window=11)


def kalman_smooth(signal: np.ndarray, process_noise: float = 0.1, measurement_noise: float = 1.0) -> np.ndarray:
    """
    Simple 1D Kalman filter with constant velocity model.
    
    Args:
        signal: Input signal
        process_noise: How much we expect the signal to change (lower = smoother)
        measurement_noise: How noisy we believe the measurements are (higher = smoother)
    """
    n = len(signal)
    if n < 2:
        return signal
    
    # State: [position, velocity]
    # Initialize
    x = np.array([signal[0], 0.0])  # Initial state
    P = np.array([[1.0, 0.0], [0.0, 1.0]])  # Initial covariance
    
    # Process noise covariance
    Q = np.array([
        [process_noise, 0.0],
        [0.0, process_noise * 0.1]
    ])
    
    # Measurement noise
    R = measurement_noise
    
    # State transition matrix (constant velocity)
    F = np.array([
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    
    # Measurement matrix (we only observe position)
    H = np.array([[1.0, 0.0]])
    
    # Forward pass
    forward_states = []
    forward_covs = []
    
    for i in range(n):
        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        
        # Update
        z = signal[i]
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T / S
        
        x = x_pred + K.flatten() * y
        P = (np.eye(2) - np.outer(K, H)) @ P_pred
        
        forward_states.append(x.copy())
        forward_covs.append(P.copy())
    
    # Backward pass (RTS smoother)
    smoothed = np.zeros(n)
    smoothed[-1] = forward_states[-1][0]
    
    x_smooth = forward_states[-1]
    
    for i in range(n - 2, -1, -1):
        x_fwd = forward_states[i]
        P_fwd = forward_covs[i]
        
        x_pred = F @ x_fwd
        P_pred = F @ P_fwd @ F.T + Q
        
        # Smoother gain
        try:
            C = P_fwd @ F.T @ np.linalg.inv(P_pred)
        except:
            C = np.zeros((2, 2))
        
        x_smooth = x_fwd + C @ (x_smooth - x_pred)
        smoothed[i] = x_smooth[0]
    
    return smoothed


def joint_guided_smooth(
    trajectory: np.ndarray,
    joint_2d_trajectory: np.ndarray,
    strength: float = 0.5,
    preserve_endpoints: bool = True
) -> np.ndarray:
    """
    Joint-Guided smoothing: Uses stable 2D joint detection to guide trajectory.
    
    The key insight is that 2D joint detection is more stable than depth estimation.
    We smooth the 2D joint trajectory and use it to correct the noisy pred_cam_t.
    
    Args:
        trajectory: [N, 3] noisy world trajectory from pred_cam_t
        joint_2d_trajectory: [N, 2] 2D screen positions of reference joint
        strength: Smoothing strength 0-1
        preserve_endpoints: Keep first/last points unchanged
    
    Returns:
        Smoothed trajectory [N, 3]
    """
    n = len(trajectory)
    if n < 4:
        return trajectory
    
    if joint_2d_trajectory is None or len(joint_2d_trajectory) != n:
        log.warning("Joint 2D trajectory mismatch, falling back to Savitzky-Golay")
        return smooth_trajectory_3d(trajectory, method="savgol", strength=strength)
    
    smoothed = trajectory.copy()
    
    # Step 1: Smooth the 2D joint trajectory (this is stable)
    joint_2d_x = joint_2d_trajectory[:, 0]
    joint_2d_y = joint_2d_trajectory[:, 1]
    
    # Apply savgol to 2D trajectory
    window = int(5 + strength * 26)
    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 == 1 else n - 1)
    
    joint_2d_x_smooth = savgol_smooth(joint_2d_x, window=window, polyorder=3)
    joint_2d_y_smooth = savgol_smooth(joint_2d_y, window=window, polyorder=3)
    
    # Step 2: Compute how much the 2D position changed after smoothing
    # This tells us the "correction" we need to apply
    delta_2d_x = joint_2d_x_smooth - joint_2d_x
    delta_2d_y = joint_2d_y_smooth - joint_2d_y
    
    # Step 3: Estimate depth (Z) from trajectory
    # Z is typically the most stable component
    z_coords = trajectory[:, 2]
    z_smooth = savgol_smooth(z_coords, window=window, polyorder=3)
    
    # Step 4: Convert 2D correction to 3D correction
    # In the projection model: screen_x = (world_x / world_z) * focal_length
    # So: world_x = screen_x * world_z / focal_length
    # The correction in world space is proportional to the 2D correction and depth
    
    # Estimate focal length from existing data (or use default)
    # We use ratio of screen position to world position as a proxy
    eps = 1e-6
    scale_factor = np.abs(z_smooth).mean() / 500.0  # Approximate scale
    
    # Apply corrections
    smoothed[:, 0] = trajectory[:, 0] + delta_2d_x * scale_factor * strength
    smoothed[:, 1] = trajectory[:, 1] + delta_2d_y * scale_factor * strength
    smoothed[:, 2] = z_smooth
    
    # Blend between original and corrected based on strength
    blend = strength
    smoothed = trajectory * (1 - blend) + smoothed * blend
    
    # Also apply savgol for additional smoothing
    for axis in range(3):
        smoothed[:, axis] = savgol_smooth(smoothed[:, axis], window=window, polyorder=3)
    
    # Preserve endpoints
    if preserve_endpoints:
        smoothed[0] = trajectory[0]
        smoothed[-1] = trajectory[-1]
    
    return smoothed


def smooth_trajectory_3d(
    trajectory: np.ndarray,
    method: str = "savgol",
    strength: float = 0.5,
    preserve_endpoints: bool = True,
    joint_2d_trajectory: np.ndarray = None
) -> np.ndarray:
    """
    Smooth a 3D trajectory [N, 3].
    
    Args:
        trajectory: [N, 3] array of XYZ positions
        method: Smoothing method (savgol, gaussian, moving_avg, spline, kalman, joint_guided)
        strength: Smoothing strength 0-1 (0 = minimal, 1 = maximum)
        preserve_endpoints: If True, keep first/last points fixed
        joint_2d_trajectory: [N, 2] 2D joint positions for joint-guided smoothing
    
    Returns:
        Smoothed trajectory [N, 3]
    """
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        log.warning(f"Invalid trajectory shape: {trajectory.shape}, expected [N, 3]")
        return trajectory
    
    n = len(trajectory)
    if n < 4:
        return trajectory
    
    # Handle joint-guided method separately
    if method == "joint_guided":
        return joint_guided_smooth(trajectory, joint_2d_trajectory, strength, preserve_endpoints)
    
    smoothed = np.zeros_like(trajectory)
    
    # Map strength (0-1) to method-specific parameters
    for axis in range(3):
        signal = trajectory[:, axis]
        
        if method == "savgol":
            # Window size: 5 (minimal) to 31 (maximum)
            window = int(5 + strength * 26)
            if window % 2 == 0:
                window += 1
            window = min(window, n if n % 2 == 1 else n - 1)
            polyorder = min(3, window - 2)
            smoothed[:, axis] = savgol_smooth(signal, window=window, polyorder=polyorder)
            
        elif method == "gaussian":
            # Sigma: 0.5 (minimal) to 5.0 (maximum)
            sigma = 0.5 + strength * 4.5
            smoothed[:, axis] = gaussian_smooth(signal, sigma=sigma)
            
        elif method == "moving_avg":
            # Window: 3 (minimal) to 21 (maximum)
            window = int(3 + strength * 18)
            if window % 2 == 0:
                window += 1
            smoothed[:, axis] = moving_average(signal, window=window)
            
        elif method == "spline":
            # Smoothing factor: 0.1 (minimal) to 0.9 (maximum)
            smoothing_factor = 0.1 + strength * 0.8
            smoothed[:, axis] = spline_smooth(signal, smoothing_factor=smoothing_factor)
            
        elif method == "kalman":
            # Process noise: 1.0 (minimal smoothing) to 0.01 (maximum smoothing)
            process_noise = 1.0 - strength * 0.99
            measurement_noise = 0.5 + strength * 4.5
            smoothed[:, axis] = kalman_smooth(signal, process_noise=process_noise, 
                                               measurement_noise=measurement_noise)
        else:
            log.warning(f"Unknown method '{method}', using savgol")
            smoothed[:, axis] = savgol_smooth(signal, window=11)
    
    # Preserve endpoints if requested
    if preserve_endpoints:
        smoothed[0] = trajectory[0]
        smoothed[-1] = trajectory[-1]
    
    return smoothed


# ============================================================================
# ComfyUI Node
# ============================================================================

class TrajectorySmoother:
    """
    Smooth trajectory data from Motion Analyzer to reduce noise/jitter.
    
    This node processes SUBJECT_MOTION and outputs smoothed trajectory data.
    The smoothed data is written back to body_world_3d and related fields.
    
    v4.8.8: Added Joint-Guided smoothing and reference joint selection.
    """
    
    CATEGORY = "SAM3DBody2abc/Analysis"
    FUNCTION = "smooth_trajectory"
    
    @classmethod
    def INPUT_TYPES(cls):
        methods = [
            "Savitzky-Golay (Best)", 
            "Gaussian", 
            "Moving Average", 
            "Spline", 
            "Kalman",
            "Joint-Guided (NEW)"  # NEW in v4.8.8
        ]
        
        # Build joint selection lists
        coco_joints = [f"{i}: {name}" for i, name in enumerate(COCO_JOINT_NAMES)]
        smplh_joints = [f"{i}: {name}" for i, name in enumerate(SMPLH_JOINT_NAMES)]
        
        return {
            "required": {
                "subject_motion": ("SUBJECT_MOTION", {
                    "tooltip": "Subject motion data from Motion Analyzer"
                }),
                "method": (methods, {
                    "default": "Savitzky-Golay (Best)",
                    "tooltip": "Smoothing method. Joint-Guided uses stable joint detection to guide trajectory."
                }),
                "strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Smoothing strength. 0 = minimal, 1 = maximum smoothing."
                }),
            },
            "optional": {
                "mesh_sequence": ("MESH_SEQUENCE", {
                    "tooltip": "Mesh sequence for Joint-Guided smoothing (provides joint_coords/keypoints)"
                }),
                "skeleton_format": (["SMPL-H (joint_coords)", "COCO (keypoints)"], {
                    "default": "SMPL-H (joint_coords)",
                    "tooltip": "Which skeleton format to use for reference joint"
                }),
                "reference_joint_smplh": (smplh_joints, {
                    "default": "0: Pelvis",
                    "tooltip": "Reference joint for SMPL-H format (joint_coords)"
                }),
                "reference_joint_coco": (coco_joints, {
                    "default": "11: L_Hip",
                    "tooltip": "Reference joint for COCO format (keypoints)"
                }),
                "smooth_xy": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Smooth X and Y (lateral and vertical) motion"
                }),
                "smooth_z": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Smooth Z (depth) motion"
                }),
                "preserve_endpoints": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep first and last positions unchanged"
                }),
                "highlight_joint": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Highlight the reference joint in Motion Analyzer output (green)"
                }),
            }
        }
    
    RETURN_TYPES = ("SUBJECT_MOTION", "IMAGE", "STRING", "INT")
    RETURN_NAMES = ("subject_motion", "comparison_image", "stats", "reference_joint_index")
    
    def smooth_trajectory(
        self,
        subject_motion: Dict,
        method: str = "Savitzky-Golay (Best)",
        strength: float = 0.5,
        mesh_sequence: List[Dict] = None,
        skeleton_format: str = "SMPL-H (joint_coords)",
        reference_joint_smplh: str = "0: Pelvis",
        reference_joint_coco: str = "11: L_Hip",
        smooth_xy: bool = True,
        smooth_z: bool = True,
        preserve_endpoints: bool = True,
        highlight_joint: bool = True
    ) -> Tuple[Dict, any, str, int]:
        """Smooth the trajectory in subject_motion."""
        
        import torch
        import cv2
        
        log.info(f"=" * 60)
        log.info(f"Trajectory Smoother v4.8.8")
        log.info(f"=" * 60)
        log.info(f"Method: {method}")
        log.info(f"Strength: {strength:.2f}")
        log.info(f"Smooth XY: {smooth_xy}, Smooth Z: {smooth_z}")
        
        # Parse reference joint index
        if skeleton_format == "SMPL-H (joint_coords)":
            ref_joint_idx = int(reference_joint_smplh.split(":")[0])
            ref_joint_name = SMPLH_JOINT_NAMES[ref_joint_idx]
            skeleton_key = "joint_coords"
        else:
            ref_joint_idx = int(reference_joint_coco.split(":")[0])
            ref_joint_name = COCO_JOINT_NAMES[ref_joint_idx]
            skeleton_key = "keypoints_2d"
        
        log.info(f"Reference joint: {ref_joint_idx} ({ref_joint_name})")
        log.info(f"Skeleton format: {skeleton_format}")
        
        # Deep copy to avoid modifying original
        result = copy.deepcopy(subject_motion)
        
        # Get trajectory data
        body_world_3d = result.get("body_world_3d", [])
        body_world_3d_raw = result.get("body_world_3d_raw", body_world_3d.copy() if body_world_3d else [])
        body_world_3d_compensated = result.get("body_world_3d_compensated", [])
        
        if not body_world_3d or len(body_world_3d) < 4:
            log.warning("Not enough trajectory data to smooth (need at least 4 frames)")
            stats = "Not enough data to smooth"
            # Return empty comparison image
            empty_img = np.zeros((256, 512, 3), dtype=np.float32)
            cv2.putText(empty_img, "Not enough data", (150, 128), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0.5, 0.5, 0.5), 2)
            return (result, torch.from_numpy(empty_img).unsqueeze(0), stats, ref_joint_idx)
        
        # Convert to numpy
        trajectory = np.array(body_world_3d)
        original_trajectory = trajectory.copy()
        
        log.info(f"Trajectory frames: {len(trajectory)}")
        
        # Extract 2D joint trajectory for Joint-Guided smoothing
        joint_2d_trajectory = None
        if "Joint-Guided" in method and mesh_sequence is not None:
            # Extract frames from mesh_sequence dict
            frames_data = mesh_sequence.get("frames", mesh_sequence)
            if isinstance(frames_data, dict):
                # Dict with keys (can be int or string like "0", "1", etc.)
                def sort_key(x):
                    if isinstance(x, int):
                        return x
                    elif isinstance(x, str) and x.isdigit():
                        return int(x)
                    else:
                        return x
                frame_keys = sorted(frames_data.keys(), key=sort_key)
                frames_list = [frames_data[k] for k in frame_keys]
            elif isinstance(frames_data, list):
                frames_list = frames_data
            else:
                frames_list = []
            
            if frames_list:
                joint_2d_trajectory = self._extract_joint_2d_trajectory(
                    frames_list, ref_joint_idx, skeleton_key
                )
                if joint_2d_trajectory is not None:
                    log.info(f"Extracted 2D trajectory for joint {ref_joint_idx}: {len(joint_2d_trajectory)} frames")
                else:
                    log.warning(f"Could not extract 2D trajectory, falling back to Savitzky-Golay")
                    method = "Savitzky-Golay (Best)"
            else:
                log.warning("No frames found in mesh_sequence, falling back to Savitzky-Golay")
                method = "Savitzky-Golay (Best)"
        
        # Map method name to internal name
        method_map = {
            "Savitzky-Golay (Best)": "savgol",
            "Gaussian": "gaussian",
            "Moving Average": "moving_avg",
            "Spline": "spline",
            "Kalman": "kalman",
            "Joint-Guided (NEW)": "joint_guided"
        }
        method_internal = method_map.get(method, "savgol")
        
        # Apply smoothing
        smoothed = smooth_trajectory_3d(
            trajectory,
            method=method_internal,
            strength=strength,
            preserve_endpoints=preserve_endpoints,
            joint_2d_trajectory=joint_2d_trajectory
        )
        
        # Selectively apply based on smooth_xy and smooth_z
        if not smooth_xy:
            smoothed[:, 0] = trajectory[:, 0]  # Keep original X
            smoothed[:, 1] = trajectory[:, 1]  # Keep original Y
        if not smooth_z:
            smoothed[:, 2] = trajectory[:, 2]  # Keep original Z
        
        # Compute statistics
        diff = smoothed - original_trajectory
        stats_dict = {
            "x_change_mean": np.abs(diff[:, 0]).mean(),
            "y_change_mean": np.abs(diff[:, 1]).mean(),
            "z_change_mean": np.abs(diff[:, 2]).mean(),
            "x_change_max": np.abs(diff[:, 0]).max(),
            "y_change_max": np.abs(diff[:, 1]).max(),
            "z_change_max": np.abs(diff[:, 2]).max(),
        }
        
        # Compute jitter reduction
        def compute_jitter(traj):
            vel = np.diff(traj, axis=0)
            accel = np.diff(vel, axis=0)
            return np.mean(np.linalg.norm(accel, axis=1))
        
        jitter_orig = compute_jitter(original_trajectory)
        jitter_smooth = compute_jitter(smoothed)
        jitter_reduction = (1 - jitter_smooth / (jitter_orig + 1e-8)) * 100
        jitter_reduction = max(0, min(100, jitter_reduction))
        stats_dict["jitter_reduction_pct"] = jitter_reduction
        
        # Update result
        result["body_world_3d"] = smoothed.tolist()
        result["body_world_3d_raw"] = original_trajectory.tolist()  # Keep original
        
        log.info(f"Jitter reduction: {jitter_reduction:.1f}%")
        
        # Also smooth compensated trajectory if available
        if body_world_3d_compensated and len(body_world_3d_compensated) >= 4:
            traj_comp = np.array(body_world_3d_compensated)
            smoothed_comp = smooth_trajectory_3d(
                traj_comp,
                method=method_internal,
                strength=strength,
                preserve_endpoints=preserve_endpoints,
                joint_2d_trajectory=joint_2d_trajectory
            )
            if not smooth_xy:
                smoothed_comp[:, 0] = traj_comp[:, 0]
                smoothed_comp[:, 1] = traj_comp[:, 1]
            if not smooth_z:
                smoothed_comp[:, 2] = traj_comp[:, 2]
            result["body_world_3d_compensated"] = smoothed_comp.tolist()
        
        # Also smooth depth_estimate if available
        depth_estimate = result.get("depth_estimate", [])
        if depth_estimate and len(depth_estimate) >= 4 and smooth_z:
            depth_arr = np.array(depth_estimate)
            if method_internal == "savgol":
                window = int(5 + strength * 26)
                if window % 2 == 0:
                    window += 1
                window = min(window, len(depth_arr) if len(depth_arr) % 2 == 1 else len(depth_arr) - 1)
                smoothed_depth = savgol_smooth(depth_arr, window=window, polyorder=3)
            else:
                smoothed_depth = gaussian_smooth(depth_arr, sigma=1 + strength * 3)
            result["depth_estimate"] = smoothed_depth.tolist()
        
        # Add smoothing metadata
        result["smoothing_applied"] = {
            "method": method,
            "strength": strength,
            "smooth_xy": smooth_xy,
            "smooth_z": smooth_z,
            "jitter_reduction_pct": jitter_reduction,
            "reference_joint_idx": ref_joint_idx,
            "reference_joint_name": ref_joint_name,
            "skeleton_format": skeleton_format
        }
        
        # Add reference joint index for Motion Analyzer to highlight
        if highlight_joint:
            result["highlight_joint_idx"] = ref_joint_idx
            result["highlight_joint_format"] = skeleton_format
        
        # Create comparison visualization
        comparison_img = self._create_comparison_image(
            original_trajectory, smoothed, 
            stats_dict, method, strength, ref_joint_name
        )
        
        # Format stats string
        stats_str = f"""Trajectory Smoothing Results (v4.8.8)
=====================================
Method: {method}
Strength: {strength:.2f}
Reference Joint: {ref_joint_idx} ({ref_joint_name})
Frames: {len(trajectory)}

Jitter Reduction: {jitter_reduction:.1f}%
  Before: {jitter_orig:.4f}
  After: {jitter_smooth:.4f}

Mean Position Change:
  X: {stats_dict['x_change_mean']:.4f}m
  Y: {stats_dict['y_change_mean']:.4f}m
  Z: {stats_dict['z_change_mean']:.4f}m

Max Position Change:
  X: {stats_dict['x_change_max']:.4f}m
  Y: {stats_dict['y_change_max']:.4f}m
  Z: {stats_dict['z_change_max']:.4f}m
"""
        
        log.info(f"Smoothing complete")
        
        return (result, comparison_img, stats_str, ref_joint_idx)
    
    def _extract_joint_2d_trajectory(
        self, 
        mesh_sequence: List[Dict], 
        joint_idx: int,
        skeleton_key: str
    ) -> Optional[np.ndarray]:
        """Extract 2D trajectory for a specific joint from mesh sequence."""
        
        trajectory_2d = []
        
        for frame in mesh_sequence:
            if not frame.get("valid", True):
                # Use last known position or skip
                if trajectory_2d:
                    trajectory_2d.append(trajectory_2d[-1])
                continue
            
            # Try different keys for 2D joint positions
            joints_2d = None
            for key in [skeleton_key, "joints_2d", "keypoints_2d", "pred_keypoints_2d"]:
                if key in frame and frame[key] is not None:
                    joints_2d = np.array(frame[key])
                    break
            
            if joints_2d is None:
                if trajectory_2d:
                    trajectory_2d.append(trajectory_2d[-1])
                continue
            
            # Handle different array shapes
            if joints_2d.ndim == 1:
                # Flat array [x0, y0, x1, y1, ...]
                if joint_idx * 2 + 1 < len(joints_2d):
                    x = joints_2d[joint_idx * 2]
                    y = joints_2d[joint_idx * 2 + 1]
                    trajectory_2d.append([x, y])
                elif trajectory_2d:
                    trajectory_2d.append(trajectory_2d[-1])
            elif joints_2d.ndim == 2:
                # [J, 2] or [J, 3]
                if joint_idx < len(joints_2d):
                    trajectory_2d.append(joints_2d[joint_idx, :2])
                elif trajectory_2d:
                    trajectory_2d.append(trajectory_2d[-1])
            else:
                if trajectory_2d:
                    trajectory_2d.append(trajectory_2d[-1])
        
        if len(trajectory_2d) < 4:
            return None
        
        return np.array(trajectory_2d)
    
    def _create_comparison_image(
        self,
        original: np.ndarray,
        smoothed: np.ndarray,
        stats: Dict,
        method: str,
        strength: float,
        ref_joint_name: str = ""
    ) -> any:
        """Create a side-by-side comparison visualization."""
        import torch
        import cv2
        
        h, w = 400, 800
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)
        
        # Split into two panels
        panel_w = w // 2
        
        # Draw both trajectories (top-down view: X vs Z)
        # Negate X for correct left-right orientation (same as motion_analyzer)
        x_orig = -original[:, 0]
        z_orig = original[:, 2]
        x_smooth = -smoothed[:, 0]
        z_smooth = smoothed[:, 2]
        
        # Compute bounds (use same scale for both)
        all_x = np.concatenate([x_orig, x_smooth])
        all_z = np.concatenate([z_orig, z_smooth])
        
        x_min, x_max = all_x.min(), all_x.max()
        z_min, z_max = all_z.min(), all_z.max()
        
        x_range = max(x_max - x_min, 0.5)
        z_range = max(z_max - z_min, 0.5)
        max_range = max(x_range, z_range)
        
        x_center = (x_min + x_max) / 2
        z_center = (z_min + z_max) / 2
        
        padding = 0.15
        usable_h = h * (1 - 2 * padding)
        usable_w = (panel_w - 20) * (1 - 2 * padding)
        scale = min(usable_h, usable_w) / max_range
        
        def world_to_panel(x, z, panel_offset):
            px = int(panel_offset + panel_w / 2 + (x - x_center) * scale)
            py = int(h * padding + (z_max - z) * scale)
            return px, py
        
        # Draw original (left panel) - RED
        for i in range(len(x_orig) - 1):
            pt1 = world_to_panel(x_orig[i], z_orig[i], 0)
            pt2 = world_to_panel(x_orig[i + 1], z_orig[i + 1], 0)
            cv2.line(canvas, pt1, pt2, (100, 100, 255), 2)  # Red-ish
        
        # Draw smoothed (right panel) - GREEN
        for i in range(len(x_smooth) - 1):
            pt1 = world_to_panel(x_smooth[i], z_smooth[i], panel_w)
            pt2 = world_to_panel(x_smooth[i + 1], z_smooth[i + 1], panel_w)
            cv2.line(canvas, pt1, pt2, (100, 255, 100), 2)  # Green
        
        # Draw smoothed on left panel for comparison (thin green line)
        for i in range(len(x_smooth) - 1):
            pt1 = world_to_panel(x_smooth[i], z_smooth[i], 0)
            pt2 = world_to_panel(x_smooth[i + 1], z_smooth[i + 1], 0)
            cv2.line(canvas, pt1, pt2, (50, 150, 50), 1)
        
        # Draw original on right panel for comparison (thin red line)
        for i in range(len(x_orig) - 1):
            pt1 = world_to_panel(x_orig[i], z_orig[i], panel_w)
            pt2 = world_to_panel(x_orig[i + 1], z_orig[i + 1], panel_w)
            cv2.line(canvas, pt1, pt2, (50, 50, 150), 1)
        
        # Draw dividing line
        cv2.line(canvas, (panel_w, 0), (panel_w, h), (80, 80, 80), 2)
        
        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "ORIGINAL", (panel_w // 2 - 40, 25), font, 0.6, (100, 100, 255), 2)
        cv2.putText(canvas, "SMOOTHED", (panel_w + panel_w // 2 - 45, 25), font, 0.6, (100, 255, 100), 2)
        
        # Stats
        jitter_pct = stats.get("jitter_reduction_pct", 0)
        cv2.putText(canvas, f"Method: {method}", (10, h - 70), font, 0.4, (180, 180, 180), 1)
        cv2.putText(canvas, f"Strength: {strength:.2f}", (10, h - 50), font, 0.4, (180, 180, 180), 1)
        if ref_joint_name:
            cv2.putText(canvas, f"Ref Joint: {ref_joint_name}", (10, h - 30), font, 0.4, (100, 255, 100), 1)
        cv2.putText(canvas, f"Jitter Reduction: {jitter_pct:.1f}%", (10, h - 10), font, 0.4, (100, 255, 100), 1)
        
        # Legend on right panel
        cv2.putText(canvas, "Green = Smoothed", (panel_w + 10, h - 40), font, 0.35, (100, 255, 100), 1)
        cv2.putText(canvas, "Red = Original", (panel_w + 10, h - 20), font, 0.35, (100, 100, 255), 1)
        
        # Convert to float tensor
        canvas_float = canvas.astype(np.float32) / 255.0
        return torch.from_numpy(canvas_float).unsqueeze(0)


# Export
NODE_CLASS_MAPPINGS = {
    "TrajectorySmoother": TrajectorySmoother,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrajectorySmoother": "Trajectory Smoother",
}
