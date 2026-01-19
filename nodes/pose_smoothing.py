"""
SMPL Pose Smoothing for SAM3DBody2abc
=====================================

Smooths joint rotations temporally to eliminate jitter while preserving motion.

Key Features:
- Quaternion-based smoothing (avoids gimbal lock)
- Multiple methods: Gaussian, Savitzky-Golay, Kalman filter
- Quaternion continuity enforcement (prevents sign flips)
- Separate smoothing for body, hands, face
- Preserves motion dynamics while reducing noise

Usage:
    from pose_smoothing import PoseSmoother, smooth_mesh_sequence
    
    # Quick usage
    smoothed_sequence = smooth_mesh_sequence(mesh_sequence, method="gaussian")
    
    # Advanced usage
    smoother = PoseSmoother(method="kalman", window=7)
    smoothed_rotations = smoother.smooth(joint_rotations)

Author: SAM3DBody2abc
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Literal
from scipy.spatial.transform import Rotation, Slerp
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from dataclasses import dataclass
from enum import Enum


class SmoothingMethod(Enum):
    """Available smoothing methods."""
    GAUSSIAN = "gaussian"
    SAVGOL = "savgol"
    KALMAN = "kalman"
    SLERP = "slerp"
    MOVING_AVERAGE = "moving_average"


@dataclass
class SmoothingConfig:
    """Configuration for pose smoothing."""
    method: SmoothingMethod = SmoothingMethod.GAUSSIAN
    window_size: int = 5          # Frames for smoothing window
    sigma: float = 1.5            # Gaussian sigma
    poly_order: int = 2           # Savitzky-Golay polynomial order
    process_noise: float = 0.01   # Kalman process noise
    measurement_noise: float = 0.1  # Kalman measurement noise
    preserve_endpoints: bool = True  # Don't smooth first/last frames
    smooth_root: bool = True      # Smooth root rotation
    smooth_body: bool = True      # Smooth body joints
    smooth_hands: bool = False    # Smooth hand joints (often noisy)
    smooth_face: bool = False     # Smooth face joints


class PoseSmoother:
    """
    Temporal smoothing for SMPL/MHR pose parameters.
    
    Works on rotation matrices (3x3) or quaternions, handles the
    complexities of rotation space smoothing.
    """
    
    def __init__(self, config: Optional[SmoothingConfig] = None):
        """
        Initialize pose smoother.
        
        Args:
            config: Smoothing configuration. Uses defaults if not provided.
        """
        self.config = config or SmoothingConfig()
    
    def smooth(
        self,
        rotations: np.ndarray,
        method: Optional[str] = None
    ) -> np.ndarray:
        """
        Smooth joint rotations temporally.
        
        Args:
            rotations: [T, J, 3, 3] rotation matrices or [T, J, 4] quaternions
            method: Override smoothing method
        
        Returns:
            Smoothed rotations in same format as input
        """
        # Detect input format
        if rotations.ndim == 4 and rotations.shape[-2:] == (3, 3):
            # Rotation matrices
            return self._smooth_matrices(rotations, method)
        elif rotations.ndim == 3 and rotations.shape[-1] == 4:
            # Quaternions
            return self._smooth_quaternions(rotations, method)
        else:
            raise ValueError(f"Unexpected rotation shape: {rotations.shape}")
    
    def _smooth_matrices(
        self,
        rotations: np.ndarray,
        method: Optional[str] = None
    ) -> np.ndarray:
        """Smooth rotation matrices by converting to quaternions."""
        T, J = rotations.shape[:2]
        
        # Convert to quaternions
        quats = np.zeros((T, J, 4))
        for t in range(T):
            for j in range(J):
                quats[t, j] = Rotation.from_matrix(rotations[t, j]).as_quat()
        
        # Smooth quaternions
        smoothed_quats = self._smooth_quaternions(quats, method)
        
        # Convert back to matrices
        smoothed = np.zeros_like(rotations)
        for t in range(T):
            for j in range(J):
                smoothed[t, j] = Rotation.from_quat(smoothed_quats[t, j]).as_matrix()
        
        return smoothed
    
    def _smooth_quaternions(
        self,
        quats: np.ndarray,
        method: Optional[str] = None
    ) -> np.ndarray:
        """
        Smooth quaternion sequences.
        
        Args:
            quats: [T, J, 4] quaternions (x, y, z, w format from scipy)
            method: Override smoothing method
        
        Returns:
            Smoothed quaternions [T, J, 4]
        """
        T, J = quats.shape[:2]
        method = SmoothingMethod(method) if method else self.config.method
        
        # Ensure quaternion continuity first
        quats = self._enforce_quaternion_continuity(quats)
        
        smoothed = np.zeros_like(quats)
        
        for j in range(J):
            joint_quats = quats[:, j, :]
            
            if method == SmoothingMethod.GAUSSIAN:
                smoothed[:, j] = self._gaussian_smooth_quats(joint_quats)
            elif method == SmoothingMethod.SAVGOL:
                smoothed[:, j] = self._savgol_smooth_quats(joint_quats)
            elif method == SmoothingMethod.KALMAN:
                smoothed[:, j] = self._kalman_smooth_quats(joint_quats)
            elif method == SmoothingMethod.SLERP:
                smoothed[:, j] = self._slerp_smooth_quats(joint_quats)
            elif method == SmoothingMethod.MOVING_AVERAGE:
                smoothed[:, j] = self._moving_average_smooth_quats(joint_quats)
            else:
                smoothed[:, j] = joint_quats
        
        # Renormalize all quaternions
        norms = np.linalg.norm(smoothed, axis=-1, keepdims=True)
        smoothed = smoothed / np.maximum(norms, 1e-8)
        
        # Preserve endpoints if requested
        if self.config.preserve_endpoints:
            smoothed[0] = quats[0]
            smoothed[-1] = quats[-1]
        
        return smoothed
    
    def _enforce_quaternion_continuity(self, quats: np.ndarray) -> np.ndarray:
        """
        Ensure quaternions don't flip signs between frames.
        
        Quaternion q and -q represent the same rotation, but interpolating
        between them causes issues. This fixes sign flips.
        """
        T, J = quats.shape[:2]
        result = quats.copy()
        
        for j in range(J):
            for t in range(1, T):
                # If dot product is negative, flip the sign
                if np.dot(result[t, j], result[t-1, j]) < 0:
                    result[t, j] = -result[t, j]
        
        return result
    
    def _gaussian_smooth_quats(self, quats: np.ndarray) -> np.ndarray:
        """Gaussian smoothing on quaternion components."""
        sigma = self.config.sigma
        smoothed = np.zeros_like(quats)
        
        for i in range(4):
            smoothed[:, i] = gaussian_filter1d(quats[:, i], sigma=sigma)
        
        return smoothed
    
    def _savgol_smooth_quats(self, quats: np.ndarray) -> np.ndarray:
        """Savitzky-Golay filter smoothing."""
        window = self.config.window_size
        poly_order = self.config.poly_order
        
        # Window must be odd
        if window % 2 == 0:
            window += 1
        
        # Window must be > poly_order
        if window <= poly_order:
            window = poly_order + 2
            if window % 2 == 0:
                window += 1
        
        # Need enough samples
        if len(quats) < window:
            return quats
        
        smoothed = np.zeros_like(quats)
        for i in range(4):
            smoothed[:, i] = savgol_filter(quats[:, i], window, poly_order)
        
        return smoothed
    
    def _kalman_smooth_quats(self, quats: np.ndarray) -> np.ndarray:
        """
        Kalman filter smoothing with RTS (Rauch-Tung-Striebel) smoother.
        
        Uses a constant-velocity model in quaternion space.
        """
        T = len(quats)
        if T < 3:
            return quats
        
        # State: [q0, q1, q2, q3, dq0, dq1, dq2, dq3] (quaternion + velocity)
        n_state = 8
        n_obs = 4
        
        # Transition matrix (constant velocity)
        dt = 1.0  # Assume unit time steps
        F = np.eye(n_state)
        F[:4, 4:] = np.eye(4) * dt
        
        # Observation matrix (we observe quaternion, not velocity)
        H = np.zeros((n_obs, n_state))
        H[:4, :4] = np.eye(4)
        
        # Process noise
        Q = np.eye(n_state) * self.config.process_noise
        Q[4:, 4:] *= 10  # Higher noise on velocity
        
        # Measurement noise
        R = np.eye(n_obs) * self.config.measurement_noise
        
        # Initialize
        x = np.zeros(n_state)
        x[:4] = quats[0]
        P = np.eye(n_state) * 0.1
        
        # Forward pass (Kalman filter)
        x_forward = np.zeros((T, n_state))
        P_forward = np.zeros((T, n_state, n_state))
        
        for t in range(T):
            # Predict
            if t > 0:
                x = F @ x
                P = F @ P @ F.T + Q
            
            # Update
            y = quats[t] - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(n_state) - K @ H) @ P
            
            x_forward[t] = x
            P_forward[t] = P
        
        # Backward pass (RTS smoother)
        x_smooth = np.zeros((T, n_state))
        x_smooth[-1] = x_forward[-1]
        
        for t in range(T - 2, -1, -1):
            P_pred = F @ P_forward[t] @ F.T + Q
            G = P_forward[t] @ F.T @ np.linalg.inv(P_pred)
            x_smooth[t] = x_forward[t] + G @ (x_smooth[t+1] - F @ x_forward[t])
        
        return x_smooth[:, :4]
    
    def _slerp_smooth_quats(self, quats: np.ndarray) -> np.ndarray:
        """
        Spherical linear interpolation (SLERP) based smoothing.
        
        Properly interpolates on the rotation manifold.
        """
        T = len(quats)
        window = self.config.window_size
        half_window = window // 2
        
        smoothed = np.zeros_like(quats)
        
        for t in range(T):
            # Gather window of quaternions
            start = max(0, t - half_window)
            end = min(T, t + half_window + 1)
            
            # Average quaternions using iterative SLERP
            avg_quat = quats[start].copy()
            count = 1
            
            for i in range(start + 1, end):
                # Incrementally blend in each quaternion
                weight = 1.0 / (count + 1)
                
                # SLERP between current average and new quaternion
                r1 = Rotation.from_quat(avg_quat)
                r2 = Rotation.from_quat(quats[i])
                
                times = [0, 1]
                rotations = Rotation.concatenate([r1, r2])
                slerp = Slerp(times, rotations)
                avg_quat = slerp([weight]).as_quat()[0]
                
                count += 1
            
            smoothed[t] = avg_quat
        
        return smoothed
    
    def _moving_average_smooth_quats(self, quats: np.ndarray) -> np.ndarray:
        """Simple moving average (in quaternion component space)."""
        T = len(quats)
        window = self.config.window_size
        half_window = window // 2
        
        smoothed = np.zeros_like(quats)
        
        for t in range(T):
            start = max(0, t - half_window)
            end = min(T, t + half_window + 1)
            smoothed[t] = np.mean(quats[start:end], axis=0)
        
        return smoothed


def smooth_mesh_sequence(
    mesh_sequence: Dict,
    method: str = "gaussian",
    window: int = 5,
    sigma: float = 1.5,
    smooth_trajectory: bool = True,
    smooth_rotations: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Smooth an entire mesh sequence from SAM3DBody.
    
    Args:
        mesh_sequence: MESH_SEQUENCE dict with frames containing joint_rotations
        method: Smoothing method ("gaussian", "savgol", "kalman", "slerp")
        window: Window size for smoothing
        sigma: Gaussian sigma (for gaussian method)
        smooth_trajectory: Also smooth pred_cam_t (translation)
        smooth_rotations: Smooth joint_rotations
        verbose: Print progress
    
    Returns:
        New mesh_sequence with smoothed data
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
        if verbose:
            print("[Pose Smoothing] No frames to smooth")
        return mesh_sequence
    
    # Extract data
    num_frames = len(frames)
    
    # Get rotation data
    first_rots = frames[0].get("joint_rotations")
    has_rotations = first_rots is not None and len(first_rots) > 0
    
    if has_rotations and smooth_rotations:
        if verbose:
            print(f"[Pose Smoothing] Smoothing {num_frames} frames with {method} method")
        
        # Stack rotations [T, J, 3, 3]
        all_rotations = []
        for frame in frames:
            rots = frame.get("joint_rotations")
            if rots is not None:
                all_rotations.append(np.array(rots))
            else:
                # Use previous frame's rotations if missing
                all_rotations.append(all_rotations[-1] if all_rotations else np.zeros((127, 3, 3)))
        
        rotations = np.array(all_rotations)
        
        # Configure smoother
        config = SmoothingConfig(
            method=SmoothingMethod(method),
            window_size=window,
            sigma=sigma,
        )
        smoother = PoseSmoother(config)
        
        # Smooth
        smoothed_rotations = smoother.smooth(rotations)
        
        # Update frames
        for i, frame in enumerate(frames):
            frame["joint_rotations"] = smoothed_rotations[i].tolist()
        
        if verbose:
            # Report jitter reduction
            original_jitter = compute_rotation_jitter(rotations)
            smoothed_jitter = compute_rotation_jitter(smoothed_rotations)
            reduction = (1 - smoothed_jitter / max(original_jitter, 1e-8)) * 100
            print(f"[Pose Smoothing] Rotation jitter reduced by {reduction:.1f}%")
    
    # Smooth trajectory (pred_cam_t)
    if smooth_trajectory:
        all_cam_t = []
        for frame in frames:
            cam_t = frame.get("pred_cam_t", frame.get("camera", [0, 0, 2]))
            if cam_t is not None:
                all_cam_t.append(np.array(cam_t).flatten()[:3])
            else:
                all_cam_t.append(all_cam_t[-1] if all_cam_t else np.array([0, 0, 2]))
        
        cam_t = np.array(all_cam_t)
        
        # Smooth each component
        if method == "gaussian":
            smoothed_cam_t = gaussian_filter1d(cam_t, sigma=sigma, axis=0)
        elif method == "savgol":
            win = window if window % 2 == 1 else window + 1
            if len(cam_t) >= win:
                smoothed_cam_t = savgol_filter(cam_t, win, min(2, win-1), axis=0)
            else:
                smoothed_cam_t = cam_t
        else:
            smoothed_cam_t = gaussian_filter1d(cam_t, sigma=sigma, axis=0)
        
        # Update frames
        for i, frame in enumerate(frames):
            frame["pred_cam_t"] = smoothed_cam_t[i].tolist()
        
        if verbose:
            original_jitter = np.mean(np.abs(np.diff(cam_t, axis=0)))
            smoothed_jitter = np.mean(np.abs(np.diff(smoothed_cam_t, axis=0)))
            reduction = (1 - smoothed_jitter / max(original_jitter, 1e-8)) * 100
            print(f"[Pose Smoothing] Trajectory jitter reduced by {reduction:.1f}%")
    
    # Create new mesh_sequence with smoothed data
    result = mesh_sequence.copy()
    
    # Preserve original format (dict or list)
    if frames_is_dict:
        result["frames"] = dict(zip(frame_keys, frames))
    else:
        result["frames"] = frames
    
    result["smoothing_applied"] = {
        "method": method,
        "window": window,
        "sigma": sigma,
        "smooth_trajectory": smooth_trajectory,
        "smooth_rotations": smooth_rotations,
    }
    
    return result


def compute_rotation_jitter(rotations: np.ndarray) -> float:
    """
    Compute average rotation jitter across frames.
    
    Args:
        rotations: [T, J, 3, 3] rotation matrices
    
    Returns:
        Average angular velocity (radians/frame)
    """
    T, J = rotations.shape[:2]
    
    total_jitter = 0.0
    count = 0
    
    for t in range(1, T):
        for j in range(J):
            # Relative rotation
            R_rel = rotations[t, j] @ rotations[t-1, j].T
            
            # Angle of rotation
            trace = np.trace(R_rel)
            angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            
            total_jitter += angle
            count += 1
    
    return total_jitter / max(count, 1)


# =============================================================================
# ComfyUI Node
# =============================================================================

class PoseSmoothingNode:
    """
    ComfyUI node for smoothing SMPL/MHR pose parameters.
    
    Reduces jitter in joint rotations while preserving motion dynamics.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE",),
            },
            "optional": {
                "method": (["Gaussian", "Savitzky-Golay", "Kalman", "SLERP", "Moving Average"], {
                    "default": "Gaussian",
                    "tooltip": "Smoothing algorithm. Gaussian is fast, Kalman is best for noisy data."
                }),
                "window_size": ("INT", {
                    "default": 5,
                    "min": 3,
                    "max": 21,
                    "step": 2,
                    "tooltip": "Smoothing window size (odd number). Larger = smoother but more lag."
                }),
                "sigma": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Gaussian sigma. Larger = smoother."
                }),
                "smooth_trajectory": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also smooth camera-relative position (pred_cam_t)."
                }),
                "smooth_rotations": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Smooth joint rotation matrices."
                }),
                "log_level": (["Normal", "Verbose", "Silent"], {
                    "default": "Normal",
                }),
            }
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE", "STRING")
    RETURN_NAMES = ("mesh_sequence", "status")
    FUNCTION = "smooth"
    CATEGORY = "SAM3DBody2abc/Processing"
    
    def smooth(
        self,
        mesh_sequence: Dict,
        method: str = "Gaussian",
        window_size: int = 5,
        sigma: float = 1.5,
        smooth_trajectory: bool = True,
        smooth_rotations: bool = True,
        log_level: str = "Normal",
    ) -> Tuple[Dict, str]:
        """Apply pose smoothing to mesh sequence."""
        verbose = log_level != "Silent"
        
        # Map method name
        method_map = {
            "Gaussian": "gaussian",
            "Savitzky-Golay": "savgol",
            "Kalman": "kalman",
            "SLERP": "slerp",
            "Moving Average": "moving_average",
        }
        method_key = method_map.get(method, "gaussian")
        
        # Apply smoothing
        result = smooth_mesh_sequence(
            mesh_sequence,
            method=method_key,
            window=window_size,
            sigma=sigma,
            smooth_trajectory=smooth_trajectory,
            smooth_rotations=smooth_rotations,
            verbose=verbose,
        )
        
        num_frames = len(result.get("frames", []))
        status = f"Smoothed {num_frames} frames using {method} (window={window_size})"
        
        return (result, status)


NODE_CLASS_MAPPINGS = {
    "PoseSmoothing": PoseSmoothingNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseSmoothing": "🔄 Pose Smoothing",
}
