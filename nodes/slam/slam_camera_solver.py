"""
SLAM Camera Solver for SAM3DBody2abc
=====================================

Integrates visual SLAM (DPVO) to recover world-coordinate camera poses
from monocular video. This enables accurate body motion capture even with moving cameras.

Pipeline:
    Video Frames → SLAM → Camera Poses → CAMERA_EXTRINSICS
                                              ↓
    SAM3DBody → MESH_SEQUENCE → Motion Analyzer → World-Coordinate Body Motion

Supported SLAM backends:
- DPVO: Lightweight, fast, 4GB VRAM (recommended)
- Fallback: Feature-based estimation when DPVO unavailable

NOTE: PyCuVSLAM removed (requires Python 3.10, incompatible with ComfyUI)

Author: SAM3DBody2abc
Version: 1.0.0
"""

import os
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from scipy.ndimage import gaussian_filter1d

# =============================================================================
# DPVO Availability Check
# =============================================================================

DPVO_AVAILABLE = False
DPVO_PATH = None

def _find_dpvo():
    """Locate DPVO installation."""
    global DPVO_AVAILABLE, DPVO_PATH
    import sys
    from pathlib import Path
    
    possible_paths = [
        Path.home() / "DPVO",
        Path.home() / "code" / "DPVO",
        Path("/opt/DPVO"),
        Path(os.environ.get("DPVO_PATH", "")),
    ]
    
    for path in possible_paths:
        if (path / "dpvo").exists():
            DPVO_PATH = path
            sys.path.insert(0, str(path))
            break
    
    if DPVO_PATH is None:
        return False
    
    try:
        from dpvo.dpvo import DPVO
        from dpvo.config import cfg
        DPVO_AVAILABLE = True
        print(f"[SLAM] DPVO found at {DPVO_PATH}")
        return True
    except ImportError as e:
        print(f"[SLAM] DPVO import error: {e}")
        return False

# Try to find DPVO on import
_find_dpvo()


class SLAMBackend(Enum):
    """Available SLAM backends."""
    DPVO = "dpvo"
    FEATURE_BASED = "feature_based"
    AUTO = "auto"


class SLAMCameraSolver:
    """
    ComfyUI node that uses visual SLAM to compute camera poses from video.
    
    Outputs CAMERA_EXTRINSICS compatible with FBX Export and Motion Analyzer.
    
    Key Features:
    - Automatic scale estimation using person height
    - Temporal alignment with mesh sequence
    - DPVO backend support
    - Feature-based fallback when DPVO unavailable
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        # Determine available backends
        backends = ["Auto (Best Available)", "Feature-Based (Fallback)"]
        if DPVO_AVAILABLE:
            backends.insert(1, "DPVO (Recommended)")
        
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                # SLAM configuration
                "slam_backend": (backends, {
                    "default": "Auto (Best Available)",
                    "tooltip": "SLAM algorithm to use. DPVO recommended for most cases."
                }),
                
                # Camera intrinsics
                "camera_intrinsics": ("CAMERA_INTRINSICS", {
                    "tooltip": "Camera intrinsics from MoGe2. If not provided, will estimate."
                }),
                
                # OR use INTRINSICS type from IntrinsicsEstimator
                "intrinsics": ("INTRINSICS", {
                    "tooltip": "Camera intrinsics from IntrinsicsEstimator."
                }),
                
                # Scale reference
                "mesh_sequence": ("MESH_SEQUENCE", {
                    "tooltip": "Mesh sequence for automatic scale calibration using person height."
                }),
                "reference_height": ("FLOAT", {
                    "default": 1.7,
                    "min": 0.5,
                    "max": 2.5,
                    "step": 0.01,
                    "tooltip": "Reference person height in meters for scale calibration."
                }),
                
                # SLAM parameters
                "keyframe_stride": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Process every Nth frame. Higher = faster but less accurate."
                }),
                "max_age": ("INT", {
                    "default": 25,
                    "min": 5,
                    "max": 100,
                    "tooltip": "Maximum keyframe age for DPVO."
                }),
                
                # Output options
                "coordinate_system": (["Y-Up (Maya/Blender)", "Z-Up (Unreal)"], {
                    "default": "Y-Up (Maya/Blender)",
                }),
                
                # Smoothing
                "apply_smoothing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply temporal smoothing to reduce jitter."
                }),
                "smoothing_window": ("INT", {
                    "default": 5,
                    "min": 3,
                    "max": 21,
                    "step": 2,
                }),
                
                # Debug
                "log_level": (["Normal", "Verbose", "Debug"], {
                    "default": "Normal",
                }),
            }
        }
    
    RETURN_TYPES = ("CAMERA_EXTRINSICS", "STRING")
    RETURN_NAMES = ("camera_extrinsics", "status")
    FUNCTION = "solve_camera"
    CATEGORY = "SAM3DBody2abc/Camera"
    
    def __init__(self):
        self.slam = None
    
    def solve_camera(
        self,
        images: torch.Tensor,
        slam_backend: str = "Auto (Best Available)",
        camera_intrinsics: Optional[Dict] = None,
        intrinsics: Optional[Dict] = None,
        mesh_sequence: Optional[Dict] = None,
        reference_height: float = 1.7,
        keyframe_stride: int = 1,
        max_age: int = 25,
        coordinate_system: str = "Y-Up (Maya/Blender)",
        apply_smoothing: bool = True,
        smoothing_window: int = 5,
        log_level: str = "Normal",
    ) -> Tuple[Dict, str]:
        """
        Run SLAM on video frames to recover camera trajectory.
        """
        verbose = log_level in ["Verbose", "Debug"]
        
        # Convert images to numpy
        if isinstance(images, torch.Tensor):
            frames = (images.cpu().numpy() * 255).astype(np.uint8)
        else:
            frames = images
        
        num_frames, height, width = frames.shape[:3]
        print(f"[SLAM Camera Solver] Processing {num_frames} frames ({width}x{height})")
        
        # Get intrinsics (try both formats)
        cam_intrinsics = self._get_intrinsics(camera_intrinsics, intrinsics, width, height)
        fx, fy, cx, cy = cam_intrinsics
        
        if verbose:
            print(f"[SLAM] Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        
        # Select backend
        backend = self._select_backend(slam_backend)
        print(f"[SLAM] Using backend: {backend.value}")
        
        # Run SLAM
        try:
            if backend == SLAMBackend.DPVO:
                poses, points, timestamps = self._run_dpvo(
                    frames, cam_intrinsics, keyframe_stride, max_age, verbose
                )
            else:
                poses, points, timestamps = self._run_feature_based(
                    frames, cam_intrinsics, verbose
                )
        except Exception as e:
            print(f"[SLAM] Error: {e}")
            import traceback
            traceback.print_exc()
            # Return identity poses as fallback
            poses = [np.eye(4) for _ in range(num_frames)]
            points = np.zeros((0, 3))
            timestamps = list(range(num_frames))
            
            camera_extrinsics = self._format_output(
                poses, timestamps, 1.0, coordinate_system,
                apply_smoothing, smoothing_window
            )
            return (camera_extrinsics, f"SLAM failed: {e}. Using identity poses.")
        
        # Estimate scale from mesh sequence
        scale = self._estimate_scale(poses, mesh_sequence, reference_height, verbose)
        
        if verbose:
            print(f"[SLAM] Estimated scale: {scale:.4f}")
        
        # Format output
        camera_extrinsics = self._format_output(
            poses, timestamps, scale, coordinate_system,
            apply_smoothing, smoothing_window
        )
        
        # Build status
        status = (
            f"SLAM completed: {backend.value} | "
            f"Frames: {num_frames} → {len(poses)} poses | "
            f"Scale: {scale:.4f} | "
            f"Points: {len(points)}"
        )
        
        return (camera_extrinsics, status)
    
    def _get_intrinsics(
        self,
        camera_intrinsics: Optional[Dict],
        intrinsics: Optional[Dict],
        width: int,
        height: int
    ) -> Tuple[float, float, float, float]:
        """Get camera intrinsics from various sources."""
        # Try CAMERA_INTRINSICS format first
        if camera_intrinsics is not None:
            fx = camera_intrinsics.get("focal_length_px", width)
            fy = fx
            cx = camera_intrinsics.get("principal_point_x", width / 2)
            cy = camera_intrinsics.get("principal_point_y", height / 2)
            return (fx, fy, cx, cy)
        
        # Try INTRINSICS format
        if intrinsics is not None:
            fx = intrinsics.get("focal_px", width)
            fy = fx
            cx = intrinsics.get("cx", width / 2)
            cy = intrinsics.get("cy", height / 2)
            return (fx, fy, cx, cy)
        
        # Estimate from image size (assume ~60° horizontal FOV)
        fx = width / (2 * np.tan(np.radians(30)))
        fy = fx
        cx = width / 2
        cy = height / 2
        return (fx, fy, cx, cy)
    
    def _select_backend(self, backend_str: str) -> SLAMBackend:
        """Select the best available SLAM backend."""
        if "DPVO" in backend_str:
            if DPVO_AVAILABLE:
                return SLAMBackend.DPVO
            else:
                print("[SLAM] DPVO requested but not available, using feature-based fallback")
                return SLAMBackend.FEATURE_BASED
        elif "Feature" in backend_str:
            return SLAMBackend.FEATURE_BASED
        else:
            # Auto selection
            if DPVO_AVAILABLE:
                return SLAMBackend.DPVO
            else:
                return SLAMBackend.FEATURE_BASED
    
    def _run_dpvo(
        self,
        frames: np.ndarray,
        intrinsics: Tuple[float, float, float, float],
        stride: int,
        max_age: int,
        verbose: bool
    ) -> Tuple[List[np.ndarray], np.ndarray, List[int]]:
        """Run DPVO on video frames."""
        if not DPVO_AVAILABLE:
            raise RuntimeError("DPVO not installed")
        
        from dpvo.dpvo import DPVO
        from dpvo.config import cfg
        
        fx, fy, cx, cy = intrinsics
        num_frames, height, width = frames.shape[:3]
        
        # Configure DPVO
        config_path = DPVO_PATH / "config" / "default.yaml"
        if config_path.exists():
            cfg.merge_from_file(str(config_path))
        
        cfg.BUFFER_SIZE = 2048
        cfg.PATCHES_PER_FRAME = 96
        cfg.MAX_AGE = max_age
        
        # Find weights
        weights_path = DPVO_PATH / "models" / "dpvo.pth"
        if not weights_path.exists():
            weights_path = DPVO_PATH / "dpvo.pth"
        
        # Initialize
        slam = DPVO(cfg, ht=height, wd=width, weights=str(weights_path))
        
        intrinsic_tensor = torch.tensor([fx, fy, cx, cy]).float().cuda()
        
        # Process frames
        for i in range(0, num_frames, stride):
            frame = frames[i]
            
            # Convert to tensor [C, H, W]
            image_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().cuda()
            
            slam(i, image_tensor, intrinsic_tensor)
            
            if verbose and i % 50 == 0:
                print(f"[DPVO] Processed frame {i}/{num_frames}")
        
        # Finalize
        slam.terminate()
        
        # Extract poses
        poses = []
        timestamps = []
        
        for i, pose in enumerate(slam.traj):
            T = np.eye(4)
            T[:3, :4] = pose.cpu().numpy().reshape(3, 4)
            poses.append(T)
            timestamps.append(i * stride)
        
        # Get points
        points = np.zeros((0, 3))
        if hasattr(slam, 'points') and slam.points is not None:
            points = slam.points.cpu().numpy()
        
        return poses, points, timestamps
    
    def _run_feature_based(
        self,
        frames: np.ndarray,
        intrinsics: Tuple[float, float, float, float],
        verbose: bool
    ) -> Tuple[List[np.ndarray], np.ndarray, List[int]]:
        """
        Feature-based fallback when DPVO not available.
        Uses homography decomposition for rotation estimation.
        """
        import cv2
        
        fx, fy, cx, cy = intrinsics
        num_frames = frames.shape[0]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        poses = [np.eye(4)]  # First frame at origin
        timestamps = [0]
        
        # Feature detector
        orb = cv2.ORB_create(nfeatures=1000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        prev_kp, prev_desc = orb.detectAndCompute(prev_gray, None)
        
        cumulative_pose = np.eye(4)
        
        for i in range(1, num_frames):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            curr_kp, curr_desc = orb.detectAndCompute(curr_gray, None)
            
            if prev_desc is None or curr_desc is None:
                poses.append(cumulative_pose.copy())
                timestamps.append(i)
                continue
            
            # Match features
            matches = bf.match(prev_desc, curr_desc)
            
            if len(matches) < 4:
                poses.append(cumulative_pose.copy())
                timestamps.append(i)
                continue
            
            # Get matched points
            pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
            pts2 = np.float32([curr_kp[m.trainIdx].pt for m in matches])
            
            # Find homography
            H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
            
            if H is not None:
                # Decompose homography to get rotation
                num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)
                
                if num > 0:
                    # Use first decomposition (typically best)
                    R = Rs[0]
                    
                    # Update cumulative pose
                    delta_pose = np.eye(4)
                    delta_pose[:3, :3] = R
                    cumulative_pose = cumulative_pose @ delta_pose
            
            poses.append(cumulative_pose.copy())
            timestamps.append(i)
            
            # Update for next iteration
            prev_gray = curr_gray
            prev_kp, prev_desc = curr_kp, curr_desc
            
            if verbose and i % 50 == 0:
                print(f"[Feature-Based] Processed frame {i}/{num_frames}")
        
        return poses, np.zeros((0, 3)), timestamps
    
    def _estimate_scale(
        self,
        poses: List[np.ndarray],
        mesh_sequence: Optional[Dict],
        reference_height: float,
        verbose: bool
    ) -> float:
        """Estimate metric scale using person height from mesh sequence."""
        if mesh_sequence is None:
            if verbose:
                print("[SLAM] No mesh sequence provided, using unit scale")
            return 1.0
        
        # Get person heights from mesh sequence
        heights = []
        frames = mesh_sequence.get("frames", [])
        
        for frame in frames:
            joints = frame.get("joint_coords")
            if joints is None:
                continue
            
            joints = np.array(joints)
            
            # Height = head to feet distance
            # MHR joint indices: 0=pelvis, 15=head, 10/11=feet
            if joints.shape[0] > 15:
                head = joints[15]
                left_foot = joints[10] if joints.shape[0] > 10 else joints[0]
                right_foot = joints[11] if joints.shape[0] > 11 else joints[0]
                feet = (left_foot + right_foot) / 2
                
                height = np.linalg.norm(head - feet)
                if height > 0.1:
                    heights.append(height)
        
        if len(heights) == 0:
            if verbose:
                print("[SLAM] Could not estimate person height, using unit scale")
            return 1.0
        
        observed_height = np.median(heights)
        scale = reference_height / observed_height
        
        if verbose:
            print(f"[SLAM] Observed height: {observed_height:.3f}, scale: {scale:.4f}")
        
        return scale
    
    def _format_output(
        self,
        poses: List[np.ndarray],
        timestamps: List[int],
        scale: float,
        coordinate_system: str,
        apply_smoothing: bool,
        smoothing_window: int
    ) -> Dict:
        """Format SLAM output for SAM3DBody2abc pipeline."""
        # Apply scale to translations
        scaled_poses = []
        for pose in poses:
            scaled = pose.copy()
            scaled[:3, 3] *= scale
            scaled_poses.append(scaled)
        
        # Convert coordinate system if needed
        if "Z-Up" in coordinate_system:
            R_conv = np.array([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]
            ])
            for i, pose in enumerate(scaled_poses):
                scaled_poses[i][:3, :3] = R_conv @ pose[:3, :3]
                scaled_poses[i][:3, 3] = R_conv @ pose[:3, 3]
        
        # Extract rotations as Euler (pan/tilt/roll)
        rotations = []
        for pose, ts in zip(scaled_poses, timestamps):
            R = pose[:3, :3]
            t = pose[:3, 3]
            
            # Convert to pan/tilt/roll (Y-X-Z Euler)
            pan = np.arctan2(R[0, 2], R[2, 2])
            tilt = np.arcsin(np.clip(-R[1, 2], -1, 1))
            roll = np.arctan2(R[1, 0], R[1, 1])
            
            rotations.append({
                "frame": ts,
                "pan": float(pan),
                "tilt": float(tilt),
                "roll": float(roll),
                "tx": float(t[0]),
                "ty": float(t[1]),
                "tz": float(t[2]),
            })
        
        # Apply smoothing
        if apply_smoothing and len(rotations) > smoothing_window:
            rotations = self._smooth_trajectory(rotations, smoothing_window)
        
        return {
            "rotations": rotations,
            "has_translation": True,
            "source": "SLAM",
            "scale": scale,
            "frame_count": len(rotations),
        }
    
    def _smooth_trajectory(
        self,
        rotations: List[Dict],
        window: int
    ) -> List[Dict]:
        """Apply Gaussian smoothing to camera trajectory."""
        keys = ["pan", "tilt", "roll", "tx", "ty", "tz"]
        available_keys = [k for k in keys if k in rotations[0]]
        
        arrays = {k: np.array([r[k] for r in rotations]) for k in available_keys}
        
        sigma = window / 3.0
        smoothed = {k: gaussian_filter1d(v, sigma) for k, v in arrays.items()}
        
        result = []
        for i, r in enumerate(rotations):
            new_r = {"frame": r["frame"]}
            for k in available_keys:
                new_r[k] = float(smoothed[k][i])
            result.append(new_r)
        
        return result


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "SLAMCameraSolver": SLAMCameraSolver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SLAMCameraSolver": "📹 SLAM Camera Solver",
}
