"""
DPVO Wrapper for SAM3DBody2abc
==============================

Simplified interface for running DPVO on video files.
Handles initialization, processing, and output formatting.

Usage:
    from dpvo_wrapper import DPVOWrapper
    
    dpvo = DPVOWrapper()
    poses, points = dpvo.process_video("input.mp4", intrinsics=(fx, fy, cx, cy))
    
    # Or with automatic intrinsics estimation
    poses, points = dpvo.process_video("input.mp4")
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
import cv2

# Check for DPVO availability
DPVO_AVAILABLE = False
DPVO_PATH = None

def find_dpvo():
    """Locate DPVO installation."""
    global DPVO_AVAILABLE, DPVO_PATH
    
    # Check common locations
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
        return True
    except ImportError as e:
        print(f"[DPVO] Import error: {e}")
        return False

# Try to find DPVO on import
find_dpvo()


@dataclass
class DPVOConfig:
    """Configuration for DPVO."""
    # Network
    weights: str = "dpvo.pth"
    
    # Buffer sizes
    buffer_size: int = 2048
    patches_per_frame: int = 96
    
    # Tracking parameters
    max_age: int = 25
    removal_window: int = 22
    
    # Motion filter
    filter_thresh: float = 2.5
    
    # Device
    device: str = "cuda:0"


class DPVOWrapper:
    """
    Wrapper for DPVO visual odometry.
    
    Provides a simple interface for video processing with
    automatic initialization and error handling.
    """
    
    def __init__(self, config: Optional[DPVOConfig] = None):
        """
        Initialize DPVO wrapper.
        
        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        if not DPVO_AVAILABLE:
            raise RuntimeError(
                "DPVO not available. Install from: "
                "https://github.com/princeton-vl/DPVO"
            )
        
        self.config = config or DPVOConfig()
        self.slam = None
        self.initialized = False
        
        # Import DPVO modules
        from dpvo.dpvo import DPVO
        from dpvo.config import cfg
        self.DPVO = DPVO
        self.cfg = cfg
    
    def _init_slam(self, height: int, width: int):
        """Initialize SLAM with image dimensions."""
        # Load config
        config_path = DPVO_PATH / "config" / "default.yaml"
        if config_path.exists():
            self.cfg.merge_from_file(str(config_path))
        
        # Override with our config
        self.cfg.BUFFER_SIZE = self.config.buffer_size
        self.cfg.PATCHES_PER_FRAME = self.config.patches_per_frame
        self.cfg.REMOVAL_WINDOW = self.config.removal_window
        self.cfg.MAX_AGE = self.config.max_age
        self.cfg.FILTER_THRESH = self.config.filter_thresh
        
        # Load weights
        weights_path = DPVO_PATH / "models" / self.config.weights
        if not weights_path.exists():
            weights_path = DPVO_PATH / self.config.weights
        
        self.slam = self.DPVO(self.cfg, ht=height, wd=width, weights=str(weights_path))
        self.initialized = True
    
    def process_video(
        self,
        video_path: Union[str, Path],
        intrinsics: Optional[Tuple[float, float, float, float]] = None,
        stride: int = 1,
        max_frames: Optional[int] = None,
        verbose: bool = True
    ) -> Tuple[List[np.ndarray], np.ndarray, List[int]]:
        """
        Process a video file with DPVO.
        
        Args:
            video_path: Path to video file
            intrinsics: (fx, fy, cx, cy) camera intrinsics. Estimated if not provided.
            stride: Process every Nth frame
            max_frames: Maximum frames to process
            verbose: Print progress
        
        Returns:
            poses: List of 4x4 camera-to-world matrices
            points: Nx3 array of 3D map points
            timestamps: Frame indices
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if verbose:
            print(f"[DPVO] Video: {width}x{height}, {total_frames} frames @ {fps:.1f} fps")
        
        # Estimate intrinsics if not provided
        if intrinsics is None:
            # Assume ~60° horizontal FOV
            fx = width / (2 * np.tan(np.radians(30)))
            fy = fx
            cx = width / 2
            cy = height / 2
            intrinsics = (fx, fy, cx, cy)
            if verbose:
                print(f"[DPVO] Estimated intrinsics: fx={fx:.1f}, fy={fy:.1f}")
        
        fx, fy, cx, cy = intrinsics
        
        # Initialize SLAM
        if not self.initialized:
            self._init_slam(height, width)
        
        # Process frames
        frame_idx = 0
        processed = 0
        
        intrinsic_tensor = torch.tensor([fx, fy, cx, cy]).float().to(self.config.device)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and processed >= max_frames:
                break
            
            if frame_idx % stride == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # To tensor [C, H, W]
                image_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()
                image_tensor = image_tensor.to(self.config.device)
                
                # Process frame
                self.slam(frame_idx, image_tensor, intrinsic_tensor)
                
                processed += 1
                
                if verbose and processed % 100 == 0:
                    print(f"[DPVO] Processed {processed} frames ({frame_idx}/{total_frames})")
            
            frame_idx += 1
        
        cap.release()
        
        if verbose:
            print(f"[DPVO] Finalizing trajectory...")
        
        # Finalize
        self.slam.terminate()
        
        # Extract results
        poses = []
        timestamps = []
        
        trajectory = self.slam.traj
        for i in range(len(trajectory)):
            pose = trajectory[i].cpu().numpy()
            T = np.eye(4)
            T[:3, :4] = pose.reshape(3, 4)
            poses.append(T)
            timestamps.append(i * stride)
        
        # Get map points
        points = np.zeros((0, 3))
        if hasattr(self.slam, 'points') and self.slam.points is not None:
            points = self.slam.points.cpu().numpy()
        
        if verbose:
            print(f"[DPVO] Complete: {len(poses)} poses, {len(points)} points")
        
        return poses, points, timestamps
    
    def process_frames(
        self,
        frames: np.ndarray,
        intrinsics: Tuple[float, float, float, float],
        stride: int = 1,
        verbose: bool = True
    ) -> Tuple[List[np.ndarray], np.ndarray, List[int]]:
        """
        Process numpy array of frames with DPVO.
        
        Args:
            frames: [N, H, W, C] uint8 RGB frames
            intrinsics: (fx, fy, cx, cy) camera intrinsics
            stride: Process every Nth frame
            verbose: Print progress
        
        Returns:
            poses: List of 4x4 camera-to-world matrices
            points: Nx3 array of 3D map points
            timestamps: Frame indices
        """
        num_frames, height, width = frames.shape[:3]
        fx, fy, cx, cy = intrinsics
        
        if verbose:
            print(f"[DPVO] Processing {num_frames} frames ({width}x{height})")
        
        # Initialize SLAM
        if not self.initialized:
            self._init_slam(height, width)
        
        intrinsic_tensor = torch.tensor([fx, fy, cx, cy]).float().to(self.config.device)
        
        for i in range(0, num_frames, stride):
            frame = frames[i]
            
            # To tensor [C, H, W]
            image_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
            image_tensor = image_tensor.to(self.config.device)
            
            # Process frame
            self.slam(i, image_tensor, intrinsic_tensor)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"[DPVO] Processed {i + 1}/{num_frames} frames")
        
        if verbose:
            print(f"[DPVO] Finalizing trajectory...")
        
        # Finalize
        self.slam.terminate()
        
        # Extract results
        poses = []
        timestamps = []
        
        trajectory = self.slam.traj
        for i in range(len(trajectory)):
            pose = trajectory[i].cpu().numpy()
            T = np.eye(4)
            T[:3, :4] = pose.reshape(3, 4)
            poses.append(T)
            timestamps.append(i * stride)
        
        # Get map points
        points = np.zeros((0, 3))
        if hasattr(self.slam, 'points') and self.slam.points is not None:
            points = self.slam.points.cpu().numpy()
        
        if verbose:
            print(f"[DPVO] Complete: {len(poses)} poses, {len(points)} points")
        
        return poses, points, timestamps
    
    def reset(self):
        """Reset SLAM state for processing a new video."""
        self.slam = None
        self.initialized = False


def poses_to_camera_extrinsics(
    poses: List[np.ndarray],
    timestamps: List[int],
    scale: float = 1.0
) -> Dict:
    """
    Convert SLAM poses to SAM3DBody2abc CAMERA_EXTRINSICS format.
    
    Args:
        poses: List of 4x4 camera-to-world matrices
        timestamps: Frame indices
        scale: Metric scale factor
    
    Returns:
        camera_extrinsics: Dict compatible with FBX Export
    """
    rotations = []
    
    for pose, ts in zip(poses, timestamps):
        R = pose[:3, :3]
        t = pose[:3, 3] * scale
        
        # Convert rotation matrix to Euler angles (Y-X-Z convention)
        # Pan = rotation around Y (vertical)
        # Tilt = rotation around X (horizontal)
        # Roll = rotation around Z (camera axis)
        
        # Extract Euler angles
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        
        if not singular:
            tilt = np.arctan2(-R[2, 0], sy)
            pan = np.arctan2(R[1, 0], R[0, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])
        else:
            tilt = np.arctan2(-R[2, 0], sy)
            pan = np.arctan2(-R[0, 1], R[1, 1])
            roll = 0
        
        rotations.append({
            "frame": ts,
            "pan": float(pan),
            "tilt": float(tilt),
            "roll": float(roll),
            "tx": float(t[0]),
            "ty": float(t[1]),
            "tz": float(t[2]),
        })
    
    return {
        "rotations": rotations,
        "has_translation": True,
        "source": "DPVO",
        "scale": scale,
        "frame_count": len(rotations),
    }


# =============================================================================
# Command-line interface
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run DPVO on video")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--output", "-o", default="trajectory.json", help="Output JSON path")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--scale", type=float, default=1.0, help="Metric scale")
    parser.add_argument("--fx", type=float, help="Focal length X")
    parser.add_argument("--fy", type=float, help="Focal length Y")
    parser.add_argument("--cx", type=float, help="Principal point X")
    parser.add_argument("--cy", type=float, help="Principal point Y")
    
    args = parser.parse_args()
    
    # Check DPVO
    if not DPVO_AVAILABLE:
        print("Error: DPVO not found. Please install DPVO first.")
        print("  git clone --recursive https://github.com/princeton-vl/DPVO.git")
        print("  cd DPVO && pip install .")
        sys.exit(1)
    
    # Build intrinsics
    intrinsics = None
    if args.fx and args.fy and args.cx and args.cy:
        intrinsics = (args.fx, args.fy, args.cx, args.cy)
    
    # Run DPVO
    dpvo = DPVOWrapper()
    poses, points, timestamps = dpvo.process_video(
        args.video,
        intrinsics=intrinsics,
        stride=args.stride
    )
    
    # Convert to camera extrinsics
    extrinsics = poses_to_camera_extrinsics(poses, timestamps, args.scale)
    
    # Save
    import json
    with open(args.output, 'w') as f:
        json.dump(extrinsics, f, indent=2)
    
    print(f"Saved trajectory to {args.output}")
