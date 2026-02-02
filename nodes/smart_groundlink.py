"""
SmartGroundLink - Unified Video-Based Foot Contact Detection + Physics Pinning
===============================================================================

Combines:
- SmartFootContact's video-based detection (TAPNet tracking, video as ground truth)
- GroundLink's physics-based foot pinning (proper IK-like adjustment)

The key insight: Use VIDEO to detect WHEN feet are in contact, then use
GroundLink's proven pinning logic to keep feet locked at those positions
without any sliding.

Author: Claude (Anthropic)
Version: 1.0.0
License: Apache 2.0
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

# =============================================================================
# Logging
# =============================================================================

class SmartGroundLinkLogger:
    """Simple logger for SmartGroundLink."""
    
    def __init__(self, level: str = "normal"):
        self.level = level
        self.timings = {}
        self._timer_starts = {}
    
    def _should_log(self, msg_level: str) -> bool:
        levels = {"silent": 0, "normal": 1, "verbose": 2, "debug": 3}
        return levels.get(msg_level, 1) <= levels.get(self.level, 1)
    
    def info(self, msg: str):
        if self._should_log("normal"):
            print(f"[SmartGroundLink] {msg}")
    
    def verbose(self, msg: str):
        if self._should_log("verbose"):
            print(f"[SmartGroundLink] {msg}")
    
    def debug(self, msg: str):
        if self._should_log("debug"):
            print(f"[SmartGroundLink] DEBUG: {msg}")
    
    def warning(self, msg: str):
        print(f"[SmartGroundLink] ⚠ WARNING: {msg}")
    
    def error(self, msg: str):
        print(f"[SmartGroundLink] ✗ ERROR: {msg}")
    
    def start_timer(self, name: str):
        import time
        self._timer_starts[name] = time.time()
    
    def end_timer(self, name: str):
        import time
        if name in self._timer_starts:
            elapsed = time.time() - self._timer_starts[name]
            self.timings[name] = elapsed
            if self._should_log("verbose"):
                print(f"[SmartGroundLink] ⏱ {name}: {elapsed:.3f}s")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SmartGroundLinkConfig:
    """Configuration for SmartGroundLink."""
    
    # Contact detection (from video)
    velocity_threshold: float = 3.0  # Pixels/frame for "stationary"
    min_contact_duration: int = 3    # Minimum frames for valid contact
    max_gap_to_bridge: int = 2       # Bridge small gaps
    
    # Pinning behavior
    pin_feet: bool = True            # Actually apply pinning
    pin_strength: float = 0.5        # How strongly to pin (0-1)
    smooth_sigma: float = 1.0        # Gaussian smoothing of result
    
    # Foot indices (SMPL-H / MHR)
    left_foot_idx: int = 10
    right_foot_idx: int = 11
    
    # Coordinate system
    up_axis: str = "y"  # "y" or "z"
    
    # Memory management
    chunk_size: int = 24
    max_tracking_resolution: int = 512
    
    # Logging
    log_level: str = "verbose"


# =============================================================================
# TAPNet Tracker (from SmartFootContact)
# =============================================================================

class TAPNetTracker:
    """TAPNet-based foot tracking for contact detection."""
    
    def __init__(self, log: SmartGroundLinkLogger):
        self.log = log
        self._model = None
        self._device = None
    
    def _ensure_model(self) -> bool:
        """Load TAPNet model."""
        if self._model is not None:
            return self._model != "fallback"
        
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            from tapnet.torch import tapir_model
            import os
            
            self.log.info(f"Loading TAPNet on {self._device}")
            
            # Check local paths first
            checkpoint_paths = [
                "/workspace/ComfyUI/models/tapir/bootstapir_checkpoint_v2.pt",
                "/workspace/models/tapir/bootstapir_checkpoint_v2.pt",
                os.path.expanduser("~/models/tapir/bootstapir_checkpoint_v2.pt"),
                os.path.expanduser("~/.cache/torch/hub/checkpoints/bootstapir_checkpoint_v2.pt"),
                "/root/.cache/torch/hub/checkpoints/bootstapir_checkpoint_v2.pt",
            ]
            
            checkpoint_path = None
            for path in checkpoint_paths:
                if os.path.exists(path):
                    checkpoint_path = path
                    break
            
            if checkpoint_path is None:
                self.log.warning("TAPNet checkpoint not found")
                self._model = "fallback"
                return False
            
            self._model = tapir_model.TAPIR(pyramid_level=1)
            self._model.load_state_dict(torch.load(checkpoint_path, map_location=self._device))
            self._model = self._model.to(self._device)
            self._model.eval()
            
            self.log.info(f"TAPNet loaded from {checkpoint_path}")
            return True
            
        except ImportError:
            self.log.warning("TAPNet not installed - using optical flow fallback")
            self._model = "fallback"
            return False
        except Exception as e:
            self.log.error(f"TAPNet load failed: {e}")
            self._model = "fallback"
            return False
    
    def track_feet(
        self,
        video: np.ndarray,
        initial_positions: np.ndarray,
        chunk_size: int = 24,
        max_resolution: int = 512
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track foot positions through video.
        
        Args:
            video: (T, H, W, 3) video frames
            initial_positions: (2, 2) initial [x, y] for [left, right] feet
            chunk_size: Frames per chunk
            max_resolution: Max dimension for tracking
        
        Returns:
            tracks: (T, 2, 2) positions [time, foot, xy]
            visible: (T, 2) visibility mask
        """
        if not self._ensure_model():
            return self._track_optical_flow(video, initial_positions)
        
        T, H, W, _ = video.shape
        
        # Resize for memory efficiency
        scale_x, scale_y = 1.0, 1.0
        if max(H, W) > max_resolution:
            scale = max_resolution / max(H, W)
            new_H = (int(H * scale) // 8) * 8
            new_W = (int(W * scale) // 8) * 8
            new_H = max(new_H, 8)
            new_W = max(new_W, 8)
            
            self.log.verbose(f"Resizing video from {W}x{H} to {new_W}x{new_H}")
            
            import cv2
            video_resized = np.zeros((T, new_H, new_W, 3), dtype=np.uint8)
            for t in range(T):
                video_resized[t] = cv2.resize(video[t], (new_W, new_H))
            video = video_resized
            
            scale_x = new_W / W
            scale_y = new_H / H
            H, W = new_H, new_W
        
        # Scale initial positions
        scaled_init = initial_positions.copy()
        scaled_init[:, 0] *= scale_x
        scaled_init[:, 1] *= scale_y
        
        # Process in chunks
        all_tracks = np.zeros((T, 2, 2), dtype=np.float32)
        all_visible = np.ones((T, 2), dtype=bool)
        
        num_chunks = (T + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, T)
            
            # Get query positions (from previous chunk end or initial)
            if chunk_idx == 0:
                query_xy = scaled_init
            else:
                query_xy = all_tracks[start - 1]
            
            chunk_tracks, chunk_visible = self._track_chunk(
                video[start:end], query_xy
            )
            
            all_tracks[start:end] = chunk_tracks
            all_visible[start:end] = chunk_visible
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Scale back to original resolution
        all_tracks[:, :, 0] /= scale_x
        all_tracks[:, :, 1] /= scale_y
        
        return all_tracks, all_visible
    
    def _track_chunk(
        self,
        video: np.ndarray,
        query_xy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Track a single chunk of video."""
        T, H, W, _ = video.shape
        
        # Prepare video tensor [1, T, H, W, C]
        video_tensor = torch.from_numpy(video).float()
        if video_tensor.max() > 1.0:
            video_tensor = video_tensor / 255.0
        video_tensor = video_tensor * 2 - 1  # Normalize to [-1, 1]
        video_tensor = video_tensor.unsqueeze(0).to(self._device)
        
        # Query points: [1, 2, 3] where each is [t, y, x]
        # Build on CPU first, then move to device
        query_data = np.array([
            [0, query_xy[0, 1], query_xy[0, 0]],  # left foot: [frame, y, x]
            [0, query_xy[1, 1], query_xy[1, 0]],  # right foot: [frame, y, x]
        ], dtype=np.float32)
        query_points = torch.from_numpy(query_data).unsqueeze(0).to(self._device)
        
        try:
            with torch.no_grad():
                outputs = self._model(video_tensor, query_points)
            
            # Handle BFloat16
            tracks_tensor = outputs['tracks'][0]  # [N, T, 2]
            if tracks_tensor.dtype == torch.bfloat16:
                tracks_tensor = tracks_tensor.float()
            tracks = tracks_tensor.cpu().numpy()
            tracks = np.transpose(tracks, (1, 0, 2))  # [T, N, 2]
            
            occ_tensor = outputs['occlusion'][0]  # [N, T]
            if occ_tensor.dtype == torch.bfloat16:
                occ_tensor = occ_tensor.float()
            occlusion = occ_tensor.cpu().numpy()
            visible = (occlusion < 0).T  # [T, N]
            
            return tracks, visible
            
        finally:
            del video_tensor, query_points
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _track_optical_flow(
        self,
        video: np.ndarray,
        initial_positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback tracking using optical flow."""
        import cv2
        
        T, H, W, _ = video.shape
        tracks = np.zeros((T, 2, 2), dtype=np.float32)
        visible = np.ones((T, 2), dtype=bool)
        
        tracks[0] = initial_positions
        
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        for t in range(1, T):
            prev_gray = cv2.cvtColor(video[t-1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(video[t], cv2.COLOR_RGB2GRAY)
            
            prev_pts = tracks[t-1].reshape(-1, 1, 2).astype(np.float32)
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None, **lk_params
            )
            
            if curr_pts is not None:
                for i in range(2):
                    if status[i] == 1:
                        tracks[t, i] = curr_pts[i, 0]
                    else:
                        tracks[t, i] = tracks[t-1, i]
                        visible[t, i] = False
            else:
                tracks[t] = tracks[t-1]
                visible[t] = False
        
        return tracks, visible


# =============================================================================
# Contact Detection
# =============================================================================

def detect_contacts_from_tracks(
    tracks: np.ndarray,
    visible: np.ndarray,
    velocity_threshold: float = 3.0,
    min_duration: int = 3,
    max_gap: int = 2
) -> np.ndarray:
    """
    Detect foot contacts based on 2D velocity (stationary = contact).
    
    Args:
        tracks: (T, 2, 2) foot positions [time, foot, xy]
        visible: (T, 2) visibility
        velocity_threshold: Pixels/frame below which foot is "stationary"
        min_duration: Minimum frames for valid contact
        max_gap: Bridge gaps smaller than this
    
    Returns:
        contacts: (T, 2) boolean contact array
    """
    T = tracks.shape[0]
    contacts = np.zeros((T, 2), dtype=bool)
    
    # Compute velocities
    velocities = np.zeros((T, 2), dtype=np.float32)
    velocities[1:] = np.linalg.norm(np.diff(tracks, axis=0), axis=2)
    velocities[0] = velocities[1] if T > 1 else 0
    
    # Smooth velocities slightly
    velocities = gaussian_filter1d(velocities, sigma=1.0, axis=0)
    
    # Low velocity = contact
    raw_contacts = velocities < velocity_threshold
    
    # Apply visibility mask
    raw_contacts = raw_contacts & visible
    
    # Process each foot
    for foot_idx in range(2):
        foot_contacts = raw_contacts[:, foot_idx].copy()
        
        # Bridge small gaps
        if max_gap > 0:
            in_gap = 0
            gap_start = 0
            for t in range(T):
                if foot_contacts[t]:
                    if in_gap > 0 and in_gap <= max_gap:
                        foot_contacts[gap_start:t] = True
                    in_gap = 0
                else:
                    if in_gap == 0:
                        gap_start = t
                    in_gap += 1
        
        # Filter short segments
        if min_duration > 1:
            segment_start = None
            for t in range(T + 1):
                is_contact = foot_contacts[t] if t < T else False
                
                if is_contact and segment_start is None:
                    segment_start = t
                elif not is_contact and segment_start is not None:
                    if t - segment_start < min_duration:
                        foot_contacts[segment_start:t] = False
                    segment_start = None
        
        contacts[:, foot_idx] = foot_contacts
    
    return contacts


# =============================================================================
# Foot Pinning (from GroundLink)
# =============================================================================

def extract_joints(frame: Dict) -> np.ndarray:
    """Extract joint positions from frame data."""
    # Try different formats - SAM3DBody uses pred_keypoints_3d
    if "pred_keypoints_3d" in frame:
        joints = frame["pred_keypoints_3d"]
    elif "joints_3d" in frame:
        joints = frame["joints_3d"]
    elif "smpl_params" in frame and "joints" in frame["smpl_params"]:
        joints = frame["smpl_params"]["joints"]
    elif "keypoints_3d" in frame:
        joints = frame["keypoints_3d"]
    elif "joint_coords" in frame:
        joints = frame["joint_coords"]
    else:
        return None
    
    if isinstance(joints, list):
        joints = np.array(joints)
    
    # Handle nested arrays
    while joints.ndim > 2:
        joints = joints[0]
    
    return joints


def enforce_contacts(
    frames: List[Dict],
    root_trans: np.ndarray,
    contacts: np.ndarray,
    config: SmartGroundLinkConfig,
    log: SmartGroundLinkLogger
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Adjust root translation to pin feet during contacts.
    
    This is the key function that prevents foot sliding - when a foot
    is in contact, we adjust the body position so the foot stays locked
    at its pin position.
    """
    T = len(frames)
    adjusted = root_trans.copy()
    pin_events = []
    
    # Extract foot positions from skeleton
    foot_positions = [[], []]  # [left, right]
    
    for frame in frames:
        joints = extract_joints(frame)
        if joints is None:
            # Fallback - use zeros
            foot_positions[0].append(np.zeros(3))
            foot_positions[1].append(np.zeros(3))
            continue
        
        n_joints = len(joints)
        left_idx = min(config.left_foot_idx, n_joints - 1)
        right_idx = min(config.right_foot_idx, n_joints - 1)
        
        foot_positions[0].append(joints[left_idx])
        foot_positions[1].append(joints[right_idx])
    
    foot_positions = [np.stack(fp) for fp in foot_positions]
    
    # Pin positions (set when contact starts)
    pin_positions = [None, None]
    foot_names = ["left", "right"]
    
    for t in range(T):
        for foot_idx in range(2):
            if contacts[t, foot_idx]:
                # Foot is in contact - compute world position
                foot_world = foot_positions[foot_idx][t] + adjusted[t]
                
                if pin_positions[foot_idx] is None:
                    # Start of contact - set pin position
                    pin_positions[foot_idx] = foot_world.copy()
                    pin_events.append({
                        "frame": t,
                        "foot": foot_names[foot_idx],
                        "event": "pin_start",
                        "position": foot_world.tolist(),
                    })
                    log.debug(f"Frame {t}: {foot_names[foot_idx]} pin START at {foot_world}")
                else:
                    # Continuing contact - adjust to keep foot at pin position
                    # This is where we prevent sliding!
                    adjustment = pin_positions[foot_idx] - foot_world
                    
                    # Don't adjust vertical (let the foot stay on ground naturally)
                    if config.up_axis == 'y':
                        adjustment[1] = 0
                    else:
                        adjustment[2] = 0
                    
                    # Apply with strength factor
                    adjusted[t] += adjustment * config.pin_strength
            else:
                # Not in contact
                if pin_positions[foot_idx] is not None:
                    # End of contact
                    pin_events.append({
                        "frame": t,
                        "foot": foot_names[foot_idx],
                        "event": "pin_end",
                        "position": pin_positions[foot_idx].tolist(),
                    })
                    log.debug(f"Frame {t}: {foot_names[foot_idx]} pin END")
                pin_positions[foot_idx] = None
    
    # Smooth the result to avoid jitter
    if config.smooth_sigma > 0:
        adjusted = gaussian_filter1d(adjusted, sigma=config.smooth_sigma, axis=0)
    
    return adjusted, pin_events


# =============================================================================
# Main Processor
# =============================================================================

class SmartGroundLinkProcessor:
    """
    Main processor combining video-based detection with physics pinning.
    """
    
    def __init__(self, config: SmartGroundLinkConfig):
        self.config = config
        self.log = SmartGroundLinkLogger(config.log_level)
        self.tracker = TAPNetTracker(self.log)
    
    def process(
        self,
        mesh_sequence: Dict,
        video: Optional[np.ndarray]
    ) -> Dict:
        """
        Process mesh sequence with video-guided foot pinning.
        
        Args:
            mesh_sequence: SAM3DBody mesh sequence
            video: (T, H, W, 3) video frames
        
        Returns:
            Updated mesh_sequence with pinned feet
        """
        self.log.info("=" * 60)
        self.log.info("SMART GROUNDLINK PROCESSING")
        self.log.info("=" * 60)
        
        # Extract frames
        frames_data = mesh_sequence.get("frames", {})
        if isinstance(frames_data, dict):
            frame_keys = sorted(frames_data.keys())
            frames = [frames_data[k] for k in frame_keys]
        else:
            frame_keys = None
            frames = list(frames_data)
        
        T = len(frames)
        self.log.info(f"Processing {T} frames")
        
        if video is None:
            self.log.warning("No video provided - using heuristic contact detection")
            return self._process_heuristic(mesh_sequence, frames, frame_keys)
        
        H, W = video.shape[1:3]
        self.log.info(f"Video size: {W}x{H}")
        
        # Step 1: Get initial foot positions from SAM3D
        self.log.start_timer("initialization")
        initial_feet_2d = self._get_initial_feet_2d(frames[0], mesh_sequence)
        if initial_feet_2d is None:
            self.log.error("Could not get initial foot positions")
            return mesh_sequence
        self.log.verbose(f"Initial feet: L={initial_feet_2d[0]}, R={initial_feet_2d[1]}")
        self.log.end_timer("initialization")
        
        # Step 2: Track feet in video
        self.log.start_timer("tracking")
        tracks, visible = self.tracker.track_feet(
            video, initial_feet_2d,
            chunk_size=self.config.chunk_size,
            max_resolution=self.config.max_tracking_resolution
        )
        self.log.end_timer("tracking")
        
        # Step 3: Detect contacts from video
        self.log.start_timer("detection")
        contacts = detect_contacts_from_tracks(
            tracks, visible,
            velocity_threshold=self.config.velocity_threshold,
            min_duration=self.config.min_contact_duration,
            max_gap=self.config.max_gap_to_bridge
        )
        
        left_frames = contacts[:, 0].sum()
        right_frames = contacts[:, 1].sum()
        self.log.info(f"Detected contacts: Left={left_frames} frames, Right={right_frames} frames")
        self.log.end_timer("detection")
        
        # Step 4: Extract root translations
        root_trans = self._extract_translations(frames)
        if root_trans is None:
            self.log.error("Could not extract translations")
            return mesh_sequence
        
        # Step 5: Apply pinning with verification loop
        self.log.start_timer("pinning")
        if self.config.pin_feet:
            adjusted_trans, pin_events = enforce_contacts(
                frames, root_trans, contacts, self.config, self.log
            )
            self.log.info(f"Applied {len(pin_events)} pin events")
        else:
            adjusted_trans = root_trans
            pin_events = []
        self.log.end_timer("pinning")
        
        # Step 6: VERIFICATION - Check pinned feet against video ground truth
        self.log.start_timer("verification")
        verification_result = self._verify_against_video(
            frames, adjusted_trans, contacts, tracks, mesh_sequence
        )
        self.log.end_timer("verification")
        
        # Step 7: Update mesh_sequence
        result = self._update_mesh_sequence(
            mesh_sequence, frames, frame_keys, adjusted_trans, contacts
        )
        
        # Add metadata
        result["smart_groundlink"] = {
            "method": "video_guided_pinning",
            "contacts_detected": {
                "left": int(left_frames),
                "right": int(right_frames),
            },
            "pin_events": pin_events,
            "verification": verification_result,
            "timings": self.log.timings,
            "config": {
                "velocity_threshold": self.config.velocity_threshold,
                "min_contact_duration": self.config.min_contact_duration,
                "pin_strength": self.config.pin_strength,
            }
        }
        
        self._log_summary(contacts, pin_events, verification_result)
        
        return result
    
    def _get_initial_feet_2d(
        self,
        frame: Dict,
        mesh_sequence: Dict
    ) -> Optional[np.ndarray]:
        """Get initial 2D foot positions from SAM3D projection."""
        # Try pred_keypoints_2d first (SAM3DBody format)
        kp2d = None
        for key in ["pred_keypoints_2d", "keypoints_2d", "joints_2d"]:
            if key in frame:
                kp2d = np.array(frame[key])
                break
        
        if kp2d is not None:
            if kp2d.ndim > 2:
                kp2d = kp2d[0]
            
            left_idx = min(self.config.left_foot_idx, len(kp2d) - 1)
            right_idx = min(self.config.right_foot_idx, len(kp2d) - 1)
            
            self.log.debug(f"Found 2D keypoints with {len(kp2d)} joints")
            return np.array([kp2d[left_idx, :2], kp2d[right_idx, :2]])
        
        # Fallback: project 3D joints
        joints = extract_joints(frame)
        if joints is None:
            self.log.warning(f"No keypoints found. Available keys: {list(frame.keys())}")
            return None
        
        self.log.debug(f"Projecting from 3D joints ({len(joints)} joints)")
        
        # Get camera parameters
        focal = frame.get("focal_length", mesh_sequence.get("focal_length", 1000.0))
        cam_t = frame.get("pred_cam_t", [0, 0, 5])
        cx = frame.get("cx", mesh_sequence.get("width", 1280) / 2)
        cy = frame.get("cy", mesh_sequence.get("height", 720) / 2)
        
        # Simple projection
        left_idx = min(self.config.left_foot_idx, len(joints) - 1)
        right_idx = min(self.config.right_foot_idx, len(joints) - 1)
        
        feet_3d = np.array([joints[left_idx], joints[right_idx]])
        feet_3d_world = feet_3d + np.array(cam_t)
        
        # Project to 2D
        z = feet_3d_world[:, 2:3]
        z = np.maximum(z, 0.1)
        feet_2d = feet_3d_world[:, :2] * focal / z
        
        # Add principal point offset
        feet_2d[:, 0] += cx
        feet_2d[:, 1] += cy
        
        return feet_2d
    
    def _extract_translations(self, frames: List[Dict]) -> Optional[np.ndarray]:
        """Extract root translations from frames."""
        translations = []
        
        for frame in frames:
            if "pred_cam_t" in frame:
                trans = frame["pred_cam_t"]
            elif "smpl_params" in frame and "transl" in frame["smpl_params"]:
                trans = frame["smpl_params"]["transl"]
            else:
                trans = [0, 0, 5]  # Default
            
            if isinstance(trans, (list, tuple)):
                trans = np.array(trans)
            
            translations.append(trans)
        
        return np.stack(translations)
    
    def _verify_against_video(
        self,
        frames: List[Dict],
        adjusted_trans: np.ndarray,
        contacts: np.ndarray,
        video_tracks: np.ndarray,
        mesh_sequence: Dict
    ) -> Dict:
        """
        Verify pinned feet match video ground truth.
        
        After pinning, we reproject the 3D foot positions to 2D and compare
        against the TAPNet tracked positions (video ground truth).
        
        Args:
            frames: Frame data with joint positions
            adjusted_trans: Adjusted root translations after pinning
            contacts: (T, 2) contact mask
            video_tracks: (T, 2, 2) TAPNet tracked foot positions [time, foot, xy]
            mesh_sequence: For camera intrinsics
        
        Returns:
            Verification results with errors per frame
        """
        T = len(frames)
        cfg = self.config
        
        # Get camera intrinsics
        focal = mesh_sequence.get("focal_length", 1000.0)
        width = mesh_sequence.get("width", 1280)
        height = mesh_sequence.get("height", 720)
        cx, cy = width / 2, height / 2
        
        # Extract foot positions and reproject with adjusted translations
        errors_left = []
        errors_right = []
        reprojected_2d = np.zeros((T, 2, 2), dtype=np.float32)  # [T, foot, xy]
        
        for t, frame in enumerate(frames):
            joints = extract_joints(frame)
            if joints is None:
                errors_left.append(np.nan)
                errors_right.append(np.nan)
                continue
            
            n_joints = len(joints)
            left_idx = min(cfg.left_foot_idx, n_joints - 1)
            right_idx = min(cfg.right_foot_idx, n_joints - 1)
            
            # 3D foot positions in world space (with adjusted translation)
            left_3d = joints[left_idx] + adjusted_trans[t]
            right_3d = joints[right_idx] + adjusted_trans[t]
            
            # Project to 2D
            # Simple pinhole projection: x_2d = fx * X/Z + cx
            for foot_idx, foot_3d in enumerate([left_3d, right_3d]):
                z = max(foot_3d[2], 0.1)  # Avoid division by zero
                x_2d = focal * foot_3d[0] / z + cx
                y_2d = focal * foot_3d[1] / z + cy
                reprojected_2d[t, foot_idx] = [x_2d, y_2d]
            
            # Compute errors against video ground truth
            if contacts[t, 0]:  # Left foot in contact
                error_l = np.linalg.norm(reprojected_2d[t, 0] - video_tracks[t, 0])
                errors_left.append(error_l)
            else:
                errors_left.append(np.nan)
            
            if contacts[t, 1]:  # Right foot in contact
                error_r = np.linalg.norm(reprojected_2d[t, 1] - video_tracks[t, 1])
                errors_right.append(error_r)
            else:
                errors_right.append(np.nan)
        
        # Compute statistics
        errors_left = np.array(errors_left)
        errors_right = np.array(errors_right)
        
        valid_left = ~np.isnan(errors_left)
        valid_right = ~np.isnan(errors_right)
        
        left_mean = np.nanmean(errors_left) if valid_left.any() else 0.0
        left_max = np.nanmax(errors_left) if valid_left.any() else 0.0
        right_mean = np.nanmean(errors_right) if valid_right.any() else 0.0
        right_max = np.nanmax(errors_right) if valid_right.any() else 0.0
        
        overall_mean = np.nanmean(np.concatenate([errors_left[valid_left], errors_right[valid_right]])) \
            if (valid_left.any() or valid_right.any()) else 0.0
        
        # Determine quality
        if overall_mean < 5.0:
            quality = "excellent"
        elif overall_mean < 10.0:
            quality = "good"
        elif overall_mean < 20.0:
            quality = "acceptable"
        else:
            quality = "poor"
        
        # Log results
        self.log.info(f"")
        self.log.info(f"=== VERIFICATION vs VIDEO GROUND TRUTH ===")
        self.log.info(f"Left foot error:  mean={left_mean:.1f}px, max={left_max:.1f}px")
        self.log.info(f"Right foot error: mean={right_mean:.1f}px, max={right_max:.1f}px")
        self.log.info(f"Overall error:    mean={overall_mean:.1f}px")
        self.log.info(f"Quality: {quality.upper()}")
        
        if quality == "poor":
            self.log.warning("High reprojection error! Feet may not match video accurately.")
            self.log.warning("Consider: lower pin_strength, check initial foot detection, or verify video quality")
        
        return {
            "left_error_mean_px": float(left_mean),
            "left_error_max_px": float(left_max),
            "right_error_mean_px": float(right_mean),
            "right_error_max_px": float(right_max),
            "overall_error_mean_px": float(overall_mean),
            "quality": quality,
            "errors_per_frame": {
                "left": errors_left.tolist(),
                "right": errors_right.tolist(),
            }
        }
    
    def _update_mesh_sequence(
        self,
        mesh_sequence: Dict,
        frames: List[Dict],
        frame_keys: Optional[List],
        adjusted_trans: np.ndarray,
        contacts: np.ndarray
    ) -> Dict:
        """Update mesh_sequence with adjusted translations and contacts."""
        result = mesh_sequence.copy()
        
        updated_frames = []
        for i, frame in enumerate(frames):
            new_frame = frame.copy()
            
            # Update translation
            new_frame["pred_cam_t"] = adjusted_trans[i].tolist()
            
            if "smpl_params" in new_frame and isinstance(new_frame["smpl_params"], dict):
                new_frame["smpl_params"] = new_frame["smpl_params"].copy()
                new_frame["smpl_params"]["transl"] = adjusted_trans[i].tolist()
            
            # Add contact info
            new_frame["foot_contact"] = {
                "left": bool(contacts[i, 0]),
                "right": bool(contacts[i, 1]),
            }
            
            updated_frames.append(new_frame)
        
        if frame_keys is not None:
            result["frames"] = {k: v for k, v in zip(frame_keys, updated_frames)}
        else:
            result["frames"] = updated_frames
        
        return result
    
    def _process_heuristic(
        self,
        mesh_sequence: Dict,
        frames: List[Dict],
        frame_keys: Optional[List]
    ) -> Dict:
        """Fallback heuristic processing when no video available."""
        # Simple velocity-based detection from 3D joints
        T = len(frames)
        contacts = np.zeros((T, 2), dtype=bool)
        
        foot_positions = [[], []]
        for frame in frames:
            joints = extract_joints(frame)
            if joints is not None:
                left_idx = min(self.config.left_foot_idx, len(joints) - 1)
                right_idx = min(self.config.right_foot_idx, len(joints) - 1)
                foot_positions[0].append(joints[left_idx])
                foot_positions[1].append(joints[right_idx])
            else:
                foot_positions[0].append(np.zeros(3))
                foot_positions[1].append(np.zeros(3))
        
        foot_positions = [np.stack(fp) for fp in foot_positions]
        
        # Compute velocities
        for foot_idx in range(2):
            velocities = np.zeros(T)
            velocities[1:] = np.linalg.norm(np.diff(foot_positions[foot_idx], axis=0), axis=1)
            velocities[0] = velocities[1] if T > 1 else 0
            
            # Low velocity = contact (threshold in 3D units)
            contacts[:, foot_idx] = velocities < 0.02
        
        # Apply pinning
        root_trans = self._extract_translations(frames)
        
        if self.config.pin_feet and root_trans is not None:
            adjusted_trans, pin_events = enforce_contacts(
                frames, root_trans, contacts, self.config, self.log
            )
        else:
            adjusted_trans = root_trans
            pin_events = []
        
        result = self._update_mesh_sequence(
            mesh_sequence, frames, frame_keys, adjusted_trans, contacts
        )
        
        result["smart_groundlink"] = {
            "method": "heuristic",
            "pin_events": pin_events,
        }
        
        return result
    
    def _log_summary(self, contacts: np.ndarray, pin_events: List[Dict], verification_result: Optional[Dict] = None):
        """Log processing summary."""
        self.log.info("=" * 60)
        self.log.info("PROCESSING SUMMARY")
        self.log.info("=" * 60)
        
        T = contacts.shape[0]
        left_frames = contacts[:, 0].sum()
        right_frames = contacts[:, 1].sum()
        
        self.log.info(f"Total frames: {T}")
        self.log.info(f"Left contact frames: {left_frames} ({100*left_frames/T:.1f}%)")
        self.log.info(f"Right contact frames: {right_frames} ({100*right_frames/T:.1f}%)")
        self.log.info(f"Pin events: {len(pin_events)}")
        
        # Verification summary
        if verification_result:
            self.log.info("")
            self.log.info("=== VERIFICATION SUMMARY ===")
            quality = verification_result.get("quality", "unknown")
            quality_emoji = {"excellent": "✓✓", "good": "✓", "acceptable": "~", "poor": "✗"}.get(quality, "?")
            self.log.info(f"Quality: {quality_emoji} {quality.upper()}")
            self.log.info(f"  Overall error: {verification_result.get('overall_error_mean_px', 0):.1f}px mean")
            self.log.info(f"  Left foot:  {verification_result.get('left_error_mean_px', 0):.1f}px mean, {verification_result.get('left_error_max_px', 0):.1f}px max")
            self.log.info(f"  Right foot: {verification_result.get('right_error_mean_px', 0):.1f}px mean, {verification_result.get('right_error_max_px', 0):.1f}px max")
        
        # Timeline
        self.log.info("")
        self.log.info("CONTACT TIMELINE")
        self.log.info("-" * 50)
        self.log.info("Frame | L | R | Event")
        self.log.info("-" * 50)
        
        prev_left, prev_right = False, False
        shown_frames = set([0, T-1])
        
        for evt in pin_events:
            shown_frames.add(evt["frame"])
        
        for t in range(0, T, max(1, T // 20)):
            shown_frames.add(t)
        
        for t in sorted(shown_frames):
            if t >= T:
                continue
            
            left = contacts[t, 0]
            right = contacts[t, 1]
            
            events = []
            if t == 0:
                if left:
                    events.append("L_START")
                if right:
                    events.append("R_START")
            else:
                if left and not prev_left:
                    events.append("L_START")
                elif not left and prev_left:
                    events.append("L_END")
                if right and not prev_right:
                    events.append("R_START")
                elif not right and prev_right:
                    events.append("R_END")
            
            left_str = "████" if left else "····"
            right_str = "████" if right else "····"
            event_str = ", ".join(events) if events else ""
            
            self.log.info(f"{t:5d} | {left_str} | {right_str} | {event_str}")
            
            prev_left, prev_right = left, right
        
        self.log.info("-" * 50)
        
        # Timing summary
        self.log.info("")
        self.log.info("=== TIMING SUMMARY ===")
        total = sum(self.log.timings.values())
        for name, elapsed in self.log.timings.items():
            self.log.info(f"{name}: {elapsed:.3f}s")
        self.log.info(f"TOTAL: {total:.3f}s")


# =============================================================================
# ComfyUI Node
# =============================================================================

class SmartGroundLinkNode:
    """
    SmartGroundLink - Video-Guided Foot Pinning
    
    Combines TAPNet-based contact detection (using video as ground truth)
    with GroundLink-style physics pinning to eliminate foot sliding.
    
    How it works:
    1. Track feet in video using TAPNet
    2. Detect contacts (stationary feet = contact)
    3. During contacts, pin feet to fixed world positions
    4. Adjust body translation to keep feet locked
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE", {
                    "tooltip": "Mesh sequence from SAM3DBody"
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Video frames - HIGHLY RECOMMENDED for accurate detection"
                }),
                "velocity_threshold": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.5,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Pixels/frame below which foot is considered stationary"
                }),
                "min_contact_duration": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 30,
                    "tooltip": "Minimum frames for valid contact"
                }),
                "pin_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "How strongly to pin feet (0=off, 1=full lock)"
                }),
                "smooth_sigma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.5,
                    "tooltip": "Gaussian smoothing of result (0=none)"
                }),
                "chunk_size": ("INT", {
                    "default": 24,
                    "min": 8,
                    "max": 100,
                    "step": 4,
                    "tooltip": "Frames per tracking chunk (lower = less memory)"
                }),
                "log_level": (["normal", "verbose", "debug", "silent"], {
                    "default": "verbose",
                    "tooltip": "Logging verbosity"
                }),
            }
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE", "STRING", "FOOT_CONTACTS")
    RETURN_NAMES = ("mesh_sequence", "debug_info", "foot_contacts")
    FUNCTION = "process"
    CATEGORY = "SAM3DBody2abc/Processing"
    
    def process(
        self,
        mesh_sequence: Dict,
        images=None,
        velocity_threshold: float = 3.0,
        min_contact_duration: int = 3,
        pin_strength: float = 0.5,
        smooth_sigma: float = 1.0,
        chunk_size: int = 24,
        log_level: str = "verbose",
    ) -> Tuple[Dict, str, Dict]:
        """Process mesh sequence with video-guided foot pinning."""
        
        # Build config
        config = SmartGroundLinkConfig(
            velocity_threshold=velocity_threshold,
            min_contact_duration=min_contact_duration,
            pin_strength=pin_strength,
            smooth_sigma=smooth_sigma,
            chunk_size=chunk_size,
            log_level=log_level,
        )
        
        # Convert images to numpy
        video_frames = None
        if images is not None:
            if torch.is_tensor(images):
                video_frames = (images.cpu().numpy() * 255).astype(np.uint8)
            else:
                video_frames = np.array(images)
                if video_frames.max() <= 1.0:
                    video_frames = (video_frames * 255).astype(np.uint8)
        
        # Process
        processor = SmartGroundLinkProcessor(config)
        result = processor.process(mesh_sequence, video_frames)
        
        # Build debug info
        metadata = result.get("smart_groundlink", {})
        pin_events = metadata.get("pin_events", [])
        timings = metadata.get("timings", {})
        contacts_info = metadata.get("contacts_detected", {})
        
        debug_lines = [
            "=== SMART GROUNDLINK RESULTS ===",
            f"Method: {metadata.get('method', 'unknown')}",
            f"Pin strength: {pin_strength}",
            "",
            f"Left contact frames: {contacts_info.get('left', 0)}",
            f"Right contact frames: {contacts_info.get('right', 0)}",
            f"Pin events: {len(pin_events)}",
            "",
        ]
        
        # Add verification results
        verification = metadata.get("verification", {})
        if verification:
            quality = verification.get("quality", "unknown")
            quality_emoji = {"excellent": "✓✓", "good": "✓", "acceptable": "~", "poor": "✗"}.get(quality, "?")
            debug_lines.extend([
                "=== VERIFICATION vs VIDEO ===",
                f"Quality: {quality_emoji} {quality.upper()}",
                f"Overall error: {verification.get('overall_error_mean_px', 0):.1f}px mean",
                f"Left foot:  {verification.get('left_error_mean_px', 0):.1f}px mean, {verification.get('left_error_max_px', 0):.1f}px max",
                f"Right foot: {verification.get('right_error_mean_px', 0):.1f}px mean, {verification.get('right_error_max_px', 0):.1f}px max",
                "",
            ])
        
        for evt in pin_events[:20]:
            debug_lines.append(f"  Frame {evt['frame']}: {evt['foot']} {evt['event']}")
        
        if len(pin_events) > 20:
            debug_lines.append(f"  ... and {len(pin_events) - 20} more")
        
        debug_lines.extend(["", "=== TIMING ==="])
        for name, elapsed in timings.items():
            debug_lines.append(f"  {name}: {elapsed:.3f}s")
        
        debug_info = "\n".join(debug_lines)
        
        # Build foot_contacts output
        frames_data = result.get("frames", {})
        if isinstance(frames_data, dict):
            frame_keys = sorted(frames_data.keys())
            frames = [frames_data[k] for k in frame_keys]
        else:
            frames = list(frames_data)
        
        contacts_array = []
        events = []
        prev_left, prev_right = False, False
        
        for i, frame in enumerate(frames):
            fc = frame.get("foot_contact", {"left": False, "right": False})
            left = fc["left"]
            right = fc["right"]
            contacts_array.append([left, right])
            
            frame_events = []
            if left and not prev_left:
                frame_events.append("L_START")
            elif not left and prev_left:
                frame_events.append("L_END")
            if right and not prev_right:
                frame_events.append("R_START")
            elif not right and prev_right:
                frame_events.append("R_END")
            
            if frame_events:
                events.append({"frame": i, "events": frame_events})
            
            prev_left, prev_right = left, right
        
        foot_contacts = {
            "contacts": contacts_array,
            "method": "smart_groundlink",
            "events": events,
            "pin_events": pin_events,
            "config": {
                "velocity_threshold": velocity_threshold,
                "min_contact_duration": min_contact_duration,
                "pin_strength": pin_strength,
            }
        }
        
        return (result, debug_info, foot_contacts)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "SmartGroundLink": SmartGroundLinkNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartGroundLink": "🦶🦿 Smart GroundLink (Video + Pinning)",
}
