"""
Foot Tracker - TAPNet/TAPIR-based Foot Contact Detection
=========================================================

Uses TAPIR to track foot points (toe, ball, ankle) in 2D pixel space,
then detects ground contact based on:
1. Relative Y position between feet (lowest = ground reference)
2. 2D velocity (stationary foot = potential contact)

This approach is camera-motion invariant since it uses relative measurements
rather than absolute ground plane calibration.

Key advantages over skeleton-based detection:
- Tracks actual visible shoe/foot surface, not estimated skeleton joints
- Pixel-level precision
- Handles camera movement naturally
- Works with noisy pose estimation

Author: Claude (Anthropic)
Version: 1.0.0
License: Apache 2.0
"""

import numpy as np
import torch
import cv2
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from scipy.signal import savgol_filter

# Import logger
try:
    from ..lib.logger import log, set_module
    set_module("Foot Tracker")
except ImportError:
    class _FallbackLog:
        def info(self, msg): print(f"[Foot Tracker] {msg}")
        def debug(self, msg): pass
        def warn(self, msg): print(f"[Foot Tracker] WARN: {msg}")
        def error(self, msg): print(f"[Foot Tracker] ERROR: {msg}")
    log = _FallbackLog()

# Try to import TAPIR
TAPIR_AVAILABLE = False
try:
    from tapnet.torch import tapir_model
    TAPIR_AVAILABLE = True
    log.info("TAPIR (tapnet) available for foot tracking")
except ImportError:
    log.info("TAPIR not found - will use skeleton joints as fallback")


# Joint indices for SAM3DBody skeleton
JOINT_INDICES = {
    # Left foot (from confirmed indices)
    "left_ankle": 17,
    "left_ball": 15,
    "left_toe": 16,
    # Right foot
    "right_ankle": 14,
    "right_ball": 18,
    "right_toe": 19,
}


@dataclass
class FootPoint:
    """Tracked foot point data."""
    name: str
    x: float  # Pixel X
    y: float  # Pixel Y
    velocity: float  # Pixel velocity magnitude
    visible: bool


@dataclass
class FootState:
    """State of one foot."""
    toe: FootPoint
    ball: FootPoint
    ankle: FootPoint
    
    @property
    def lowest_y(self) -> float:
        """Get lowest Y (highest pixel value = lowest in image)."""
        return max(self.toe.y, self.ball.y, self.ankle.y)
    
    @property
    def avg_velocity(self) -> float:
        """Average velocity of visible points."""
        points = [self.toe, self.ball, self.ankle]
        visible = [p for p in points if p.visible]
        if not visible:
            return 0.0
        return sum(p.velocity for p in visible) / len(visible)
    
    @property
    def min_velocity(self) -> float:
        """Minimum velocity (most stationary point)."""
        points = [self.toe, self.ball, self.ankle]
        visible = [p for p in points if p.visible]
        if not visible:
            return 0.0
        return min(p.velocity for p in visible)


class FootTracker:
    """
    Track foot contact using TAPIR 2D point tracking.
    
    Pipeline:
    1. Get initial foot positions from skeleton joints (keypoints_2d)
    2. Use TAPIR to track these points through the video
    3. Apply Savitzky-Golay smoothing to reduce noise
    4. Detect contact using relative position + velocity
    """
    
    SMOOTHING_METHODS = ["None", "Savitzky-Golay", "Moving Average"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input video frames"
                }),
                "mesh_sequence": ("MESH_SEQUENCE", {
                    "tooltip": "SAM3DBody output with keypoints_2d for initial positions"
                }),
            },
            "optional": {
                # Tracking settings
                "use_tapir": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use TAPIR for tracking. Disable to use skeleton joints directly."
                }),
                "tapir_batch_frames": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 200,
                    "tooltip": "Process video in batches. Lower = less VRAM."
                }),
                
                # Smoothing
                "smoothing_method": (cls.SMOOTHING_METHODS, {
                    "default": "Savitzky-Golay",
                    "tooltip": "Temporal smoothing for tracked points"
                }),
                "smoothing_window": ("INT", {
                    "default": 7,
                    "min": 3,
                    "max": 21,
                    "step": 2,
                    "tooltip": "Smoothing window size (odd number)"
                }),
                
                # Contact detection thresholds
                "velocity_threshold": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.5,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Max velocity (pixels/frame) to count as stationary"
                }),
                "height_threshold": ("FLOAT", {
                    "default": 15.0,
                    "min": 1.0,
                    "max": 50.0,
                    "step": 1.0,
                    "tooltip": "Max Y difference (pixels) from lowest foot to count as grounded"
                }),
                "min_contact_frames": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Minimum consecutive frames for valid contact"
                }),
                
                # Debug
                "output_debug_video": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Output visualization"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MESH_SEQUENCE", "STRING")
    RETURN_NAMES = ("debug_overlay", "mesh_sequence", "status")
    FUNCTION = "track_feet"
    CATEGORY = "SAM3DBody2abc/Tracking"
    
    def __init__(self):
        self.tapir_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_tapir(self) -> bool:
        """Load TAPIR model if available."""
        if not TAPIR_AVAILABLE:
            return False
        
        if self.tapir_model is not None:
            return True
        
        try:
            import os
            checkpoint_paths = [
                "/workspace/ComfyUI/models/tapir/bootstapir_checkpoint_v2.pt",
                "/workspace/models/tapir/bootstapir_checkpoint_v2.pt",
                os.path.expanduser("~/models/tapir/bootstapir_checkpoint_v2.pt"),
                "bootstapir_checkpoint_v2.pt",
            ]
            
            checkpoint_path = None
            for path in checkpoint_paths:
                if os.path.exists(path):
                    checkpoint_path = path
                    break
            
            if checkpoint_path is None:
                log.warn("TAPIR checkpoint not found")
                return False
            
            self.tapir_model = tapir_model.TAPIR(pyramid_level=1)
            self.tapir_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.tapir_model.to(self.device)
            self.tapir_model.eval()
            
            log.info(f"TAPIR loaded from {checkpoint_path}")
            return True
        
        except Exception as e:
            log.warn(f"Error loading TAPIR: {e}")
            return False
    
    def _get_initial_foot_points(self, mesh_sequence: Dict, frame_idx: int = 0) -> Dict[str, np.ndarray]:
        """
        Get initial 2D foot positions from skeleton keypoints.
        
        Returns dict mapping joint name to (x, y) pixel coordinates.
        """
        frames = mesh_sequence.get("frames", {})
        if isinstance(frames, dict):
            frame_keys = sorted(frames.keys())
            frame_data = frames[frame_keys[frame_idx]] if frame_idx < len(frame_keys) else None
        else:
            frame_data = frames[frame_idx] if frame_idx < len(frames) else None
        
        if frame_data is None:
            raise ValueError(f"No frame data at index {frame_idx}")
        
        # Try keypoints_2d first (COCO format projected to 2D)
        keypoints_2d = frame_data.get("keypoints_2d")
        
        points = {}
        
        if keypoints_2d is not None:
            keypoints_2d = np.array(keypoints_2d)
            if keypoints_2d.ndim == 3:
                keypoints_2d = keypoints_2d[0]
            
            # Map COCO joints to our foot joints
            # COCO: 15=L_Ankle, 16=R_Ankle (no toe/ball in COCO)
            # We'll use ankle as reference and estimate toe/ball
            
            if len(keypoints_2d) >= 17:
                # Left foot - use ankle, estimate others
                l_ankle = keypoints_2d[15]  # COCO L_Ankle
                points["left_ankle"] = l_ankle[:2]
                # Estimate toe as below and forward of ankle
                points["left_toe"] = l_ankle[:2] + np.array([0, 20])  # 20px below
                points["left_ball"] = l_ankle[:2] + np.array([0, 15])
                
                # Right foot
                r_ankle = keypoints_2d[16]  # COCO R_Ankle
                points["right_ankle"] = r_ankle[:2]
                points["right_toe"] = r_ankle[:2] + np.array([0, 20])
                points["right_ball"] = r_ankle[:2] + np.array([0, 15])
        
        # Try joint_coords (SMPL-H format) with camera projection
        if not points:
            joint_coords = frame_data.get("joint_coords")
            pred_cam_t = frame_data.get("pred_cam_t")
            
            if joint_coords is not None:
                joint_coords = np.array(joint_coords)
                if joint_coords.ndim == 3:
                    joint_coords = joint_coords[0]
                
                # Project 3D joints to 2D (simplified)
                # For now, use X and Y directly (assumes orthographic-ish)
                img_size = mesh_sequence.get("image_size", [512, 512])
                H, W = img_size[0], img_size[1]
                
                for name, idx in JOINT_INDICES.items():
                    if idx < len(joint_coords):
                        # Simple projection: scale and center
                        x_3d, y_3d, z_3d = joint_coords[idx]
                        # Approximate 2D projection
                        x_2d = W / 2 + x_3d * 200  # Scale factor
                        y_2d = H / 2 - y_3d * 200  # Flip Y
                        points[name] = np.array([x_2d, y_2d])
        
        if not points:
            raise ValueError("Could not extract foot positions from mesh_sequence")
        
        return points
    
    def _track_with_tapir(
        self,
        images: np.ndarray,
        init_points: Dict[str, np.ndarray],
        batch_frames: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        Track foot points using TAPIR.
        
        Args:
            images: [N, H, W, 3] video frames
            init_points: Dict of initial (x, y) positions per joint
            batch_frames: Batch size for processing
        
        Returns:
            Dict mapping joint name to [N, 2] array of tracked positions
        """
        import gc
        
        N, H, W, _ = images.shape
        num_points = len(init_points)
        point_names = list(init_points.keys())
        
        log.info(f"TAPIR tracking {num_points} foot points over {N} frames")
        
        # Prepare initial query points
        init_xy = np.array([init_points[name] for name in point_names])  # [num_points, 2]
        
        # Result arrays
        all_tracks = {name: np.zeros((N, 2)) for name in point_names}
        all_visible = {name: np.ones(N, dtype=bool) for name in point_names}
        
        # Process in batches
        num_batches = (N + batch_frames - 1) // batch_frames
        
        for batch_idx in range(num_batches):
            start_frame = batch_idx * batch_frames
            end_frame = min(start_frame + batch_frames, N)
            batch_size = end_frame - start_frame
            
            log.info(f"TAPIR batch {batch_idx + 1}/{num_batches} (frames {start_frame}-{end_frame - 1})")
            
            batch_images = images[start_frame:end_frame]
            
            # Prepare video tensor
            video = torch.from_numpy(batch_images).float()
            if video.max() > 1.0:
                video = video / 255.0
            video = video * 2 - 1  # Normalize to [-1, 1]
            video = video.unsqueeze(0).to(self.device)  # [1, T, H, W, C]
            
            # Query points for this batch
            if batch_idx == 0:
                query_xy = init_xy
            else:
                # Use last known position from previous batch
                query_xy = np.array([all_tracks[name][start_frame - 1] for name in point_names])
            
            # Query points: [B, N, 3] where each is [t, y, x]
            query_points = torch.zeros((1, num_points, 3), dtype=torch.float32, device=self.device)
            query_points[0, :, 0] = 0  # Query frame (relative to batch)
            query_points[0, :, 1] = torch.from_numpy(query_xy[:, 1].astype(np.float32))  # y
            query_points[0, :, 2] = torch.from_numpy(query_xy[:, 0].astype(np.float32))  # x
            
            try:
                with torch.no_grad():
                    outputs = self.tapir_model(video, query_points)
                
                tracks = outputs['tracks'][0].float().cpu().numpy()  # [num_points, T, 2]
                occlusion = torch.sigmoid(outputs['occlusion'][0].float()).cpu().numpy()
                
                # Store results
                for i, name in enumerate(point_names):
                    all_tracks[name][start_frame:end_frame] = tracks[i, :batch_size, :]
                    all_visible[name][start_frame:end_frame] = occlusion[i, :batch_size] < 0.5
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log.warn(f"GPU OOM - falling back to skeleton for this batch")
                    # Just copy the query points
                    for i, name in enumerate(point_names):
                        all_tracks[name][start_frame:end_frame] = query_xy[i]
                else:
                    raise e
            finally:
                del video, query_points
                if 'outputs' in locals():
                    del outputs
                torch.cuda.empty_cache()
                gc.collect()
        
        return all_tracks
    
    def _track_from_skeleton(
        self,
        mesh_sequence: Dict,
        num_frames: int
    ) -> Dict[str, np.ndarray]:
        """
        Fallback: Extract foot positions directly from skeleton joints.
        """
        frames = mesh_sequence.get("frames", {})
        if isinstance(frames, dict):
            frame_keys = sorted(frames.keys())
            frame_list = [frames[k] for k in frame_keys]
        else:
            frame_list = frames
        
        tracks = {name: np.zeros((num_frames, 2)) for name in JOINT_INDICES.keys()}
        
        for i, frame_data in enumerate(frame_list):
            if i >= num_frames:
                break
            
            keypoints_2d = frame_data.get("keypoints_2d")
            if keypoints_2d is not None:
                keypoints_2d = np.array(keypoints_2d)
                if keypoints_2d.ndim == 3:
                    keypoints_2d = keypoints_2d[0]
                
                if len(keypoints_2d) >= 17:
                    # COCO format
                    l_ankle = keypoints_2d[15][:2]
                    r_ankle = keypoints_2d[16][:2]
                    
                    tracks["left_ankle"][i] = l_ankle
                    tracks["left_toe"][i] = l_ankle + np.array([0, 20])
                    tracks["left_ball"][i] = l_ankle + np.array([0, 15])
                    tracks["right_ankle"][i] = r_ankle
                    tracks["right_toe"][i] = r_ankle + np.array([0, 20])
                    tracks["right_ball"][i] = r_ankle + np.array([0, 15])
        
        return tracks
    
    def _smooth_tracks(
        self,
        tracks: Dict[str, np.ndarray],
        method: str,
        window: int
    ) -> Dict[str, np.ndarray]:
        """Apply temporal smoothing to all tracks."""
        if method == "None":
            return tracks
        
        window = window if window % 2 == 1 else window + 1
        smoothed = {}
        
        for name, track in tracks.items():
            if len(track) < window:
                smoothed[name] = track
                continue
            
            smoothed_track = track.copy()
            
            if method == "Savitzky-Golay":
                poly_order = min(3, window - 1)
                smoothed_track[:, 0] = savgol_filter(track[:, 0], window, poly_order)
                smoothed_track[:, 1] = savgol_filter(track[:, 1], window, poly_order)
            
            elif method == "Moving Average":
                kernel = np.ones(window) / window
                smoothed_track[:, 0] = np.convolve(track[:, 0], kernel, mode='same')
                smoothed_track[:, 1] = np.convolve(track[:, 1], kernel, mode='same')
            
            smoothed[name] = smoothed_track
        
        return smoothed
    
    def _compute_velocities(self, tracks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute velocity magnitude for each tracked point."""
        velocities = {}
        
        for name, track in tracks.items():
            vel = np.zeros(len(track))
            vel[1:] = np.linalg.norm(track[1:] - track[:-1], axis=1)
            vel[0] = vel[1] if len(vel) > 1 else 0
            velocities[name] = vel
        
        return velocities
    
    def _detect_contact(
        self,
        tracks: Dict[str, np.ndarray],
        velocities: Dict[str, np.ndarray],
        velocity_threshold: float,
        height_threshold: float,
        min_contact_frames: int
    ) -> Tuple[List[str], List[FootState], List[FootState]]:
        """
        Detect foot contact using relative position and velocity.
        
        Logic:
        1. Find lowest foot each frame (highest Y in image coords)
        2. Foot is grounded if:
           - It's the lowest (or within threshold of lowest)
           - AND velocity is below threshold (stationary)
        """
        num_frames = len(tracks["left_toe"])
        contact_states = []
        left_states = []
        right_states = []
        
        for i in range(num_frames):
            # Build foot states
            left = FootState(
                toe=FootPoint("left_toe", tracks["left_toe"][i, 0], tracks["left_toe"][i, 1],
                             velocities["left_toe"][i], True),
                ball=FootPoint("left_ball", tracks["left_ball"][i, 0], tracks["left_ball"][i, 1],
                              velocities["left_ball"][i], True),
                ankle=FootPoint("left_ankle", tracks["left_ankle"][i, 0], tracks["left_ankle"][i, 1],
                               velocities["left_ankle"][i], True),
            )
            
            right = FootState(
                toe=FootPoint("right_toe", tracks["right_toe"][i, 0], tracks["right_toe"][i, 1],
                             velocities["right_toe"][i], True),
                ball=FootPoint("right_ball", tracks["right_ball"][i, 0], tracks["right_ball"][i, 1],
                              velocities["right_ball"][i], True),
                ankle=FootPoint("right_ankle", tracks["right_ankle"][i, 0], tracks["right_ankle"][i, 1],
                               velocities["right_ankle"][i], True),
            )
            
            left_states.append(left)
            right_states.append(right)
            
            # Find ground reference (lowest Y = highest pixel value)
            ground_y = max(left.lowest_y, right.lowest_y)
            
            # Check if each foot is grounded
            left_height_ok = (ground_y - left.lowest_y) <= height_threshold
            left_velocity_ok = left.min_velocity <= velocity_threshold
            left_grounded = left_height_ok and left_velocity_ok
            
            right_height_ok = (ground_y - right.lowest_y) <= height_threshold
            right_velocity_ok = right.min_velocity <= velocity_threshold
            right_grounded = right_height_ok and right_velocity_ok
            
            if left_grounded and right_grounded:
                contact_states.append("both")
            elif left_grounded:
                contact_states.append("left")
            elif right_grounded:
                contact_states.append("right")
            else:
                contact_states.append("none")
        
        # Apply min_contact_frames filter
        if min_contact_frames > 1:
            contact_states = self._filter_contact_frames(contact_states, min_contact_frames)
        
        return contact_states, left_states, right_states
    
    def _filter_contact_frames(self, states: List[str], min_frames: int) -> List[str]:
        """Filter out short contact periods."""
        result = states.copy()
        n = len(states)
        
        # For each unique state, check if it persists long enough
        i = 0
        while i < n:
            state = states[i]
            
            # Count consecutive frames with this state
            j = i
            while j < n and states[j] == state:
                j += 1
            
            duration = j - i
            
            # If too short and not "none", convert to "none"
            if duration < min_frames and state != "none":
                for k in range(i, j):
                    result[k] = "none"
            
            i = j
        
        return result
    
    def _draw_debug_overlay(
        self,
        images: np.ndarray,
        tracks: Dict[str, np.ndarray],
        velocities: Dict[str, np.ndarray],
        contact_states: List[str],
        left_states: List[FootState],
        right_states: List[FootState],
        velocity_threshold: float,
        height_threshold: float,
    ) -> np.ndarray:
        """Draw debug visualization."""
        N, H, W, _ = images.shape
        output = []
        
        # Colors (BGR for OpenCV)
        COLOR_LEFT = (255, 100, 100)    # Blue-ish
        COLOR_RIGHT = (100, 100, 255)   # Red-ish
        COLOR_GROUNDED = (0, 255, 0)    # Green
        COLOR_AIRBORNE = (0, 0, 255)    # Red
        COLOR_TEXT = (255, 255, 255)
        COLOR_BG = (0, 0, 0)
        
        for i in range(N):
            # Convert frame
            if images[i].max() <= 1.0:
                frame = (images[i] * 255).astype(np.uint8)
            else:
                frame = images[i].astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            contact = contact_states[i]
            left = left_states[i]
            right = right_states[i]
            
            # Draw foot points
            for name, color in [("left", COLOR_LEFT), ("right", COLOR_RIGHT)]:
                state = left if name == "left" else right
                
                for point in [state.toe, state.ball, state.ankle]:
                    x, y = int(point.x), int(point.y)
                    
                    # Point circle
                    cv2.circle(frame, (x, y), 5, color, -1)
                    cv2.circle(frame, (x, y), 6, COLOR_TEXT, 1)
                    
                    # Velocity vector
                    if point.velocity > 0:
                        scale = min(point.velocity * 3, 30)
                        # Just draw as circle size for now
                        cv2.circle(frame, (x, y), int(5 + scale / 3), color, 1)
            
            # Draw info panel
            panel_h = 200
            cv2.rectangle(frame, (5, 5), (320, panel_h), COLOR_BG, -1)
            cv2.rectangle(frame, (5, 5), (320, panel_h), COLOR_TEXT, 1)
            
            y_off = 25
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Frame and contact state
            cv2.putText(frame, f"Frame: {i}", (15, y_off), font, 0.5, COLOR_TEXT, 1)
            y_off += 20
            
            contact_color = COLOR_GROUNDED if contact != "none" else COLOR_AIRBORNE
            cv2.putText(frame, f"Contact: {contact.upper()}", (15, y_off), font, 0.6, contact_color, 2)
            y_off += 25
            
            # Ground reference
            ground_y = max(left.lowest_y, right.lowest_y)
            cv2.putText(frame, f"Ground Y: {ground_y:.1f}px", (15, y_off), font, 0.4, (200, 200, 200), 1)
            y_off += 18
            
            # Left foot
            left_grounded = contact in ["left", "both"]
            left_color = COLOR_GROUNDED if left_grounded else COLOR_AIRBORNE
            cv2.putText(frame, f"LEFT: {'GND' if left_grounded else 'AIR'}", (15, y_off), font, 0.5, left_color, 1)
            y_off += 18
            cv2.putText(frame, f"  lowest_y={left.lowest_y:.1f} vel={left.min_velocity:.1f}", 
                       (15, y_off), font, 0.35, (180, 180, 180), 1)
            y_off += 15
            
            # Right foot
            right_grounded = contact in ["right", "both"]
            right_color = COLOR_GROUNDED if right_grounded else COLOR_AIRBORNE
            cv2.putText(frame, f"RIGHT: {'GND' if right_grounded else 'AIR'}", (15, y_off), font, 0.5, right_color, 1)
            y_off += 18
            cv2.putText(frame, f"  lowest_y={right.lowest_y:.1f} vel={right.min_velocity:.1f}", 
                       (15, y_off), font, 0.35, (180, 180, 180), 1)
            y_off += 18
            
            # Thresholds
            cv2.putText(frame, f"Thresholds: vel<{velocity_threshold:.1f}px/f height<{height_threshold:.1f}px", 
                       (15, y_off), font, 0.35, (150, 150, 150), 1)
            
            # Convert back to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output.append(frame)
        
        return np.stack(output, axis=0)
    
    def track_feet(
        self,
        images: torch.Tensor,
        mesh_sequence: Dict,
        use_tapir: bool = True,
        tapir_batch_frames: int = 50,
        smoothing_method: str = "Savitzky-Golay",
        smoothing_window: int = 7,
        velocity_threshold: float = 3.0,
        height_threshold: float = 15.0,
        min_contact_frames: int = 2,
        output_debug_video: bool = True,
    ) -> Tuple[torch.Tensor, Dict, str]:
        """Main tracking function."""
        
        # Convert images to numpy
        images_np = images.cpu().numpy()
        if images_np.max() <= 1.0:
            images_np_uint8 = (images_np * 255).astype(np.uint8)
        else:
            images_np_uint8 = images_np.astype(np.uint8)
        
        N, H, W, _ = images_np.shape
        log.info(f"Processing {N} frames at {W}x{H}")
        
        # Get initial foot positions
        try:
            init_points = self._get_initial_foot_points(mesh_sequence, frame_idx=0)
            log.info(f"Initial foot points: {list(init_points.keys())}")
        except Exception as e:
            log.error(f"Failed to get initial foot positions: {e}")
            raise
        
        # Track points
        tapir_loaded = use_tapir and self._load_tapir()
        
        if tapir_loaded:
            log.info("Using TAPIR for foot tracking")
            tracks = self._track_with_tapir(images_np_uint8, init_points, tapir_batch_frames)
        else:
            log.info("Using skeleton joints for foot tracking (TAPIR not available)")
            tracks = self._track_from_skeleton(mesh_sequence, N)
        
        # Apply smoothing
        log.info(f"Applying {smoothing_method} smoothing (window={smoothing_window})")
        tracks = self._smooth_tracks(tracks, smoothing_method, smoothing_window)
        
        # Compute velocities
        velocities = self._compute_velocities(tracks)
        
        # Detect contact
        contact_states, left_states, right_states = self._detect_contact(
            tracks, velocities, velocity_threshold, height_threshold, min_contact_frames
        )
        
        # Statistics
        both_count = contact_states.count("both")
        left_count = contact_states.count("left")
        right_count = contact_states.count("right")
        none_count = contact_states.count("none")
        
        status = (
            f"Foot Contact Detection Results\n"
            f"===============================\n"
            f"Tracker: {'TAPIR' if tapir_loaded else 'Skeleton'}\n"
            f"Smoothing: {smoothing_method} (window={smoothing_window})\n"
            f"Thresholds: velocity<{velocity_threshold}px/f, height<{height_threshold}px\n"
            f"\n"
            f"Contact Statistics ({N} frames):\n"
            f"  Both feet: {both_count} ({100*both_count/N:.1f}%)\n"
            f"  Left only: {left_count} ({100*left_count/N:.1f}%)\n"
            f"  Right only: {right_count} ({100*right_count/N:.1f}%)\n"
            f"  Airborne: {none_count} ({100*none_count/N:.1f}%)\n"
        )
        log.info(f"Contact: both={both_count}, left={left_count}, right={right_count}, air={none_count}")
        
        # Create debug video
        if output_debug_video:
            debug_frames = self._draw_debug_overlay(
                images_np, tracks, velocities, contact_states,
                left_states, right_states, velocity_threshold, height_threshold
            )
            debug_video = torch.from_numpy(debug_frames).float() / 255.0
        else:
            debug_video = images
        
        # Update mesh_sequence with contact data
        import copy
        result_sequence = copy.deepcopy(mesh_sequence)
        result_sequence["foot_contact"] = {
            "contact_states": contact_states,
            "tracker": "tapir" if tapir_loaded else "skeleton",
            "thresholds": {
                "velocity": velocity_threshold,
                "height": height_threshold,
                "min_frames": min_contact_frames,
            },
            "smoothing": {
                "method": smoothing_method,
                "window": smoothing_window,
            },
            "tracks_2d": {name: track.tolist() for name, track in tracks.items()},
            "velocities": {name: vel.tolist() for name, vel in velocities.items()},
        }
        
        return (debug_video, result_sequence, status)


# Node registration
NODE_CLASS_MAPPINGS = {
    "FootTracker": FootTracker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FootTracker": "🦶 Foot Tracker (TAPNet)",
}
