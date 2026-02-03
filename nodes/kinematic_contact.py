"""
Kinematic Contact Detector
==========================

Detect foot contacts using pure kinematics - no ML models, no tracking.
Based on biomechanical principles:
  1. Contact = foot is flat (ankle, toe, heel at same height)
  2. Contact = pelvis moving away from planted foot
  3. Confidence = reprojection error (3D→2D consistency)

Works universally for walking, running, sprinting - any bipedal gait.

Author: Claude (Anthropic)
Version: 1.0.0
License: Apache 2.0
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class KinematicContactConfig:
    """Configuration for kinematic contact detection."""
    
    # Joint indices (configurable for different skeleton formats)
    pelvis_idx: int = 0
    
    # Left foot joints
    l_ankle_idx: int = 15
    l_toe_idx: int = 17
    l_heel_idx: int = 19
    
    # Right foot joints  
    r_ankle_idx: int = 16
    r_toe_idx: int = 18
    r_heel_idx: int = 20
    
    # Hip/Torso joints for visualization
    l_hip_idx: int = 11
    r_hip_idx: int = 12
    spine_idx: int = 6
    
    # Detection thresholds
    height_threshold: float = 0.03  # meters - tolerance for "same height"
    distance_threshold: float = 0.001  # meters - minimum pelvis movement
    
    # Smoothing
    min_contact_frames: int = 2  # Minimum frames for valid contact
    
    # Visualization
    show_feet: bool = True
    show_hips: bool = True
    show_torso: bool = True
    show_contact_status: bool = True
    show_trajectories: bool = True
    trajectory_length: int = 30  # frames of trajectory to show
    
    # Colors (BGR for OpenCV)
    color_left_contact: Tuple[int, int, int] = (0, 255, 0)      # Green
    color_left_flight: Tuple[int, int, int] = (0, 255, 255)     # Yellow
    color_right_contact: Tuple[int, int, int] = (0, 200, 0)     # Dark Green
    color_right_flight: Tuple[int, int, int] = (0, 200, 200)    # Dark Yellow
    color_hip: Tuple[int, int, int] = (255, 0, 255)             # Magenta
    color_torso: Tuple[int, int, int] = (255, 255, 0)           # Cyan
    color_pelvis: Tuple[int, int, int] = (0, 0, 255)            # Red
    
    joint_radius: int = 6
    trajectory_thickness: int = 2


# =============================================================================
# Logger
# =============================================================================

class KinematicLogger:
    """Simple logger for Kinematic Contact Detector."""
    
    def __init__(self, level: str = "normal"):
        self.level = level
        self.levels = {"silent": 0, "normal": 1, "verbose": 2, "debug": 3}
    
    def _log(self, msg: str, min_level: str = "normal"):
        if self.levels.get(self.level, 1) >= self.levels.get(min_level, 1):
            print(f"[KinematicContact] {msg}")
    
    def info(self, msg): self._log(msg, "normal")
    def verbose(self, msg): self._log(msg, "verbose")
    def debug(self, msg): self._log(f"DEBUG: {msg}", "debug")
    def warning(self, msg): self._log(f"⚠ WARNING: {msg}", "normal")
    def error(self, msg): self._log(f"✗ ERROR: {msg}", "normal")


# =============================================================================
# Core Detection
# =============================================================================

class KinematicContactDetector:
    """
    Detect foot contacts using pure kinematics.
    No ML models, no tracking - just geometry and physics.
    """
    
    def __init__(self, config: KinematicContactConfig, log: KinematicLogger):
        self.config = config
        self.log = log
    
    def detect(self, frames: List[Dict], global_intrinsics: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Detect contacts for all frames.
        
        Args:
            frames: List of frame data with pred_keypoints_3d, pred_keypoints_2d
            global_intrinsics: Optional dict with focal_length, cx, cy, width, height
        
        Returns:
            contacts: (T, 2) boolean array [left, right]
            confidence: (T, 2) float array [0-1]
            debug_data: Dictionary with detailed per-frame info
        """
        T = len(frames)
        contacts = np.zeros((T, 2), dtype=bool)
        confidence = np.zeros((T, 2), dtype=np.float32)
        
        # Store global intrinsics for use in reprojection
        self.global_intrinsics = global_intrinsics or {}
        
        # Debug data storage
        debug_data = {
            "foot_heights": {"left": [], "right": []},
            "foot_flat": {"left": [], "right": []},
            "foot_to_pelvis": {"left": [], "right": []},
            "pelvis_moving_away": {"left": [], "right": []},
            "reproj_errors": {"left": [], "right": []},
            "pelvis_positions": [],
            "foot_positions": {"left": [], "right": []},
            "hip_positions": {"left": [], "right": []},
            "events": [],
        }
        
        prev_foot_to_pelvis = [None, None]
        prev_contacts = [False, False]
        
        for t, frame in enumerate(frames):
            joints_3d = self._get_joints_3d(frame)
            joints_2d = self._get_joints_2d(frame)
            
            if joints_3d is None:
                self.log.warning(f"Frame {t}: No 3D joints found")
                continue
            
            # Get pelvis position
            pelvis_idx = min(self.config.pelvis_idx, len(joints_3d) - 1)
            pelvis = joints_3d[pelvis_idx]
            debug_data["pelvis_positions"].append(pelvis.tolist())
            
            # Get hip positions for debug
            l_hip_idx = min(self.config.l_hip_idx, len(joints_3d) - 1)
            r_hip_idx = min(self.config.r_hip_idx, len(joints_3d) - 1)
            debug_data["hip_positions"]["left"].append(joints_3d[l_hip_idx].tolist())
            debug_data["hip_positions"]["right"].append(joints_3d[r_hip_idx].tolist())
            
            # Process each foot
            for foot_idx, side in enumerate(["left", "right"]):
                # Get foot joint indices
                if side == "left":
                    ankle_idx = self.config.l_ankle_idx
                    toe_idx = self.config.l_toe_idx
                    heel_idx = self.config.l_heel_idx
                else:
                    ankle_idx = self.config.r_ankle_idx
                    toe_idx = self.config.r_toe_idx
                    heel_idx = self.config.r_heel_idx
                
                # Clamp indices to valid range
                n_joints = len(joints_3d)
                ankle_idx = min(ankle_idx, n_joints - 1)
                toe_idx = min(toe_idx, n_joints - 1)
                heel_idx = min(heel_idx, n_joints - 1)
                
                ankle = joints_3d[ankle_idx]
                toe = joints_3d[toe_idx]
                heel = joints_3d[heel_idx]
                
                foot_center = (ankle + toe + heel) / 3
                debug_data["foot_positions"][side].append(foot_center.tolist())
                
                # Condition 1: Foot is flat (all joints at same height)
                heights = [ankle[1], toe[1], heel[1]]  # Y is vertical
                height_range = max(heights) - min(heights)
                foot_flat = height_range < self.config.height_threshold
                
                debug_data["foot_heights"][side].append({
                    "ankle": float(ankle[1]),
                    "toe": float(toe[1]),
                    "heel": float(heel[1]),
                    "range": float(height_range),
                })
                debug_data["foot_flat"][side].append(foot_flat)
                
                # Condition 2: Pelvis moving away from foot
                foot_to_pelvis = np.linalg.norm(pelvis - foot_center)
                debug_data["foot_to_pelvis"][side].append(float(foot_to_pelvis))
                
                if prev_foot_to_pelvis[foot_idx] is not None:
                    distance_delta = foot_to_pelvis - prev_foot_to_pelvis[foot_idx]
                    pelvis_moving_away = distance_delta > self.config.distance_threshold
                else:
                    pelvis_moving_away = False
                    distance_delta = 0
                
                debug_data["pelvis_moving_away"][side].append({
                    "moving_away": pelvis_moving_away,
                    "delta": float(distance_delta),
                })
                
                prev_foot_to_pelvis[foot_idx] = foot_to_pelvis
                
                # Contact = flat foot + pelvis moving away
                is_contact = foot_flat and pelvis_moving_away
                contacts[t, foot_idx] = is_contact
                
                # Confidence from reprojection error
                if joints_2d is not None:
                    reproj_error = self._compute_reproj_error(
                        frame, 
                        [ankle, toe, heel],
                        joints_2d,
                        [ankle_idx, toe_idx, heel_idx]
                    )
                    # Convert error to confidence (0-1)
                    # <5px = 1.0, >20px = 0.0
                    conf = np.clip(1.0 - (reproj_error - 5) / 15, 0, 1)
                    confidence[t, foot_idx] = conf
                    debug_data["reproj_errors"][side].append(float(reproj_error))
                else:
                    confidence[t, foot_idx] = 0.5  # Unknown confidence
                    debug_data["reproj_errors"][side].append(None)
                
                # Track events
                if is_contact and not prev_contacts[foot_idx]:
                    debug_data["events"].append({
                        "frame": t,
                        "foot": side,
                        "event": "contact_start",
                    })
                elif not is_contact and prev_contacts[foot_idx]:
                    debug_data["events"].append({
                        "frame": t,
                        "foot": side,
                        "event": "contact_end",
                    })
                
                prev_contacts[foot_idx] = is_contact
        
        # Post-process: filter short contacts
        contacts = self._filter_short_contacts(contacts)
        
        return contacts, confidence, debug_data
    
    def _get_joints_3d(self, frame: Dict) -> Optional[np.ndarray]:
        """Extract 3D joints from frame."""
        for key in ["pred_keypoints_3d", "keypoints_3d", "joints_3d", "joint_coords"]:
            if key in frame:
                joints = np.array(frame[key])
                while joints.ndim > 2:
                    joints = joints[0]
                return joints
        return None
    
    def _get_joints_2d(self, frame: Dict) -> Optional[np.ndarray]:
        """Extract 2D joints from frame."""
        for key in ["pred_keypoints_2d", "keypoints_2d", "joints_2d"]:
            if key in frame:
                joints = np.array(frame[key])
                while joints.ndim > 2:
                    joints = joints[0]
                return joints
        return None
    
    def _compute_reproj_error(
        self,
        frame: Dict,
        joints_3d: List[np.ndarray],
        joints_2d: np.ndarray,
        indices: List[int]
    ) -> float:
        """Compute reprojection error for given joints."""
        # Get intrinsics - try frame first, then global, then defaults
        focal = frame.get("focal_length")
        if focal is None:
            focal = self.global_intrinsics.get("focal_length")
        if focal is None:
            focal = 1000.0
        
        cx = frame.get("cx")
        if cx is None:
            cx = self.global_intrinsics.get("cx")
        
        cy = frame.get("cy")
        if cy is None:
            cy = self.global_intrinsics.get("cy")
        
        # If cx/cy still None, try to get from image_size
        if cx is None or cy is None:
            image_size = frame.get("image_size") or self.global_intrinsics.get("image_size")
            width = self.global_intrinsics.get("width")
            height = self.global_intrinsics.get("height")
            
            if image_size is not None and len(image_size) >= 2:
                cx = image_size[1] / 2 if cx is None else cx  # width / 2
                cy = image_size[0] / 2 if cy is None else cy  # height / 2
            elif width is not None and height is not None:
                cx = width / 2 if cx is None else cx
                cy = height / 2 if cy is None else cy
            else:
                # Final fallback - assume 1280x720
                cx = 640.0 if cx is None else cx
                cy = 360.0 if cy is None else cy
        
        trans = frame.get("pred_cam_t")
        if trans is None:
            trans = [0, 0, 5]
        trans = np.array(trans)
        
        errors = []
        for j3d, idx in zip(joints_3d, indices):
            if idx >= len(joints_2d):
                continue
            
            j2d_obs = joints_2d[idx]
            
            # Project 3D to 2D
            world = j3d + trans
            z = max(world[2], 0.1)
            proj_x = focal * world[0] / z + cx
            proj_y = focal * world[1] / z + cy
            
            error = np.sqrt((proj_x - j2d_obs[0])**2 + (proj_y - j2d_obs[1])**2)
            errors.append(error)
        
        return np.mean(errors) if errors else 0.0
    
    def _filter_short_contacts(self, contacts: np.ndarray) -> np.ndarray:
        """Filter out contacts shorter than minimum duration."""
        T = contacts.shape[0]
        min_frames = self.config.min_contact_frames
        
        for foot_idx in range(2):
            foot_contacts = contacts[:, foot_idx].copy()
            
            # Find contact segments
            in_contact = False
            segment_start = 0
            
            for t in range(T + 1):
                is_contact = foot_contacts[t] if t < T else False
                
                if is_contact and not in_contact:
                    segment_start = t
                    in_contact = True
                elif not is_contact and in_contact:
                    segment_length = t - segment_start
                    if segment_length < min_frames:
                        foot_contacts[segment_start:t] = False
                    in_contact = False
            
            contacts[:, foot_idx] = foot_contacts
        
        return contacts


# =============================================================================
# Foot Stabilization
# =============================================================================

class FootStabilizer:
    """
    Stabilize foot positions during detected contacts.
    Adjusts root translation to pin feet without IK.
    """
    
    def __init__(self, config: KinematicContactConfig, log: KinematicLogger):
        self.config = config
        self.log = log
    
    def stabilize(
        self,
        frames: List[Dict],
        contacts: np.ndarray,
        pin_strength: float = 0.8
    ) -> Tuple[List[Dict], Dict]:
        """
        Stabilize foot positions during contacts.
        
        Args:
            frames: Frame data
            contacts: (T, 2) contact array
            pin_strength: How strongly to pin (0-1)
        
        Returns:
            stabilized_frames: Updated frames
            stabilization_info: Debug info
        """
        T = len(frames)
        stabilized_frames = []
        
        pin_positions = [None, None]  # Current pin position for each foot
        pin_events = []
        total_adjustments = []
        
        for t, frame in enumerate(frames):
            new_frame = frame.copy()
            joints_3d = self._get_joints_3d(frame)
            
            if joints_3d is None:
                stabilized_frames.append(new_frame)
                continue
            
            trans = np.array(frame.get("pred_cam_t", [0, 0, 5]))
            adjustment = np.zeros(3)
            
            for foot_idx, side in enumerate(["left", "right"]):
                # Get foot center
                if side == "left":
                    indices = [self.config.l_ankle_idx, self.config.l_toe_idx, self.config.l_heel_idx]
                else:
                    indices = [self.config.r_ankle_idx, self.config.r_toe_idx, self.config.r_heel_idx]
                
                n_joints = len(joints_3d)
                indices = [min(i, n_joints - 1) for i in indices]
                foot_joints = [joints_3d[i] for i in indices]
                foot_center = np.mean(foot_joints, axis=0)
                foot_world = foot_center + trans
                
                if contacts[t, foot_idx]:
                    if pin_positions[foot_idx] is None:
                        # Start of contact - set pin position
                        pin_positions[foot_idx] = foot_world.copy()
                        pin_events.append({
                            "frame": t,
                            "foot": side,
                            "event": "pin_start",
                            "position": foot_world.tolist(),
                        })
                    else:
                        # During contact - calculate adjustment to keep foot pinned
                        foot_adjustment = pin_positions[foot_idx] - foot_world
                        
                        # Only adjust horizontally (X, Z), not vertically (Y)
                        foot_adjustment[1] = 0
                        
                        adjustment += foot_adjustment * pin_strength
                else:
                    if pin_positions[foot_idx] is not None:
                        # End of contact
                        pin_events.append({
                            "frame": t,
                            "foot": side,
                            "event": "pin_end",
                            "position": pin_positions[foot_idx].tolist(),
                        })
                        pin_positions[foot_idx] = None
            
            # Average adjustment if both feet are pinned
            if contacts[t, 0] and contacts[t, 1]:
                adjustment /= 2
            
            # Apply adjustment
            new_trans = trans + adjustment
            new_frame["pred_cam_t"] = new_trans.tolist()
            
            if "smpl_params" in new_frame and isinstance(new_frame["smpl_params"], dict):
                new_frame["smpl_params"] = new_frame["smpl_params"].copy()
                new_frame["smpl_params"]["transl"] = new_trans.tolist()
            
            # Add contact info to frame
            new_frame["foot_contact"] = {
                "left": bool(contacts[t, 0]),
                "right": bool(contacts[t, 1]),
            }
            
            total_adjustments.append(np.linalg.norm(adjustment))
            stabilized_frames.append(new_frame)
        
        stabilization_info = {
            "pin_events": pin_events,
            "total_adjustments": total_adjustments,
            "avg_adjustment": np.mean(total_adjustments) if total_adjustments else 0,
            "max_adjustment": np.max(total_adjustments) if total_adjustments else 0,
        }
        
        return stabilized_frames, stabilization_info
    
    def _get_joints_3d(self, frame: Dict) -> Optional[np.ndarray]:
        """Extract 3D joints from frame."""
        for key in ["pred_keypoints_3d", "keypoints_3d", "joints_3d", "joint_coords"]:
            if key in frame:
                joints = np.array(frame[key])
                while joints.ndim > 2:
                    joints = joints[0]
                return joints
        return None


# =============================================================================
# Debug Visualization
# =============================================================================

class KinematicVisualizer:
    """
    Visualize kinematic contact detection results.
    Renders joints, contacts, and motion trajectories on video frames.
    """
    
    def __init__(self, config: KinematicContactConfig):
        self.config = config
    
    def render_overlay(
        self,
        images: np.ndarray,
        frames: List[Dict],
        contacts: np.ndarray,
        confidence: np.ndarray,
        debug_data: Dict,
        highlight_joints: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Render debug overlay on video frames.
        
        Args:
            images: (T, H, W, 3) video frames
            frames: Frame data with joints
            contacts: (T, 2) contact array
            confidence: (T, 2) confidence array
            debug_data: Debug data from detector
            highlight_joints: Additional joint indices to highlight
        
        Returns:
            overlay_frames: (T, H, W, 3) frames with overlay
        """
        T, H, W, C = images.shape
        output = images.copy()
        
        # Ensure uint8
        if output.dtype != np.uint8:
            if output.max() <= 1.0:
                output = (output * 255).astype(np.uint8)
            else:
                output = output.astype(np.uint8)
        
        # Collect trajectory history
        trajectory_history = {
            "l_foot": [], "r_foot": [],
            "l_hip": [], "r_hip": [],
            "pelvis": [],
        }
        
        for t in range(T):
            frame_img = output[t].copy()
            frame_data = frames[t] if t < len(frames) else None
            
            if frame_data is None:
                continue
            
            joints_2d = self._get_joints_2d(frame_data)
            if joints_2d is None:
                continue
            
            n_joints = len(joints_2d)
            
            # Get joint positions
            l_ankle_idx = min(self.config.l_ankle_idx, n_joints - 1)
            l_toe_idx = min(self.config.l_toe_idx, n_joints - 1)
            l_heel_idx = min(self.config.l_heel_idx, n_joints - 1)
            r_ankle_idx = min(self.config.r_ankle_idx, n_joints - 1)
            r_toe_idx = min(self.config.r_toe_idx, n_joints - 1)
            r_heel_idx = min(self.config.r_heel_idx, n_joints - 1)
            l_hip_idx = min(self.config.l_hip_idx, n_joints - 1)
            r_hip_idx = min(self.config.r_hip_idx, n_joints - 1)
            pelvis_idx = min(self.config.pelvis_idx, n_joints - 1)
            spine_idx = min(self.config.spine_idx, n_joints - 1)
            
            # Calculate foot centers
            l_foot_center = (joints_2d[l_ankle_idx] + joints_2d[l_toe_idx] + joints_2d[l_heel_idx]) / 3
            r_foot_center = (joints_2d[r_ankle_idx] + joints_2d[r_toe_idx] + joints_2d[r_heel_idx]) / 3
            
            # Update trajectory history
            trajectory_history["l_foot"].append(tuple(l_foot_center.astype(int)))
            trajectory_history["r_foot"].append(tuple(r_foot_center.astype(int)))
            trajectory_history["l_hip"].append(tuple(joints_2d[l_hip_idx].astype(int)))
            trajectory_history["r_hip"].append(tuple(joints_2d[r_hip_idx].astype(int)))
            trajectory_history["pelvis"].append(tuple(joints_2d[pelvis_idx].astype(int)))
            
            # Trim trajectory history
            max_len = self.config.trajectory_length
            for key in trajectory_history:
                if len(trajectory_history[key]) > max_len:
                    trajectory_history[key] = trajectory_history[key][-max_len:]
            
            # Draw trajectories
            if self.config.show_trajectories:
                self._draw_trajectory(frame_img, trajectory_history["l_foot"], 
                                     self.config.color_left_contact, alpha=0.5)
                self._draw_trajectory(frame_img, trajectory_history["r_foot"],
                                     self.config.color_right_contact, alpha=0.5)
                if self.config.show_hips:
                    self._draw_trajectory(frame_img, trajectory_history["pelvis"],
                                         self.config.color_pelvis, alpha=0.3)
            
            # Draw feet joints
            if self.config.show_feet:
                # Left foot
                l_contact = contacts[t, 0] if t < len(contacts) else False
                l_color = self.config.color_left_contact if l_contact else self.config.color_left_flight
                l_conf = confidence[t, 0] if t < len(confidence) else 0.5
                
                for idx in [l_ankle_idx, l_toe_idx, l_heel_idx]:
                    pos = tuple(joints_2d[idx].astype(int))
                    cv2.circle(frame_img, pos, self.config.joint_radius, l_color, -1)
                    cv2.circle(frame_img, pos, self.config.joint_radius + 2, l_color, 1)
                
                # Connect foot joints
                cv2.line(frame_img, tuple(joints_2d[l_ankle_idx].astype(int)),
                        tuple(joints_2d[l_toe_idx].astype(int)), l_color, 2)
                cv2.line(frame_img, tuple(joints_2d[l_ankle_idx].astype(int)),
                        tuple(joints_2d[l_heel_idx].astype(int)), l_color, 2)
                
                # Right foot
                r_contact = contacts[t, 1] if t < len(contacts) else False
                r_color = self.config.color_right_contact if r_contact else self.config.color_right_flight
                r_conf = confidence[t, 1] if t < len(confidence) else 0.5
                
                for idx in [r_ankle_idx, r_toe_idx, r_heel_idx]:
                    pos = tuple(joints_2d[idx].astype(int))
                    cv2.circle(frame_img, pos, self.config.joint_radius, r_color, -1)
                    cv2.circle(frame_img, pos, self.config.joint_radius + 2, r_color, 1)
                
                cv2.line(frame_img, tuple(joints_2d[r_ankle_idx].astype(int)),
                        tuple(joints_2d[r_toe_idx].astype(int)), r_color, 2)
                cv2.line(frame_img, tuple(joints_2d[r_ankle_idx].astype(int)),
                        tuple(joints_2d[r_heel_idx].astype(int)), r_color, 2)
            
            # Draw hip joints
            if self.config.show_hips:
                cv2.circle(frame_img, tuple(joints_2d[l_hip_idx].astype(int)),
                          self.config.joint_radius, self.config.color_hip, -1)
                cv2.circle(frame_img, tuple(joints_2d[r_hip_idx].astype(int)),
                          self.config.joint_radius, self.config.color_hip, -1)
                cv2.circle(frame_img, tuple(joints_2d[pelvis_idx].astype(int)),
                          self.config.joint_radius + 2, self.config.color_pelvis, -1)
            
            # Draw torso
            if self.config.show_torso:
                cv2.circle(frame_img, tuple(joints_2d[spine_idx].astype(int)),
                          self.config.joint_radius, self.config.color_torso, -1)
                # Connect pelvis to spine
                cv2.line(frame_img, tuple(joints_2d[pelvis_idx].astype(int)),
                        tuple(joints_2d[spine_idx].astype(int)), self.config.color_torso, 2)
            
            # Draw custom highlight joints
            if highlight_joints:
                for idx in highlight_joints:
                    if idx < n_joints:
                        pos = tuple(joints_2d[idx].astype(int))
                        cv2.circle(frame_img, pos, self.config.joint_radius + 4, (255, 255, 255), 2)
                        cv2.circle(frame_img, pos, self.config.joint_radius, (0, 165, 255), -1)
            
            # Draw contact status
            if self.config.show_contact_status:
                self._draw_contact_status(frame_img, t, contacts, confidence, W, H)
            
            output[t] = frame_img
        
        return output
    
    def _get_joints_2d(self, frame: Dict) -> Optional[np.ndarray]:
        """Extract 2D joints from frame."""
        for key in ["pred_keypoints_2d", "keypoints_2d", "joints_2d"]:
            if key in frame:
                joints = np.array(frame[key])
                while joints.ndim > 2:
                    joints = joints[0]
                return joints
        return None
    
    def _draw_trajectory(
        self,
        img: np.ndarray,
        points: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        alpha: float = 0.5
    ):
        """Draw trajectory as fading line."""
        if len(points) < 2:
            return
        
        for i in range(1, len(points)):
            # Fade based on age
            age = len(points) - i
            fade = 1.0 - (age / len(points)) * (1.0 - alpha)
            faded_color = tuple(int(c * fade) for c in color)
            
            pt1 = points[i - 1]
            pt2 = points[i]
            
            # Check bounds
            if all(0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0] for p in [pt1, pt2]):
                cv2.line(img, pt1, pt2, faded_color, self.config.trajectory_thickness)
    
    def _draw_contact_status(
        self,
        img: np.ndarray,
        frame_idx: int,
        contacts: np.ndarray,
        confidence: np.ndarray,
        W: int,
        H: int
    ):
        """Draw contact status indicator."""
        # Background box
        box_h = 80
        box_w = 200
        box_x = 10
        box_y = H - box_h - 10
        
        overlay = img.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # Frame number
        cv2.putText(img, f"Frame: {frame_idx}", (box_x + 10, box_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Left foot status
        l_contact = contacts[frame_idx, 0] if frame_idx < len(contacts) else False
        l_conf = confidence[frame_idx, 0] if frame_idx < len(confidence) else 0
        l_status = "CONTACT" if l_contact else "FLIGHT"
        l_color = self.config.color_left_contact if l_contact else self.config.color_left_flight
        
        cv2.putText(img, f"L: {l_status}", (box_x + 10, box_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, l_color, 1)
        cv2.putText(img, f"conf: {l_conf:.2f}", (box_x + 100, box_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Right foot status
        r_contact = contacts[frame_idx, 1] if frame_idx < len(contacts) else False
        r_conf = confidence[frame_idx, 1] if frame_idx < len(confidence) else 0
        r_status = "CONTACT" if r_contact else "FLIGHT"
        r_color = self.config.color_right_contact if r_contact else self.config.color_right_flight
        
        cv2.putText(img, f"R: {r_status}", (box_x + 10, box_y + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, r_color, 1)
        cv2.putText(img, f"conf: {r_conf:.2f}", (box_x + 100, box_y + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


# =============================================================================
# Debug Info Generator
# =============================================================================

def get_quality_rating(error_px: float) -> Tuple[str, str]:
    """Convert reprojection error to quality rating."""
    if error_px < 5:
        return "EXCELLENT", "✓✓"
    elif error_px < 10:
        return "GOOD", "✓"
    elif error_px < 20:
        return "ACCEPTABLE", "~"
    else:
        return "POOR", "✗"


def generate_debug_info(
    contacts: np.ndarray,
    confidence: np.ndarray,
    debug_data: Dict,
    stabilization_info: Optional[Dict] = None
) -> str:
    """Generate comprehensive debug info string."""
    T = contacts.shape[0]
    
    # Get reprojection errors
    reproj_left = [e for e in debug_data.get("reproj_errors", {}).get("left", []) if e is not None]
    reproj_right = [e for e in debug_data.get("reproj_errors", {}).get("right", []) if e is not None]
    
    # Calculate overall quality
    all_errors = reproj_left + reproj_right
    if all_errors:
        overall_error = np.mean(all_errors)
        overall_quality, overall_symbol = get_quality_rating(overall_error)
    else:
        overall_error = 0
        overall_quality, overall_symbol = "UNKNOWN", "?"
    
    lines = [
        "=" * 70,
        "KINEMATIC CONTACT DETECTION RESULTS",
        "=" * 70,
        "",
        f"Total frames: {T}",
        f"Left contact frames: {contacts[:, 0].sum()} ({100*contacts[:, 0].mean():.1f}%)",
        f"Right contact frames: {contacts[:, 1].sum()} ({100*contacts[:, 1].mean():.1f}%)",
        "",
    ]
    
    # === REPROJECTION QUALITY SUMMARY (Prominent) ===
    lines.append("=" * 70)
    lines.append(f"  REPROJECTION QUALITY: {overall_symbol} {overall_quality} (mean={overall_error:.1f}px)")
    lines.append("=" * 70)
    lines.append("")
    
    if reproj_left:
        l_mean = np.mean(reproj_left)
        l_max = np.max(reproj_left)
        l_quality, l_sym = get_quality_rating(l_mean)
        lines.append(f"  Left foot:  {l_sym} {l_quality:10s} mean={l_mean:5.1f}px, max={l_max:5.1f}px")
    
    if reproj_right:
        r_mean = np.mean(reproj_right)
        r_max = np.max(reproj_right)
        r_quality, r_sym = get_quality_rating(r_mean)
        lines.append(f"  Right foot: {r_sym} {r_quality:10s} mean={r_mean:5.1f}px, max={r_max:5.1f}px")
    
    lines.append("")
    lines.append("  Quality thresholds: <5px=EXCELLENT, <10px=GOOD, <20px=ACCEPTABLE, ≥20px=POOR")
    lines.append("")
    
    # Events
    events = debug_data.get("events", [])
    if events:
        lines.append("=== CONTACT EVENTS ===")
        for evt in events[:30]:  # Limit to 30
            lines.append(f"  Frame {evt['frame']:4d}: {evt['foot']:5s} {evt['event']}")
        if len(events) > 30:
            lines.append(f"  ... and {len(events) - 30} more events")
        lines.append("")
    
    # Stabilization info
    if stabilization_info:
        lines.append("=== STABILIZATION ===")
        lines.append(f"  Pin events: {len(stabilization_info.get('pin_events', []))}")
        lines.append(f"  Avg adjustment: {stabilization_info.get('avg_adjustment', 0):.4f}m")
        lines.append(f"  Max adjustment: {stabilization_info.get('max_adjustment', 0):.4f}m")
        lines.append("")
    
    # === CONTACT TIMELINE WITH QUALITY ===
    lines.append("=== CONTACT TIMELINE WITH QUALITY ===")
    lines.append("-" * 70)
    lines.append("Frame |  L   |  R   | L_Err | R_Err | Quality")
    lines.append("-" * 70)
    
    # Get per-frame reprojection errors
    reproj_left_all = debug_data.get("reproj_errors", {}).get("left", [])
    reproj_right_all = debug_data.get("reproj_errors", {}).get("right", [])
    
    # Show key frames + every Nth frame
    shown_frames = set([0, T - 1])
    for evt in events:
        shown_frames.add(evt["frame"])
    for t in range(0, T, max(1, T // 25)):
        shown_frames.add(t)
    
    for t in sorted(shown_frames):
        if t >= T:
            continue
        
        l_char = "████" if contacts[t, 0] else "····"
        r_char = "████" if contacts[t, 1] else "····"
        
        # Get errors for this frame
        l_err = reproj_left_all[t] if t < len(reproj_left_all) and reproj_left_all[t] is not None else None
        r_err = reproj_right_all[t] if t < len(reproj_right_all) and reproj_right_all[t] is not None else None
        
        l_err_str = f"{l_err:5.1f}" if l_err is not None else "  N/A"
        r_err_str = f"{r_err:5.1f}" if r_err is not None else "  N/A"
        
        # Frame quality based on average error
        frame_errors = [e for e in [l_err, r_err] if e is not None]
        if frame_errors:
            frame_avg = np.mean(frame_errors)
            _, quality_sym = get_quality_rating(frame_avg)
        else:
            quality_sym = "?"
        
        lines.append(f"{t:5d} | {l_char} | {r_char} | {l_err_str} | {r_err_str} |   {quality_sym}")
    
    lines.append("-" * 70)
    
    return "\n".join(lines)


def generate_motion_graph(
    debug_data: Dict,
    contacts: np.ndarray,
    confidence: np.ndarray,
    width: int = 1200,
    height: int = 800
) -> np.ndarray:
    """
    Generate motion analysis graph as an image.
    
    Creates a multi-panel plot showing:
    - Foot heights over time
    - Foot-to-pelvis distance
    - Reprojection errors
    - Contact timeline
    
    Returns:
        graph_image: (H, W, 3) numpy array
    """
    import cv2
    
    T = len(debug_data.get("pelvis_positions", []))
    if T == 0:
        # Return blank image if no data
        return np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Extract data
    left_heights = []
    right_heights = []
    for h in debug_data.get("foot_heights", {}).get("left", []):
        if isinstance(h, dict):
            left_heights.append(h.get("ankle", 0))
        else:
            left_heights.append(0)
    
    for h in debug_data.get("foot_heights", {}).get("right", []):
        if isinstance(h, dict):
            right_heights.append(h.get("ankle", 0))
        else:
            right_heights.append(0)
    
    left_to_pelvis = debug_data.get("foot_to_pelvis", {}).get("left", [])
    right_to_pelvis = debug_data.get("foot_to_pelvis", {}).get("right", [])
    
    reproj_left = debug_data.get("reproj_errors", {}).get("left", [])
    reproj_right = debug_data.get("reproj_errors", {}).get("right", [])
    
    # Create figure
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Layout: 4 panels stacked vertically
    panel_height = height // 4
    margin_left = 80
    margin_right = 40
    margin_top = 25
    margin_bottom = 25
    plot_width = width - margin_left - margin_right
    plot_height = panel_height - margin_top - margin_bottom
    
    # Colors
    color_left = (0, 150, 0)       # Green
    color_right = (150, 0, 0)     # Blue (BGR)
    color_contact = (0, 200, 0)   # Bright green
    color_flight = (200, 200, 200)  # Gray
    color_grid = (220, 220, 220)
    color_text = (40, 40, 40)
    color_excellent = (0, 180, 0)
    color_good = (0, 200, 200)
    color_acceptable = (0, 150, 255)
    color_poor = (0, 0, 200)
    
    def draw_panel(panel_idx: int, title: str, data_left: list, data_right: list, 
                   y_label: str, show_quality_bands: bool = False):
        """Draw a single panel."""
        y_offset = panel_idx * panel_height
        
        # Panel background
        cv2.rectangle(img, (0, y_offset), (width, y_offset + panel_height), (250, 250, 250), -1)
        cv2.rectangle(img, (0, y_offset), (width, y_offset + panel_height), (200, 200, 200), 1)
        
        # Title
        cv2.putText(img, title, (margin_left, y_offset + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1)
        
        # Y-axis label
        cv2.putText(img, y_label, (5, y_offset + panel_height // 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, color_text, 1)
        
        # Plot area
        plot_x = margin_left
        plot_y = y_offset + margin_top
        
        # Draw quality bands for reprojection panel
        if show_quality_bands:
            # Excellent: 0-5px, Good: 5-10px, Acceptable: 10-20px, Poor: 20+
            max_err = 30  # Assume max error for scaling
            
            def y_for_value(v):
                return int(plot_y + plot_height - (v / max_err) * plot_height)
            
            # Draw bands
            cv2.rectangle(img, (plot_x, y_for_value(5)), (plot_x + plot_width, y_for_value(0)), 
                         (220, 255, 220), -1)  # Excellent - light green
            cv2.rectangle(img, (plot_x, y_for_value(10)), (plot_x + plot_width, y_for_value(5)), 
                         (220, 255, 255), -1)  # Good - light cyan
            cv2.rectangle(img, (plot_x, y_for_value(20)), (plot_x + plot_width, y_for_value(10)), 
                         (220, 240, 255), -1)  # Acceptable - light orange
            cv2.rectangle(img, (plot_x, y_for_value(max_err)), (plot_x + plot_width, y_for_value(20)), 
                         (220, 220, 255), -1)  # Poor - light red
            
            # Labels
            cv2.putText(img, "EXCELLENT", (plot_x + plot_width + 5, y_for_value(2.5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_excellent, 1)
            cv2.putText(img, "GOOD", (plot_x + plot_width + 5, y_for_value(7.5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_good, 1)
            cv2.putText(img, "ACCEPT", (plot_x + plot_width + 5, y_for_value(15)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_acceptable, 1)
            cv2.putText(img, "POOR", (plot_x + plot_width + 5, y_for_value(25)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_poor, 1)
        
        # Grid lines
        for i in range(5):
            y = plot_y + int(i * plot_height / 4)
            cv2.line(img, (plot_x, y), (plot_x + plot_width, y), color_grid, 1)
        
        # Get data range
        all_data = [d for d in data_left + data_right if d is not None]
        if not all_data:
            return
        
        if show_quality_bands:
            data_min, data_max = 0, 30
        else:
            data_min = min(all_data)
            data_max = max(all_data)
            margin = (data_max - data_min) * 0.1 + 0.001
            data_min -= margin
            data_max += margin
        
        def x_for_frame(f):
            return int(plot_x + (f / max(T - 1, 1)) * plot_width)
        
        def y_for_value(v):
            if data_max == data_min:
                return plot_y + plot_height // 2
            return int(plot_y + plot_height - ((v - data_min) / (data_max - data_min)) * plot_height)
        
        # Draw contact regions as background
        for t in range(T):
            x = x_for_frame(t)
            if t < len(contacts):
                if contacts[t, 0]:  # Left contact
                    cv2.line(img, (x, plot_y), (x, plot_y + plot_height), (200, 255, 200), 1)
                if contacts[t, 1]:  # Right contact
                    cv2.line(img, (x, plot_y), (x, plot_y + plot_height), (255, 200, 200), 1)
        
        # Draw data lines
        def draw_line(data: list, color: tuple):
            points = []
            for t, v in enumerate(data):
                if v is not None:
                    x = x_for_frame(t)
                    y = y_for_value(v)
                    y = max(plot_y, min(plot_y + plot_height, y))
                    points.append((x, y))
            
            for i in range(1, len(points)):
                cv2.line(img, points[i-1], points[i], color, 2)
        
        draw_line(data_left, color_left)
        draw_line(data_right, color_right)
        
        # Y-axis ticks
        for i in range(5):
            y = plot_y + int(i * plot_height / 4)
            val = data_max - i * (data_max - data_min) / 4
            cv2.putText(img, f"{val:.2f}", (plot_x - 45, y + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_text, 1)
        
        # X-axis ticks (frames)
        for f in range(0, T, max(1, T // 10)):
            x = x_for_frame(f)
            cv2.putText(img, str(f), (x - 10, plot_y + plot_height + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_text, 1)
    
    # Panel 1: Foot Heights
    draw_panel(0, "Foot Heights (Y position) - Green=Left, Blue=Right", 
               left_heights, right_heights, "Height")
    
    # Panel 2: Foot-to-Pelvis Distance  
    draw_panel(1, "Foot-to-Pelvis Distance (Contact = increasing)", 
               left_to_pelvis, right_to_pelvis, "Distance")
    
    # Panel 3: Reprojection Errors with quality bands
    draw_panel(2, "Reprojection Error (px) - Quality Bands", 
               reproj_left, reproj_right, "Error(px)", show_quality_bands=True)
    
    # Panel 4: Contact Timeline
    y_offset = 3 * panel_height
    cv2.rectangle(img, (0, y_offset), (width, y_offset + panel_height), (250, 250, 250), -1)
    cv2.putText(img, "Contact Timeline - Green=Contact, Gray=Flight", 
               (margin_left, y_offset + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1)
    
    plot_x = margin_left
    plot_y = y_offset + margin_top
    bar_height = (plot_height - 20) // 2
    
    # Left foot contacts
    cv2.putText(img, "Left:", (10, plot_y + bar_height // 2 + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_text, 1)
    for t in range(T):
        x = int(plot_x + (t / max(T - 1, 1)) * plot_width)
        color = color_contact if contacts[t, 0] else color_flight
        cv2.line(img, (x, plot_y), (x, plot_y + bar_height), color, max(1, plot_width // T))
    
    # Right foot contacts
    cv2.putText(img, "Right:", (10, plot_y + bar_height + 15 + bar_height // 2 + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_text, 1)
    for t in range(T):
        x = int(plot_x + (t / max(T - 1, 1)) * plot_width)
        color = color_contact if contacts[t, 1] else color_flight
        cv2.line(img, (x, plot_y + bar_height + 15), (x, plot_y + 2 * bar_height + 15), color, max(1, plot_width // T))
    
    # X-axis (frames)
    for f in range(0, T, max(1, T // 10)):
        x = int(plot_x + (f / max(T - 1, 1)) * plot_width)
        cv2.putText(img, str(f), (x - 10, y_offset + panel_height - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_text, 1)
    
    # Legend
    legend_y = 15
    cv2.rectangle(img, (width - 180, legend_y - 10), (width - 10, legend_y + 35), (255, 255, 255), -1)
    cv2.rectangle(img, (width - 180, legend_y - 10), (width - 10, legend_y + 35), (150, 150, 150), 1)
    cv2.line(img, (width - 170, legend_y), (width - 140, legend_y), color_left, 2)
    cv2.putText(img, "Left", (width - 135, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_text, 1)
    cv2.line(img, (width - 170, legend_y + 20), (width - 140, legend_y + 20), color_right, 2)
    cv2.putText(img, "Right", (width - 135, legend_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_text, 1)
    
    return img


# =============================================================================
# ComfyUI Node
# =============================================================================

class KinematicContactNode:
    """
    Kinematic Contact Detector - Pure geometry-based foot contact detection.
    
    Detects foot contacts using biomechanical principles:
    - Foot flat (ankle, toe, heel at same height)
    - Pelvis moving away from planted foot
    
    Works for walking, running, sprinting - any bipedal gait.
    No ML models required.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Video frames for overlay visualization"
                }),
                "mesh_sequence": ("MESH_SEQUENCE", {
                    "tooltip": "SAM3DBody output with 3D/2D joints"
                }),
            },
            "optional": {
                # Joint indices
                "pelvis_idx": ("INT", {"default": 0, "min": 0, "max": 50}),
                "l_ankle_idx": ("INT", {"default": 15, "min": 0, "max": 50}),
                "l_toe_idx": ("INT", {"default": 17, "min": 0, "max": 50}),
                "l_heel_idx": ("INT", {"default": 19, "min": 0, "max": 50}),
                "r_ankle_idx": ("INT", {"default": 16, "min": 0, "max": 50}),
                "r_toe_idx": ("INT", {"default": 18, "min": 0, "max": 50}),
                "r_heel_idx": ("INT", {"default": 20, "min": 0, "max": 50}),
                "l_hip_idx": ("INT", {"default": 11, "min": 0, "max": 50}),
                "r_hip_idx": ("INT", {"default": 12, "min": 0, "max": 50}),
                "spine_idx": ("INT", {"default": 6, "min": 0, "max": 50}),
                
                # Detection parameters
                "height_threshold": ("FLOAT", {
                    "default": 0.03,
                    "min": 0.01,
                    "max": 0.1,
                    "step": 0.005,
                    "tooltip": "Height tolerance for 'foot flat' detection (meters)"
                }),
                "min_contact_frames": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Minimum frames for valid contact"
                }),
                
                # Stabilization
                "enable_stabilization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable foot pinning during contacts"
                }),
                "pin_strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "How strongly to pin feet (0=off, 1=full lock)"
                }),
                
                # Visualization
                "show_feet": ("BOOLEAN", {"default": True}),
                "show_hips": ("BOOLEAN", {"default": True}),
                "show_torso": ("BOOLEAN", {"default": True}),
                "show_trajectories": ("BOOLEAN", {"default": True}),
                "highlight_joints": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated joint indices to highlight (e.g., '14,15,16')"
                }),
                
                "log_level": (["normal", "verbose", "debug", "silent"], {
                    "default": "verbose"
                }),
            }
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("stabilized_sequence", "debug_overlay", "motion_graph", "debug_info")
    FUNCTION = "process"
    CATEGORY = "SAM3DBody2abc/Processing"
    
    def process(
        self,
        images,
        mesh_sequence: Dict,
        pelvis_idx: int = 0,
        l_ankle_idx: int = 15,
        l_toe_idx: int = 17,
        l_heel_idx: int = 19,
        r_ankle_idx: int = 16,
        r_toe_idx: int = 18,
        r_heel_idx: int = 20,
        l_hip_idx: int = 11,
        r_hip_idx: int = 12,
        spine_idx: int = 6,
        height_threshold: float = 0.03,
        min_contact_frames: int = 2,
        enable_stabilization: bool = True,
        pin_strength: float = 0.8,
        show_feet: bool = True,
        show_hips: bool = True,
        show_torso: bool = True,
        show_trajectories: bool = True,
        highlight_joints: str = "",
        log_level: str = "verbose",
    ):
        """Process mesh sequence with kinematic contact detection."""
        
        log = KinematicLogger(log_level)
        log.info("=" * 60)
        log.info("KINEMATIC CONTACT DETECTION")
        log.info("=" * 60)
        
        # Build config
        config = KinematicContactConfig(
            pelvis_idx=pelvis_idx,
            l_ankle_idx=l_ankle_idx,
            l_toe_idx=l_toe_idx,
            l_heel_idx=l_heel_idx,
            r_ankle_idx=r_ankle_idx,
            r_toe_idx=r_toe_idx,
            r_heel_idx=r_heel_idx,
            l_hip_idx=l_hip_idx,
            r_hip_idx=r_hip_idx,
            spine_idx=spine_idx,
            height_threshold=height_threshold,
            min_contact_frames=min_contact_frames,
            show_feet=show_feet,
            show_hips=show_hips,
            show_torso=show_torso,
            show_trajectories=show_trajectories,
        )
        
        # Parse highlight joints
        highlight_list = []
        if highlight_joints.strip():
            try:
                highlight_list = [int(x.strip()) for x in highlight_joints.split(",") if x.strip()]
            except ValueError:
                log.warning(f"Could not parse highlight_joints: {highlight_joints}")
        
        # Extract frames
        frames_data = mesh_sequence.get("frames", {})
        if isinstance(frames_data, dict):
            frame_keys = sorted(frames_data.keys())
            frames = [frames_data[k] for k in frame_keys]
        else:
            frame_keys = None
            frames = list(frames_data)
        
        T = len(frames)
        log.info(f"Processing {T} frames")
        log.info(f"Joint indices: ankle=({l_ankle_idx},{r_ankle_idx}), toe=({l_toe_idx},{r_toe_idx}), heel=({l_heel_idx},{r_heel_idx})")
        
        # Extract global intrinsics from mesh_sequence
        global_intrinsics = {
            "focal_length": mesh_sequence.get("focal_length"),
            "cx": mesh_sequence.get("cx"),
            "cy": mesh_sequence.get("cy"),
            "width": mesh_sequence.get("width"),
            "height": mesh_sequence.get("height"),
        }
        
        # Also try to get from first frame if not in mesh_sequence
        if frames and global_intrinsics["focal_length"] is None:
            first_frame = frames[0]
            global_intrinsics["focal_length"] = first_frame.get("focal_length")
            if global_intrinsics["width"] is None:
                img_size = first_frame.get("image_size")
                if img_size and len(img_size) >= 2:
                    global_intrinsics["height"] = img_size[0]
                    global_intrinsics["width"] = img_size[1]
        
        log.verbose(f"Intrinsics: focal={global_intrinsics.get('focal_length')}, size={global_intrinsics.get('width')}x{global_intrinsics.get('height')}")
        
        # Detect contacts
        detector = KinematicContactDetector(config, log)
        contacts, confidence, debug_data = detector.detect(frames, global_intrinsics)
        
        log.info(f"Detected: Left={contacts[:, 0].sum()} frames, Right={contacts[:, 1].sum()} frames")
        
        # Stabilize if enabled
        stabilization_info = None
        if enable_stabilization:
            log.info(f"Applying stabilization (pin_strength={pin_strength})")
            stabilizer = FootStabilizer(config, log)
            frames, stabilization_info = stabilizer.stabilize(frames, contacts, pin_strength)
            log.info(f"Stabilization: {len(stabilization_info.get('pin_events', []))} pin events")
        
        # Update mesh_sequence
        result = mesh_sequence.copy()
        if frame_keys is not None:
            result["frames"] = {k: v for k, v in zip(frame_keys, frames)}
        else:
            result["frames"] = frames
        
        # Add metadata
        result["kinematic_contact"] = {
            "method": "kinematic",
            "contacts_detected": {
                "left": int(contacts[:, 0].sum()),
                "right": int(contacts[:, 1].sum()),
            },
            "avg_confidence": {
                "left": float(confidence[:, 0].mean()),
                "right": float(confidence[:, 1].mean()),
            },
            "stabilization_enabled": enable_stabilization,
        }
        
        # Convert images to numpy
        if torch.is_tensor(images):
            images_np = images.cpu().numpy()
        else:
            images_np = np.array(images)
        
        if images_np.dtype != np.uint8:
            if images_np.max() <= 1.0:
                images_np = (images_np * 255).astype(np.uint8)
            else:
                images_np = images_np.astype(np.uint8)
        
        # Generate overlay
        visualizer = KinematicVisualizer(config)
        overlay = visualizer.render_overlay(
            images_np, frames, contacts, confidence, debug_data, highlight_list
        )
        
        # Convert back to tensor
        overlay_tensor = torch.from_numpy(overlay.astype(np.float32) / 255.0)
        
        # Generate debug info
        debug_info = generate_debug_info(contacts, confidence, debug_data, stabilization_info)
        
        # Generate motion graph image
        graph_img = generate_motion_graph(debug_data, contacts, confidence)
        # Convert BGR to RGB and to tensor
        graph_img_rgb = cv2.cvtColor(graph_img, cv2.COLOR_BGR2RGB)
        graph_tensor = torch.from_numpy(graph_img_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        
        # === Log quality summary ===
        reproj_left = [e for e in debug_data.get("reproj_errors", {}).get("left", []) if e is not None]
        reproj_right = [e for e in debug_data.get("reproj_errors", {}).get("right", []) if e is not None]
        all_errors = reproj_left + reproj_right
        
        if all_errors:
            overall_error = np.mean(all_errors)
            quality, symbol = get_quality_rating(overall_error)
            log.info("")
            log.info("=" * 60)
            log.info(f"  REPROJECTION QUALITY: {symbol} {quality} (mean={overall_error:.1f}px)")
            log.info("=" * 60)
            if reproj_left:
                l_quality, l_sym = get_quality_rating(np.mean(reproj_left))
                log.info(f"  Left foot:  {l_sym} {l_quality} (mean={np.mean(reproj_left):.1f}px)")
            if reproj_right:
                r_quality, r_sym = get_quality_rating(np.mean(reproj_right))
                log.info(f"  Right foot: {r_sym} {r_quality} (mean={np.mean(reproj_right):.1f}px)")
            log.info("")
        
        log.info("Processing complete")
        
        return (result, overlay_tensor, graph_tensor, debug_info)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "KinematicContactDetector": KinematicContactNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KinematicContactDetector": "🦶📐 Kinematic Contact Detector",
}
