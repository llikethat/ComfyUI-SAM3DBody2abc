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
    Detect foot contacts using reference-based geometry comparison.
    
    Method:
    1. Auto-detect or use user-specified reference frames where each foot is flat
    2. Extract foot geometry signature from reference frames
    3. Compare each frame's foot geometry to reference
    4. Normalize by camera distance (pred_cam_t) to handle depth changes
    
    This approach is robust to:
    - Character moving toward/away from camera
    - Camera motion
    - Different skeleton scales
    """
    
    def __init__(self, config: KinematicContactConfig, log: KinematicLogger):
        self.config = config
        self.log = log
        self.global_intrinsics = {}
    
    def detect(
        self, 
        frames: List[Dict], 
        global_intrinsics: Optional[Dict] = None,
        left_ref_frame: int = -1,
        right_ref_frame: int = -1
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Detect contacts for all frames using reference-based geometry.
        
        Args:
            frames: List of frame data with pred_keypoints_3d, pred_keypoints_2d
            global_intrinsics: Optional dict with focal_length, cx, cy, width, height
            left_ref_frame: Frame number where left foot is flat (-1 = auto-detect)
            right_ref_frame: Frame number where right foot is flat (-1 = auto-detect)
        
        Returns:
            contacts: (T, 2) boolean array [left, right]
            confidence: (T, 2) float array [0-1]
            debug_data: Dictionary with detailed per-frame info
        """
        T = len(frames)
        contacts = np.zeros((T, 2), dtype=bool)
        confidence = np.zeros((T, 2), dtype=np.float32)
        
        self.global_intrinsics = global_intrinsics or {}
        
        # Debug data storage
        debug_data = {
            "foot_heights": {"left": [], "right": []},
            "foot_flat": {"left": [], "right": []},
            "foot_to_pelvis": {"left": [], "right": []},
            "pelvis_moving_away": {"left": [], "right": []},
            "reproj_errors": {"left": [], "right": []},
            "geometry_diff": {"left": [], "right": []},  # NEW: geometry difference from reference
            "depth_scale": [],  # NEW: depth scaling factor
            "pelvis_positions": [],
            "foot_positions": {"left": [], "right": []},
            "hip_positions": {"left": [], "right": []},
            "events": [],
            "joints_2d": {
                "l_ankle": [], "l_toe": [], "l_heel": [],
                "r_ankle": [], "r_toe": [], "r_heel": [],
                "pelvis": [],
            },
            "joints_3d": {
                "l_ankle": [], "l_toe": [], "l_heel": [],
                "r_ankle": [], "r_toe": [], "r_heel": [],
                "pelvis": [],
            },
            "reference_frames": {"left": None, "right": None},
            "reference_geometry": {"left": None, "right": None},
        }
        
        # Step 1: Extract all foot geometries for analysis
        self.log.info("Extracting foot geometries from all frames...")
        all_foot_data = self._extract_all_foot_data(frames)
        
        if not all_foot_data["left"] or not all_foot_data["right"]:
            self.log.warning("Could not extract foot data from frames")
            return contacts, confidence, debug_data
        
        # Step 2: Find reference frames (auto-detect or use user-specified)
        left_ref = left_ref_frame if left_ref_frame >= 0 else self._auto_detect_reference_frame(all_foot_data["left"], "left")
        right_ref = right_ref_frame if right_ref_frame >= 0 else self._auto_detect_reference_frame(all_foot_data["right"], "right")
        
        debug_data["reference_frames"]["left"] = left_ref
        debug_data["reference_frames"]["right"] = right_ref
        
        self.log.info(f"Reference frames: Left={left_ref}, Right={right_ref}")
        
        # Step 3: Extract reference geometry and depth
        left_ref_geom, left_ref_depth = self._get_reference_geometry(all_foot_data["left"], left_ref)
        right_ref_geom, right_ref_depth = self._get_reference_geometry(all_foot_data["right"], right_ref)
        
        debug_data["reference_geometry"]["left"] = left_ref_geom
        debug_data["reference_geometry"]["right"] = right_ref_geom
        
        if left_ref_geom:
            self.log.info(f"Left ref geometry: Y-spread={left_ref_geom['y_spread']:.4f}m, depth={left_ref_depth:.2f}")
        if right_ref_geom:
            self.log.info(f"Right ref geometry: Y-spread={right_ref_geom['y_spread']:.4f}m, depth={right_ref_depth:.2f}")
        
        # Step 4: Process each frame
        prev_contacts = [False, False]
        
        for t, frame in enumerate(frames):
            joints_3d = self._get_joints_3d(frame)
            joints_2d = self._get_joints_2d(frame)
            
            if joints_3d is None:
                self.log.debug(f"Frame {t}: No 3D joints found")
                # Append None/empty values for this frame
                self._append_empty_debug_data(debug_data)
                continue
            
            # Get camera depth for scaling
            pred_cam_t = frame.get("pred_cam_t", [0, 0, 5])
            curr_depth = pred_cam_t[2] if len(pred_cam_t) > 2 else 5.0
            debug_data["depth_scale"].append(curr_depth)
            
            # Get pelvis position
            pelvis_idx = min(self.config.pelvis_idx, len(joints_3d) - 1)
            pelvis = joints_3d[pelvis_idx]
            debug_data["pelvis_positions"].append(pelvis.tolist())
            debug_data["joints_3d"]["pelvis"].append(pelvis.tolist())
            if joints_2d is not None and pelvis_idx < len(joints_2d):
                debug_data["joints_2d"]["pelvis"].append(joints_2d[pelvis_idx].tolist())
            else:
                debug_data["joints_2d"]["pelvis"].append(None)
            
            # Get hip positions
            l_hip_idx = min(self.config.l_hip_idx, len(joints_3d) - 1)
            r_hip_idx = min(self.config.r_hip_idx, len(joints_3d) - 1)
            debug_data["hip_positions"]["left"].append(joints_3d[l_hip_idx].tolist())
            debug_data["hip_positions"]["right"].append(joints_3d[r_hip_idx].tolist())
            
            # Process each foot
            for foot_idx, side in enumerate(["left", "right"]):
                ref_geom = left_ref_geom if side == "left" else right_ref_geom
                ref_depth = left_ref_depth if side == "left" else right_ref_depth
                
                # Get joint indices
                if side == "left":
                    ankle_idx = min(self.config.l_ankle_idx, len(joints_3d) - 1)
                    toe_idx = min(self.config.l_toe_idx, len(joints_3d) - 1)
                    heel_idx = min(self.config.l_heel_idx, len(joints_3d) - 1)
                    prefix = "l_"
                else:
                    ankle_idx = min(self.config.r_ankle_idx, len(joints_3d) - 1)
                    toe_idx = min(self.config.r_toe_idx, len(joints_3d) - 1)
                    heel_idx = min(self.config.r_heel_idx, len(joints_3d) - 1)
                    prefix = "r_"
                
                ankle = joints_3d[ankle_idx]
                toe = joints_3d[toe_idx]
                heel = joints_3d[heel_idx]
                
                foot_center = (ankle + toe + heel) / 3
                debug_data["foot_positions"][side].append(foot_center.tolist())
                
                # Store 3D positions
                debug_data["joints_3d"][f"{prefix}ankle"].append(ankle.tolist())
                debug_data["joints_3d"][f"{prefix}toe"].append(toe.tolist())
                debug_data["joints_3d"][f"{prefix}heel"].append(heel.tolist())
                
                # Store 2D positions
                if joints_2d is not None:
                    n_joints_2d = len(joints_2d)
                    debug_data["joints_2d"][f"{prefix}ankle"].append(
                        joints_2d[ankle_idx].tolist() if ankle_idx < n_joints_2d else None)
                    debug_data["joints_2d"][f"{prefix}toe"].append(
                        joints_2d[toe_idx].tolist() if toe_idx < n_joints_2d else None)
                    debug_data["joints_2d"][f"{prefix}heel"].append(
                        joints_2d[heel_idx].tolist() if heel_idx < n_joints_2d else None)
                else:
                    debug_data["joints_2d"][f"{prefix}ankle"].append(None)
                    debug_data["joints_2d"][f"{prefix}toe"].append(None)
                    debug_data["joints_2d"][f"{prefix}heel"].append(None)
                
                # Calculate current foot geometry
                curr_geom = self._calculate_foot_geometry(ankle, toe, heel)
                
                # Store heights
                debug_data["foot_heights"][side].append({
                    "ankle": float(ankle[1]),
                    "toe": float(toe[1]),
                    "heel": float(heel[1]),
                    "range": float(curr_geom["y_spread"]),
                })
                
                # Calculate foot-to-pelvis distance
                foot_to_pelvis = np.linalg.norm(pelvis - foot_center)
                debug_data["foot_to_pelvis"][side].append(float(foot_to_pelvis))
                
                # Compare geometry to reference (with depth normalization)
                if ref_geom is not None and ref_depth > 0:
                    # Depth scaling factor
                    depth_scale = curr_depth / ref_depth
                    
                    # Calculate geometry difference
                    geom_diff = self._compare_geometry(curr_geom, ref_geom, depth_scale)
                    debug_data["geometry_diff"][side].append(geom_diff)
                    
                    # Primary: Y-axis spread comparison
                    y_diff = abs(curr_geom["y_spread"] - ref_geom["y_spread"] * depth_scale)
                    
                    # Contact if geometry is similar to reference (foot is flat)
                    is_flat = geom_diff < self.config.height_threshold
                    debug_data["foot_flat"][side].append(is_flat)
                    
                    # Secondary validation: pelvis moving away (body moving over planted foot)
                    pelvis_moving_away = False
                    if t > 0 and len(debug_data["foot_to_pelvis"][side]) > 1:
                        prev_dist = debug_data["foot_to_pelvis"][side][-2]
                        pelvis_moving_away = foot_to_pelvis > prev_dist - 0.01  # Small tolerance
                    debug_data["pelvis_moving_away"][side].append(pelvis_moving_away)
                    
                    # Contact decision: flat geometry AND pelvis moving away (or first frame)
                    is_contact = is_flat and (pelvis_moving_away or t == 0)
                    contacts[t, foot_idx] = is_contact
                    
                    # Confidence based on how close to reference geometry
                    # Lower difference = higher confidence
                    conf = max(0.0, 1.0 - (geom_diff / (self.config.height_threshold * 2)))
                    confidence[t, foot_idx] = conf
                    
                else:
                    debug_data["geometry_diff"][side].append(None)
                    debug_data["foot_flat"][side].append(False)
                    debug_data["pelvis_moving_away"][side].append(False)
                
                # Note: Reprojection error removed - 2D px vs 3D meters comparison is invalid
                debug_data["reproj_errors"][side].append(None)
                
                # Log contact events
                is_contact = contacts[t, foot_idx]
                if is_contact != prev_contacts[foot_idx]:
                    event = "CONTACT_START" if is_contact else "CONTACT_END"
                    debug_data["events"].append({
                        "frame": t,
                        "foot": side,
                        "event": event,
                    })
                    self.log.debug(f"Frame {t}: {side} {event}")
                
                prev_contacts[foot_idx] = is_contact
        
        # Filter short contacts
        contacts = self._filter_short_contacts(contacts)
        
        return contacts, confidence, debug_data
    
    def _extract_all_foot_data(self, frames: List[Dict]) -> Dict:
        """Extract foot joint data from all frames for analysis."""
        all_data = {"left": [], "right": []}
        
        for t, frame in enumerate(frames):
            joints_3d = self._get_joints_3d(frame)
            if joints_3d is None:
                all_data["left"].append(None)
                all_data["right"].append(None)
                continue
            
            pred_cam_t = frame.get("pred_cam_t", [0, 0, 5])
            depth = pred_cam_t[2] if len(pred_cam_t) > 2 else 5.0
            
            n_joints = len(joints_3d)
            
            # Left foot
            l_ankle = joints_3d[min(self.config.l_ankle_idx, n_joints - 1)]
            l_toe = joints_3d[min(self.config.l_toe_idx, n_joints - 1)]
            l_heel = joints_3d[min(self.config.l_heel_idx, n_joints - 1)]
            l_geom = self._calculate_foot_geometry(l_ankle, l_toe, l_heel)
            all_data["left"].append({
                "frame": t,
                "geometry": l_geom,
                "depth": depth,
                "joints": [l_ankle, l_toe, l_heel],
            })
            
            # Right foot
            r_ankle = joints_3d[min(self.config.r_ankle_idx, n_joints - 1)]
            r_toe = joints_3d[min(self.config.r_toe_idx, n_joints - 1)]
            r_heel = joints_3d[min(self.config.r_heel_idx, n_joints - 1)]
            r_geom = self._calculate_foot_geometry(r_ankle, r_toe, r_heel)
            all_data["right"].append({
                "frame": t,
                "geometry": r_geom,
                "depth": depth,
                "joints": [r_ankle, r_toe, r_heel],
            })
        
        return all_data
    
    def _calculate_foot_geometry(self, ankle: np.ndarray, toe: np.ndarray, heel: np.ndarray) -> Dict:
        """Calculate foot geometry signature."""
        # Y-axis spread (primary - height difference)
        heights = [ankle[1], toe[1], heel[1]]
        y_spread = max(heights) - min(heights)
        
        # X-axis spread (secondary - lateral spread)
        x_coords = [ankle[0], toe[0], heel[0]]
        x_spread = max(x_coords) - min(x_coords)
        
        # Z-axis spread (secondary - depth spread)
        z_coords = [ankle[2], toe[2], heel[2]]
        z_spread = max(z_coords) - min(z_coords)
        
        # Relative vectors (ankle as origin)
        ankle_to_toe = toe - ankle
        ankle_to_heel = heel - ankle
        
        return {
            "y_spread": float(y_spread),
            "x_spread": float(x_spread),
            "z_spread": float(z_spread),
            "ankle_to_toe": ankle_to_toe.tolist(),
            "ankle_to_heel": ankle_to_heel.tolist(),
            "min_height": float(min(heights)),
            "max_height": float(max(heights)),
        }
    
    def _auto_detect_reference_frame(self, foot_data: List, side: str) -> int:
        """Auto-detect the best reference frame (where foot is most flat)."""
        best_frame = 0
        best_score = float('inf')
        
        for data in foot_data:
            if data is None:
                continue
            
            geom = data["geometry"]
            # Score = Y-spread (lower is better = flatter foot)
            # Add small penalty for extreme depths (might have more noise)
            score = geom["y_spread"]
            
            if score < best_score:
                best_score = score
                best_frame = data["frame"]
        
        self.log.info(f"Auto-detected {side} foot reference: frame {best_frame} (y_spread={best_score:.4f}m)")
        return best_frame
    
    def _get_reference_geometry(self, foot_data: List, ref_frame: int) -> Tuple[Optional[Dict], float]:
        """Get geometry and depth from reference frame."""
        if ref_frame < 0 or ref_frame >= len(foot_data):
            return None, 1.0
        
        data = foot_data[ref_frame]
        if data is None:
            return None, 1.0
        
        return data["geometry"], data["depth"]
    
    def _compare_geometry(self, curr: Dict, ref: Dict, depth_scale: float) -> float:
        """
        Compare current foot geometry to reference.
        Returns a difference score (lower = more similar = more likely contact).
        """
        # Primary: Y-spread comparison (height difference between joints)
        # Scale reference by depth ratio to account for perspective changes
        y_diff = abs(curr["y_spread"] - ref["y_spread"] * depth_scale)
        
        # Secondary: X and Z spread (should be roughly similar when foot is flat)
        x_diff = abs(curr["x_spread"] - ref["x_spread"] * depth_scale) * 0.3  # Lower weight
        z_diff = abs(curr["z_spread"] - ref["z_spread"] * depth_scale) * 0.3  # Lower weight
        
        # Combined score (Y is primary)
        total_diff = y_diff + x_diff + z_diff
        
        return total_diff
    
    def _append_empty_debug_data(self, debug_data: Dict):
        """Append None/empty values when frame has no data."""
        debug_data["pelvis_positions"].append(None)
        debug_data["depth_scale"].append(None)
        debug_data["joints_3d"]["pelvis"].append(None)
        debug_data["joints_2d"]["pelvis"].append(None)
        
        for side in ["left", "right"]:
            debug_data["foot_heights"][side].append(None)
            debug_data["foot_flat"][side].append(False)
            debug_data["foot_to_pelvis"][side].append(None)
            debug_data["pelvis_moving_away"][side].append(False)
            debug_data["reproj_errors"][side].append(None)
            debug_data["geometry_diff"][side].append(None)
            debug_data["foot_positions"][side].append(None)
            debug_data["hip_positions"][side].append(None)
        
        for key in debug_data["joints_3d"]:
            debug_data["joints_3d"][key].append(None)
        for key in debug_data["joints_2d"]:
            debug_data["joints_2d"][key].append(None)
    
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
                # Get reprojection errors for this frame
                reproj_left = debug_data.get("reproj_errors", {}).get("left", [])
                reproj_right = debug_data.get("reproj_errors", {}).get("right", [])
                l_err = reproj_left[t] if t < len(reproj_left) and reproj_left[t] is not None else None
                r_err = reproj_right[t] if t < len(reproj_right) and reproj_right[t] is not None else None
                
                # Left foot
                l_contact = contacts[t, 0] if t < len(contacts) else False
                l_color = self.config.color_left_contact if l_contact else self.config.color_left_flight
                l_conf = confidence[t, 0] if t < len(confidence) else 0.5
                
                # Draw joints with index labels
                l_joint_info = [
                    (l_ankle_idx, "A"),  # Ankle
                    (l_toe_idx, "T"),    # Toe
                    (l_heel_idx, "H"),   # Heel
                ]
                for idx, label in l_joint_info:
                    pos = tuple(joints_2d[idx].astype(int))
                    # Draw joint circle
                    cv2.circle(frame_img, pos, self.config.joint_radius, l_color, -1)
                    cv2.circle(frame_img, pos, self.config.joint_radius + 2, l_color, 1)
                    # Draw index label
                    label_pos = (pos[0] - 20, pos[1] - 10)
                    cv2.putText(frame_img, f"{idx}:{label}", label_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 2)
                    cv2.putText(frame_img, f"{idx}:{label}", label_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, l_color, 1)
                
                # Connect foot joints
                cv2.line(frame_img, tuple(joints_2d[l_ankle_idx].astype(int)),
                        tuple(joints_2d[l_toe_idx].astype(int)), l_color, 2)
                cv2.line(frame_img, tuple(joints_2d[l_ankle_idx].astype(int)),
                        tuple(joints_2d[l_heel_idx].astype(int)), l_color, 2)
                
                # Draw reprojection error indicator for left foot
                if l_err is not None:
                    l_foot_center = (joints_2d[l_ankle_idx] + joints_2d[l_toe_idx] + joints_2d[l_heel_idx]) / 3
                    err_pos = tuple(l_foot_center.astype(int))
                    # Color based on error quality
                    if l_err < 5:
                        err_color = (0, 200, 0)  # Green - excellent
                    elif l_err < 10:
                        err_color = (0, 200, 200)  # Yellow - good
                    elif l_err < 20:
                        err_color = (0, 150, 255)  # Orange - acceptable
                    else:
                        err_color = (0, 0, 200)  # Red - poor
                    cv2.putText(frame_img, f"{l_err:.1f}px", (err_pos[0] - 15, err_pos[1] + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                    cv2.putText(frame_img, f"{l_err:.1f}px", (err_pos[0] - 15, err_pos[1] + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, err_color, 1)
                
                # Right foot
                r_contact = contacts[t, 1] if t < len(contacts) else False
                r_color = self.config.color_right_contact if r_contact else self.config.color_right_flight
                r_conf = confidence[t, 1] if t < len(confidence) else 0.5
                
                # Draw joints with index labels
                r_joint_info = [
                    (r_ankle_idx, "A"),  # Ankle
                    (r_toe_idx, "T"),    # Toe
                    (r_heel_idx, "H"),   # Heel
                ]
                for idx, label in r_joint_info:
                    pos = tuple(joints_2d[idx].astype(int))
                    # Draw joint circle
                    cv2.circle(frame_img, pos, self.config.joint_radius, r_color, -1)
                    cv2.circle(frame_img, pos, self.config.joint_radius + 2, r_color, 1)
                    # Draw index label
                    label_pos = (pos[0] + 8, pos[1] - 10)
                    cv2.putText(frame_img, f"{idx}:{label}", label_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 2)
                    cv2.putText(frame_img, f"{idx}:{label}", label_pos, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, r_color, 1)
                
                cv2.line(frame_img, tuple(joints_2d[r_ankle_idx].astype(int)),
                        tuple(joints_2d[r_toe_idx].astype(int)), r_color, 2)
                cv2.line(frame_img, tuple(joints_2d[r_ankle_idx].astype(int)),
                        tuple(joints_2d[r_heel_idx].astype(int)), r_color, 2)
                
                # Draw reprojection error indicator for right foot
                if r_err is not None:
                    r_foot_center = (joints_2d[r_ankle_idx] + joints_2d[r_toe_idx] + joints_2d[r_heel_idx]) / 3
                    err_pos = tuple(r_foot_center.astype(int))
                    # Color based on error quality
                    if r_err < 5:
                        err_color = (0, 200, 0)  # Green - excellent
                    elif r_err < 10:
                        err_color = (0, 200, 200)  # Yellow - good
                    elif r_err < 20:
                        err_color = (0, 150, 255)  # Orange - acceptable
                    else:
                        err_color = (0, 0, 200)  # Red - poor
                    cv2.putText(frame_img, f"{r_err:.1f}px", (err_pos[0] - 15, err_pos[1] + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                    cv2.putText(frame_img, f"{r_err:.1f}px", (err_pos[0] - 15, err_pos[1] + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, err_color, 1)
            
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
            
            # Draw joint indices legend (top-right corner)
            self._draw_joint_legend(frame_img, W, H, 
                                   l_ankle_idx, l_toe_idx, l_heel_idx,
                                   r_ankle_idx, r_toe_idx, r_heel_idx)
            
            output[t] = frame_img
        
        return output
    
    def _draw_joint_legend(
        self,
        img: np.ndarray,
        W: int, H: int,
        l_ankle: int, l_toe: int, l_heel: int,
        r_ankle: int, r_toe: int, r_heel: int
    ):
        """Draw legend showing joint indices used for reprojection."""
        # Legend box position (top-right)
        box_w = 200
        box_h = 105
        box_x = W - box_w - 10
        box_y = 10
        
        # Semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (100, 100, 100), 1)
        
        # Title
        cv2.putText(img, "REPROJECTION JOINTS", (box_x + 10, box_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Left foot (anatomical - character's left)
        cv2.putText(img, "Left:", (box_x + 10, box_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.config.color_left_contact, 1)
        cv2.putText(img, f"A={l_ankle} T={l_toe} H={l_heel}", (box_x + 50, box_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Right foot (anatomical - character's right)
        cv2.putText(img, "Right:", (box_x + 10, box_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.config.color_right_contact, 1)
        cv2.putText(img, f"A={r_ankle} T={r_toe} H={r_heel}", (box_x + 50, box_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # Legend for A, T, H
        cv2.putText(img, "A=Ankle T=Toe H=Heel", (box_x + 10, box_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        
        # Clarification: anatomical naming
        cv2.putText(img, "(Anatomical: Char's L/R)", (box_x + 10, box_y + 92),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.28, (120, 120, 120), 1)
    
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
    
    # Get geometry differences
    geom_left = [g for g in debug_data.get("geometry_diff", {}).get("left", []) if g is not None]
    geom_right = [g for g in debug_data.get("geometry_diff", {}).get("right", []) if g is not None]
    
    # Reference frames
    ref_frames = debug_data.get("reference_frames", {})
    ref_geom = debug_data.get("reference_geometry", {})
    
    lines = [
        "=" * 70,
        "KINEMATIC CONTACT DETECTION RESULTS (Reference-Based)",
        "=" * 70,
        "",
        f"Total frames: {T}",
        f"Left contact frames: {contacts[:, 0].sum()} ({100*contacts[:, 0].mean():.1f}%)",
        f"Right contact frames: {contacts[:, 1].sum()} ({100*contacts[:, 1].mean():.1f}%)",
        "",
    ]
    
    # === REFERENCE FRAMES ===
    lines.append("=== REFERENCE FRAMES (Flat Foot) ===")
    lines.append(f"  Left foot reference:  Frame {ref_frames.get('left', 'N/A')}")
    if ref_geom.get('left'):
        lines.append(f"    Y-spread: {ref_geom['left'].get('y_spread', 0):.4f}m")
    lines.append(f"  Right foot reference: Frame {ref_frames.get('right', 'N/A')}")
    if ref_geom.get('right'):
        lines.append(f"    Y-spread: {ref_geom['right'].get('y_spread', 0):.4f}m")
    lines.append("")
    
    # === GEOMETRY DIFFERENCE SUMMARY ===
    lines.append("=== GEOMETRY DIFFERENCE FROM REFERENCE ===")
    if geom_left:
        l_mean = np.mean(geom_left)
        l_min = np.min(geom_left)
        l_max = np.max(geom_left)
        lines.append(f"  Left foot:  mean={l_mean:.4f}m, min={l_min:.4f}m, max={l_max:.4f}m")
    if geom_right:
        r_mean = np.mean(geom_right)
        r_min = np.min(geom_right)
        r_max = np.max(geom_right)
        lines.append(f"  Right foot: mean={r_mean:.4f}m, min={r_min:.4f}m, max={r_max:.4f}m")
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
    
    # === CONTACT TIMELINE ===
    lines.append("=== CONTACT TIMELINE ===")
    lines.append("-" * 70)
    lines.append("Frame |  L   |  R   | L_Geom | R_Geom | L_Conf | R_Conf")
    lines.append("-" * 70)
    
    # Get per-frame geometry differences
    geom_left_all = debug_data.get("geometry_diff", {}).get("left", [])
    geom_right_all = debug_data.get("geometry_diff", {}).get("right", [])
    
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
        
        # Get geometry diff for this frame
        l_geom = geom_left_all[t] if t < len(geom_left_all) and geom_left_all[t] is not None else None
        r_geom = geom_right_all[t] if t < len(geom_right_all) and geom_right_all[t] is not None else None
        
        l_geom_str = f"{l_geom:.4f}" if l_geom is not None else "  N/A "
        r_geom_str = f"{r_geom:.4f}" if r_geom is not None else "  N/A "
        
        l_conf = confidence[t, 0] if t < len(confidence) else 0
        r_conf = confidence[t, 1] if t < len(confidence) else 0
        
        lines.append(f"{t:5d} | {l_char} | {r_char} | {l_geom_str} | {r_geom_str} | {l_conf:.2f}  | {r_conf:.2f}")
    
    lines.append("-" * 70)
    
    return "\n".join(lines)


def generate_motion_graph(
    debug_data: Dict,
    contacts: np.ndarray,
    confidence: np.ndarray,
    width: int = 1400,
    height: int = 1600
) -> np.ndarray:
    """
    Generate motion analysis graph as an image.
    
    Creates a multi-panel plot showing:
    - 2D Screen Space: X, Y positions of foot joints (px)
    - 3D World Space: X, Y, Z positions of foot joints (m)
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
    
    # Extract existing data
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
    
    # Extract 2D joint positions
    joints_2d = debug_data.get("joints_2d", {})
    joints_3d = debug_data.get("joints_3d", {})
    
    # Helper to extract component from list of [x,y] or [x,y,z]
    def extract_component(data_list, component_idx):
        result = []
        for item in data_list:
            if item is not None and len(item) > component_idx:
                result.append(item[component_idx])
            else:
                result.append(None)
        return result
    
    # Create figure - 8 panels
    num_panels = 8
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    panel_height = height // num_panels
    margin_left = 80
    margin_right = 100
    margin_top = 22
    margin_bottom = 18
    plot_width = width - margin_left - margin_right
    plot_height = panel_height - margin_top - margin_bottom
    
    # Colors - using distinct colors for ankle, toe, heel
    color_l_ankle = (0, 180, 0)      # Green
    color_l_toe = (0, 220, 100)      # Light green
    color_l_heel = (0, 140, 0)       # Dark green
    color_r_ankle = (180, 0, 0)      # Blue
    color_r_toe = (220, 100, 0)      # Light blue
    color_r_heel = (140, 0, 0)       # Dark blue
    color_pelvis = (0, 0, 180)       # Red
    color_contact = (0, 200, 0)
    color_flight = (200, 200, 200)
    color_grid = (230, 230, 230)
    color_text = (40, 40, 40)
    color_excellent = (0, 180, 0)
    color_good = (0, 200, 200)
    color_acceptable = (0, 150, 255)
    color_poor = (0, 0, 200)
    
    def draw_multi_line_panel(panel_idx: int, title: str, data_dict: Dict[str, tuple], 
                              y_label: str, show_quality_bands: bool = False):
        """Draw a panel with multiple data lines.
        data_dict: {label: (data_list, color)}
        """
        y_offset = panel_idx * panel_height
        
        # Panel background
        cv2.rectangle(img, (0, y_offset), (width, y_offset + panel_height), (252, 252, 252), -1)
        cv2.rectangle(img, (0, y_offset), (width, y_offset + panel_height), (200, 200, 200), 1)
        
        # Title
        cv2.putText(img, title, (margin_left, y_offset + 16), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_text, 1)
        
        # Y-axis label
        cv2.putText(img, y_label, (5, y_offset + panel_height // 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.32, color_text, 1)
        
        plot_x = margin_left
        plot_y = y_offset + margin_top
        
        # Draw quality bands for reprojection panel
        if show_quality_bands:
            max_err = 30
            def y_for_val(v):
                return int(plot_y + plot_height - (v / max_err) * plot_height)
            cv2.rectangle(img, (plot_x, y_for_val(5)), (plot_x + plot_width, y_for_val(0)), (220, 255, 220), -1)
            cv2.rectangle(img, (plot_x, y_for_val(10)), (plot_x + plot_width, y_for_val(5)), (220, 255, 255), -1)
            cv2.rectangle(img, (plot_x, y_for_val(20)), (plot_x + plot_width, y_for_val(10)), (220, 240, 255), -1)
            cv2.rectangle(img, (plot_x, y_for_val(max_err)), (plot_x + plot_width, y_for_val(20)), (220, 220, 255), -1)
        
        # Grid lines
        for i in range(5):
            y = plot_y + int(i * plot_height / 4)
            cv2.line(img, (plot_x, y), (plot_x + plot_width, y), color_grid, 1)
        
        # Collect all data for range calculation
        all_data = []
        for label, (data, color) in data_dict.items():
            all_data.extend([d for d in data if d is not None])
        
        if not all_data:
            cv2.putText(img, "No data", (plot_x + plot_width//2 - 30, plot_y + plot_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            return
        
        if show_quality_bands:
            data_min, data_max = 0, 30
        else:
            data_min = min(all_data)
            data_max = max(all_data)
            margin_val = (data_max - data_min) * 0.1 + 0.001
            data_min -= margin_val
            data_max += margin_val
        
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
                if contacts[t, 0]:
                    cv2.line(img, (x, plot_y), (x, plot_y + plot_height), (220, 255, 220), 1)
                if contacts[t, 1]:
                    cv2.line(img, (x, plot_y), (x, plot_y + plot_height), (255, 220, 220), 1)
        
        # Draw data lines
        legend_x = plot_x + plot_width + 5
        legend_y = plot_y + 10
        for label, (data, color) in data_dict.items():
            points = []
            for t, v in enumerate(data):
                if v is not None:
                    x = x_for_frame(t)
                    y = y_for_value(v)
                    y = max(plot_y, min(plot_y + plot_height, y))
                    points.append((x, y))
            
            for i in range(1, len(points)):
                cv2.line(img, points[i-1], points[i], color, 1)
            
            # Legend entry
            cv2.line(img, (legend_x, legend_y), (legend_x + 15, legend_y), color, 2)
            cv2.putText(img, label, (legend_x + 18, legend_y + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, color_text, 1)
            legend_y += 12
        
        # Y-axis ticks
        for i in range(5):
            y = plot_y + int(i * plot_height / 4)
            val = data_max - i * (data_max - data_min) / 4
            cv2.putText(img, f"{val:.1f}", (plot_x - 40, y + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.28, color_text, 1)
        
        # X-axis ticks
        for f in range(0, T, max(1, T // 8)):
            x = x_for_frame(f)
            cv2.putText(img, str(f), (x - 8, plot_y + plot_height + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, color_text, 1)
    
    # Panel 0: 2D Screen X (px)
    draw_multi_line_panel(0, "2D Screen X Position (px) - Ankle/Toe/Heel", {
        "L_Ank": (extract_component(joints_2d.get("l_ankle", []), 0), color_l_ankle),
        "L_Toe": (extract_component(joints_2d.get("l_toe", []), 0), color_l_toe),
        "L_Heel": (extract_component(joints_2d.get("l_heel", []), 0), color_l_heel),
        "R_Ank": (extract_component(joints_2d.get("r_ankle", []), 0), color_r_ankle),
        "R_Toe": (extract_component(joints_2d.get("r_toe", []), 0), color_r_toe),
        "R_Heel": (extract_component(joints_2d.get("r_heel", []), 0), color_r_heel),
    }, "X (px)")
    
    # Panel 1: 2D Screen Y (px)
    draw_multi_line_panel(1, "2D Screen Y Position (px) - Ankle/Toe/Heel", {
        "L_Ank": (extract_component(joints_2d.get("l_ankle", []), 1), color_l_ankle),
        "L_Toe": (extract_component(joints_2d.get("l_toe", []), 1), color_l_toe),
        "L_Heel": (extract_component(joints_2d.get("l_heel", []), 1), color_l_heel),
        "R_Ank": (extract_component(joints_2d.get("r_ankle", []), 1), color_r_ankle),
        "R_Toe": (extract_component(joints_2d.get("r_toe", []), 1), color_r_toe),
        "R_Heel": (extract_component(joints_2d.get("r_heel", []), 1), color_r_heel),
    }, "Y (px)")
    
    # Panel 2: 3D World X (m)
    draw_multi_line_panel(2, "3D World X Position (m) - Ankle/Toe/Heel", {
        "L_Ank": (extract_component(joints_3d.get("l_ankle", []), 0), color_l_ankle),
        "L_Toe": (extract_component(joints_3d.get("l_toe", []), 0), color_l_toe),
        "L_Heel": (extract_component(joints_3d.get("l_heel", []), 0), color_l_heel),
        "R_Ank": (extract_component(joints_3d.get("r_ankle", []), 0), color_r_ankle),
        "R_Toe": (extract_component(joints_3d.get("r_toe", []), 0), color_r_toe),
        "R_Heel": (extract_component(joints_3d.get("r_heel", []), 0), color_r_heel),
    }, "X (m)")
    
    # Panel 3: 3D World Y (m) - Height
    draw_multi_line_panel(3, "3D World Y Position (m) - HEIGHT - Ankle/Toe/Heel", {
        "L_Ank": (extract_component(joints_3d.get("l_ankle", []), 1), color_l_ankle),
        "L_Toe": (extract_component(joints_3d.get("l_toe", []), 1), color_l_toe),
        "L_Heel": (extract_component(joints_3d.get("l_heel", []), 1), color_l_heel),
        "R_Ank": (extract_component(joints_3d.get("r_ankle", []), 1), color_r_ankle),
        "R_Toe": (extract_component(joints_3d.get("r_toe", []), 1), color_r_toe),
        "R_Heel": (extract_component(joints_3d.get("r_heel", []), 1), color_r_heel),
        "Pelvis": (extract_component(joints_3d.get("pelvis", []), 1), color_pelvis),
    }, "Y (m)")
    
    # Panel 4: 3D World Z (m) - Depth
    draw_multi_line_panel(4, "3D World Z Position (m) - DEPTH - Ankle/Toe/Heel", {
        "L_Ank": (extract_component(joints_3d.get("l_ankle", []), 2), color_l_ankle),
        "L_Toe": (extract_component(joints_3d.get("l_toe", []), 2), color_l_toe),
        "L_Heel": (extract_component(joints_3d.get("l_heel", []), 2), color_l_heel),
        "R_Ank": (extract_component(joints_3d.get("r_ankle", []), 2), color_r_ankle),
        "R_Toe": (extract_component(joints_3d.get("r_toe", []), 2), color_r_toe),
        "R_Heel": (extract_component(joints_3d.get("r_heel", []), 2), color_r_heel),
    }, "Z (m)")
    
    # Panel 5: Foot-to-Pelvis Distance
    draw_multi_line_panel(5, "Foot-to-Pelvis Distance (m) - Contact = Increasing", {
        "Left": (left_to_pelvis, color_l_ankle),
        "Right": (right_to_pelvis, color_r_ankle),
    }, "Dist (m)")
    
    # Panel 6: Geometry Difference from Reference (used for contact detection)
    geom_diff_left = debug_data.get("geometry_diff", {}).get("left", [])
    geom_diff_right = debug_data.get("geometry_diff", {}).get("right", [])
    threshold = 0.03  # Default geometry_threshold
    
    # Draw with threshold line
    draw_multi_line_panel(6, f"Geometry Diff from Reference (m) - Below threshold = CONTACT", {
        "Left": (geom_diff_left, color_l_ankle),
        "Right": (geom_diff_right, color_r_ankle),
    }, "Diff (m)")
    
    # Add threshold line to panel 6
    panel_idx = 6
    y_offset = panel_idx * panel_height
    plot_x = margin_left
    plot_y = y_offset + margin_top
    
    # Get data range for threshold line positioning
    all_geom = [g for g in geom_diff_left + geom_diff_right if g is not None]
    if all_geom:
        data_min = min(all_geom)
        data_max = max(all_geom)
        margin_val = (data_max - data_min) * 0.1 + 0.001
        data_min -= margin_val
        data_max += margin_val
        
        # Draw threshold line
        if data_max > data_min:
            thresh_y = int(plot_y + plot_height - ((threshold - data_min) / (data_max - data_min)) * plot_height)
            if plot_y <= thresh_y <= plot_y + plot_height:
                cv2.line(img, (plot_x, thresh_y), (plot_x + plot_width, thresh_y), (0, 0, 255), 2)
                cv2.putText(img, f"threshold={threshold:.3f}", (plot_x + plot_width - 120, thresh_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    # Panel 7: Contact Timeline
    panel_idx = 7
    y_offset = panel_idx * panel_height
    cv2.rectangle(img, (0, y_offset), (width, y_offset + panel_height), (252, 252, 252), -1)
    cv2.rectangle(img, (0, y_offset), (width, y_offset + panel_height), (200, 200, 200), 1)
    cv2.putText(img, "Contact Timeline - Green=Contact, Gray=Flight (Anatomical L/R)", 
               (margin_left, y_offset + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_text, 1)
    
    plot_x = margin_left
    plot_y = y_offset + margin_top
    bar_height = (plot_height - 15) // 2
    
    # Left foot contacts
    cv2.putText(img, "Left:", (10, plot_y + bar_height // 2 + 4), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, color_text, 1)
    for t in range(T):
        x = int(plot_x + (t / max(T - 1, 1)) * plot_width)
        color = color_contact if contacts[t, 0] else color_flight
        cv2.line(img, (x, plot_y), (x, plot_y + bar_height), color, max(1, plot_width // T))
    
    # Right foot contacts
    cv2.putText(img, "Right:", (10, plot_y + bar_height + 12 + bar_height // 2 + 4), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, color_text, 1)
    for t in range(T):
        x = int(plot_x + (t / max(T - 1, 1)) * plot_width)
        color = color_contact if contacts[t, 1] else color_flight
        cv2.line(img, (x, plot_y + bar_height + 12), (x, plot_y + 2 * bar_height + 12), color, max(1, plot_width // T))
    
    # X-axis
    for f in range(0, T, max(1, T // 8)):
        x = int(plot_x + (f / max(T - 1, 1)) * plot_width)
        cv2.putText(img, str(f), (x - 8, y_offset + panel_height - 3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.25, color_text, 1)
    
    return img


# =============================================================================
# ComfyUI Node
# =============================================================================

class KinematicContactNode:
    """
    Kinematic Contact Detector - Reference-based foot contact detection.
    
    Detects foot contacts by comparing foot geometry to a reference frame
    where the foot is known to be flat on the ground.
    
    Method:
    1. Auto-detect (or user-specify) reference frames where each foot is flat
    2. Compare each frame's foot geometry to reference
    3. Normalize by camera distance (pred_cam_t) for depth changes
    
    Works for walking, running, sprinting - any bipedal gait.
    Robust to character moving toward/away from camera.
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
                # Reference frames (NEW)
                "left_ref_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "tooltip": "Frame where LEFT foot is flat on ground (-1 = auto-detect)"
                }),
                "right_ref_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "tooltip": "Frame where RIGHT foot is flat on ground (-1 = auto-detect)"
                }),
                
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
                "geometry_threshold": ("FLOAT", {
                    "default": 0.03,
                    "min": 0.005,
                    "max": 0.15,
                    "step": 0.005,
                    "tooltip": "Max geometry difference from reference to count as contact (meters). Lower = stricter."
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
        left_ref_frame: int = -1,
        right_ref_frame: int = -1,
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
        geometry_threshold: float = 0.03,
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
        """Process mesh sequence with reference-based contact detection."""
        
        log = KinematicLogger(log_level)
        log.info("=" * 60)
        log.info("KINEMATIC CONTACT DETECTION (Reference-Based)")
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
            height_threshold=geometry_threshold,  # Reuse this field for geometry threshold
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
        
        # Log reference frame settings
        if left_ref_frame >= 0:
            log.info(f"Left foot reference frame: {left_ref_frame} (user-specified)")
        else:
            log.info("Left foot reference frame: auto-detect")
        if right_ref_frame >= 0:
            log.info(f"Right foot reference frame: {right_ref_frame} (user-specified)")
        else:
            log.info("Right foot reference frame: auto-detect")
        
        # Detect contacts
        detector = KinematicContactDetector(config, log)
        contacts, confidence, debug_data = detector.detect(
            frames, global_intrinsics, 
            left_ref_frame=left_ref_frame,
            right_ref_frame=right_ref_frame
        )
        
        # Log reference frames used
        ref_left = debug_data.get("reference_frames", {}).get("left", "N/A")
        ref_right = debug_data.get("reference_frames", {}).get("right", "N/A")
        log.info(f"Reference frames used: Left={ref_left}, Right={ref_right}")
        
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
