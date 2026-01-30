"""
Foot Contact Detection Test Node (v3 - Ground Plane Based)
==========================================================

Detects foot ground contact using skeleton joints with a proper ground plane.

Key Concepts:
1. Single ground plane defined by calibration frames
2. Left and right ground points connected to form plane with perspective
3. Each joint checked against interpolated ground height
4. Anatomically correct: toe is lowest, then ball, then ankle

Joint Indices (confirmed):
    Left:  ankle=17, ball=15, toe=16
    Right: ankle=14, ball=18, toe=19

Usage:
1. Connect mesh_sequence and images
2. Provide ONE calibration frame per foot (where foot is FLAT on ground)
3. View debug overlay showing ground plane and contact states
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Import logger
try:
    from ..lib.logger import log, set_module
    set_module("Foot Contact Test")
except ImportError:
    class _FallbackLog:
        def info(self, msg): print(f"[Foot Contact Test] {msg}")
        def debug(self, msg): pass
        def warn(self, msg): print(f"[Foot Contact Test] WARN: {msg}")
        def error(self, msg): print(f"[Foot Contact Test] ERROR: {msg}")
    log = _FallbackLog()


# Confirmed joint indices
JOINT_INDICES = {
    # Left foot
    "left_ankle": 17,
    "left_ball": 15,
    "left_toe": 16,
    # Right foot
    "right_ankle": 14,
    "right_ball": 18,
    "right_toe": 19,
}


@dataclass
class GroundPlane:
    """Ground plane defined by left and right foot ground positions."""
    left_ground_y: float
    right_ground_y: float
    left_x: float  # X position of left foot at calibration
    right_x: float  # X position of right foot at calibration
    
    @property
    def slope(self) -> float:
        """Ground plane slope (difference between left and right)."""
        return self.right_ground_y - self.left_ground_y
    
    @property
    def average_ground_y(self) -> float:
        """Average ground height."""
        return (self.left_ground_y + self.right_ground_y) / 2
    
    def get_ground_y_at_x(self, x: float) -> float:
        """
        Interpolate ground height at given X position.
        
        Uses linear interpolation between left and right ground points.
        Extrapolates for positions outside the foot range.
        """
        if abs(self.right_x - self.left_x) < 0.001:
            # Feet at same X position - use average
            return self.average_ground_y
        
        # Linear interpolation/extrapolation
        t = (x - self.left_x) / (self.right_x - self.left_x)
        return self.left_ground_y + t * (self.right_ground_y - self.left_ground_y)


@dataclass 
class JointState:
    """State of a single joint relative to ground."""
    name: str
    y: float
    x: float
    ground_y: float
    distance: float  # Positive = above ground, negative = below
    velocity: float
    grounded: bool


@dataclass
class FootState:
    """State of one foot (3 joints)."""
    ankle: JointState
    ball: JointState
    toe: JointState
    
    @property
    def any_grounded(self) -> bool:
        """True if any joint is grounded."""
        return self.ankle.grounded or self.ball.grounded or self.toe.grounded
    
    @property
    def lowest_joint(self) -> JointState:
        """Return the joint closest to ground."""
        joints = [self.toe, self.ball, self.ankle]
        return min(joints, key=lambda j: j.distance)
    
    @property
    def contact_type(self) -> str:
        """Return contact type based on which joints are grounded."""
        if self.toe.grounded and self.ball.grounded and self.ankle.grounded:
            return "full"
        elif self.toe.grounded and self.ball.grounded:
            return "forefoot"
        elif self.toe.grounded:
            return "toe"
        elif self.ankle.grounded:
            return "heel"
        elif self.ball.grounded:
            return "ball"
        else:
            return "air"


class FootContactTest:
    """
    Test node for foot contact detection with ground plane.
    
    Uses a single ground plane defined by two calibration frames
    (one per foot, where each foot is flat on ground).
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE",),
                "images": ("IMAGE",),
            },
            "optional": {
                # Calibration frames (one per foot)
                "left_foot_ground_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "tooltip": "Frame where LEFT foot is FLAT on ground"
                }),
                "right_foot_ground_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "tooltip": "Frame where RIGHT foot is FLAT on ground"
                }),
                # Detection parameters
                "height_threshold": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.01,
                    "max": 0.3,
                    "step": 0.01,
                    "tooltip": "Distance from ground plane to count as contact (meters)"
                }),
                "velocity_threshold": ("FLOAT", {
                    "default": 0.03,
                    "min": 0.005,
                    "max": 0.2,
                    "step": 0.005,
                    "tooltip": "Max joint velocity to count as grounded (m/frame)"
                }),
                "min_contact_frames": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Minimum consecutive frames for valid contact"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MESH_SEQUENCE", "STRING")
    RETURN_NAMES = ("debug_overlay", "mesh_sequence", "status")
    FUNCTION = "test_detection"
    CATEGORY = "SAM3DBody2abc/Testing"
    
    def _get_joint_position(self, joint_coords: np.ndarray, joint_idx: int) -> np.ndarray:
        """Get position of a specific joint."""
        if joint_coords.ndim == 3:
            joint_coords = joint_coords[0]
        
        if joint_idx >= len(joint_coords):
            log.warn(f"Joint index {joint_idx} out of range (max {len(joint_coords)-1})")
            return np.array([0.0, 0.0, 0.0])
        
        return joint_coords[joint_idx].copy()
    
    def _compute_velocity(self, curr_pos: np.ndarray, prev_pos: np.ndarray) -> float:
        """Compute velocity magnitude between two positions."""
        return float(np.linalg.norm(curr_pos - prev_pos))
    
    def _calibrate_ground_plane(
        self,
        frames: List[Dict],
        left_frame_idx: int,
        right_frame_idx: int,
    ) -> GroundPlane:
        """
        Calibrate ground plane from two frames.
        
        For each foot, finds the lowest joint Y position as ground reference.
        """
        # Get left foot ground (lowest joint in calibration frame)
        left_joint_coords = frames[left_frame_idx].get("joint_coords")
        if left_joint_coords is None:
            raise ValueError(f"No joint_coords in frame {left_frame_idx}")
        
        left_toe = self._get_joint_position(left_joint_coords, JOINT_INDICES["left_toe"])
        left_ball = self._get_joint_position(left_joint_coords, JOINT_INDICES["left_ball"])
        left_ankle = self._get_joint_position(left_joint_coords, JOINT_INDICES["left_ankle"])
        
        # Ground = lowest point of left foot
        left_ground_y = min(left_toe[1], left_ball[1], left_ankle[1])
        left_x = (left_toe[0] + left_ball[0] + left_ankle[0]) / 3  # Average X
        
        # Get right foot ground
        right_joint_coords = frames[right_frame_idx].get("joint_coords")
        if right_joint_coords is None:
            raise ValueError(f"No joint_coords in frame {right_frame_idx}")
        
        right_toe = self._get_joint_position(right_joint_coords, JOINT_INDICES["right_toe"])
        right_ball = self._get_joint_position(right_joint_coords, JOINT_INDICES["right_ball"])
        right_ankle = self._get_joint_position(right_joint_coords, JOINT_INDICES["right_ankle"])
        
        # Ground = lowest point of right foot
        right_ground_y = min(right_toe[1], right_ball[1], right_ankle[1])
        right_x = (right_toe[0] + right_ball[0] + right_ankle[0]) / 3
        
        log.info(f"Left foot ground: Y={left_ground_y:.4f} at X={left_x:.4f} (frame {left_frame_idx})")
        log.info(f"Right foot ground: Y={right_ground_y:.4f} at X={right_x:.4f} (frame {right_frame_idx})")
        log.info(f"Ground plane slope: {right_ground_y - left_ground_y:.4f}")
        
        return GroundPlane(
            left_ground_y=left_ground_y,
            right_ground_y=right_ground_y,
            left_x=left_x,
            right_x=right_x,
        )
    
    def _detect_foot_contact(
        self,
        joint_coords: np.ndarray,
        ground_plane: GroundPlane,
        prev_positions: Dict[str, np.ndarray],
        side: str,  # "left" or "right"
        height_threshold: float,
        velocity_threshold: float,
        is_first_frame: bool,
    ) -> Tuple[FootState, Dict[str, np.ndarray]]:
        """Detect contact state for one foot."""
        
        # Get joint indices for this side
        ankle_idx = JOINT_INDICES[f"{side}_ankle"]
        ball_idx = JOINT_INDICES[f"{side}_ball"]
        toe_idx = JOINT_INDICES[f"{side}_toe"]
        
        # Get current positions
        ankle_pos = self._get_joint_position(joint_coords, ankle_idx)
        ball_pos = self._get_joint_position(joint_coords, ball_idx)
        toe_pos = self._get_joint_position(joint_coords, toe_idx)
        
        # Compute velocities
        if is_first_frame:
            ankle_vel = ball_vel = toe_vel = 0.0
        else:
            ankle_vel = self._compute_velocity(ankle_pos, prev_positions.get(f"{side}_ankle", ankle_pos))
            ball_vel = self._compute_velocity(ball_pos, prev_positions.get(f"{side}_ball", ball_pos))
            toe_vel = self._compute_velocity(toe_pos, prev_positions.get(f"{side}_toe", toe_pos))
        
        # Get ground Y at each joint's X position (interpolated from ground plane)
        ankle_ground_y = ground_plane.get_ground_y_at_x(ankle_pos[0])
        ball_ground_y = ground_plane.get_ground_y_at_x(ball_pos[0])
        toe_ground_y = ground_plane.get_ground_y_at_x(toe_pos[0])
        
        # Calculate distances from ground
        ankle_dist = ankle_pos[1] - ankle_ground_y
        ball_dist = ball_pos[1] - ball_ground_y
        toe_dist = toe_pos[1] - toe_ground_y
        
        # Determine contact for each joint
        def is_grounded(dist: float, vel: float) -> bool:
            height_ok = dist <= height_threshold
            velocity_ok = is_first_frame or (vel <= velocity_threshold)
            return height_ok and velocity_ok
        
        ankle_grounded = is_grounded(ankle_dist, ankle_vel)
        ball_grounded = is_grounded(ball_dist, ball_vel)
        toe_grounded = is_grounded(toe_dist, toe_vel)
        
        # Build joint states
        ankle_state = JointState(
            name="ankle", y=ankle_pos[1], x=ankle_pos[0],
            ground_y=ankle_ground_y, distance=ankle_dist,
            velocity=ankle_vel, grounded=ankle_grounded
        )
        ball_state = JointState(
            name="ball", y=ball_pos[1], x=ball_pos[0],
            ground_y=ball_ground_y, distance=ball_dist,
            velocity=ball_vel, grounded=ball_grounded
        )
        toe_state = JointState(
            name="toe", y=toe_pos[1], x=toe_pos[0],
            ground_y=toe_ground_y, distance=toe_dist,
            velocity=toe_vel, grounded=toe_grounded
        )
        
        # Update previous positions
        new_prev = {
            f"{side}_ankle": ankle_pos,
            f"{side}_ball": ball_pos,
            f"{side}_toe": toe_pos,
        }
        
        return FootState(ankle=ankle_state, ball=ball_state, toe=toe_state), new_prev
    
    def _apply_min_contact_frames(
        self,
        contact_history: List[bool],
        min_frames: int
    ) -> List[bool]:
        """Apply minimum contact frames filter."""
        if min_frames <= 1:
            return contact_history
        
        result = []
        consecutive = 0
        
        for contact in contact_history:
            if contact:
                consecutive += 1
            else:
                consecutive = 0
            result.append(consecutive >= min_frames)
        
        return result
    
    def _draw_debug_overlay(
        self,
        image: np.ndarray,
        frame_idx: int,
        left_state: FootState,
        right_state: FootState,
        ground_plane: GroundPlane,
        height_threshold: float,
        velocity_threshold: float,
        min_contact_frames: int,
    ) -> np.ndarray:
        """Draw debug visualization on frame."""
        
        # Make copy and convert to uint8 if needed
        if image.dtype == np.float32 or image.dtype == np.float64:
            frame = (image * 255).astype(np.uint8).copy()
        else:
            frame = image.copy()
        
        # Ensure BGR format for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        h, w = frame.shape[:2]
        
        # Colors
        COLOR_GROUNDED = (0, 255, 0)      # Green
        COLOR_AIRBORNE = (0, 0, 255)      # Red
        COLOR_PARTIAL = (0, 255, 255)     # Yellow
        COLOR_GROUND_LINE = (255, 255, 0) # Cyan
        COLOR_TEXT = (255, 255, 255)      # White
        COLOR_TEXT_BG = (0, 0, 0)         # Black
        COLOR_GRAY = (180, 180, 180)      # Gray
        
        # Overall contact state
        left_grounded = left_state.any_grounded
        right_grounded = right_state.any_grounded
        
        if left_grounded and right_grounded:
            overall_state = "BOTH"
            overall_color = COLOR_GROUNDED
        elif left_grounded:
            overall_state = "LEFT"
            overall_color = COLOR_PARTIAL
        elif right_grounded:
            overall_state = "RIGHT"
            overall_color = COLOR_PARTIAL
        else:
            overall_state = "NONE"
            overall_color = COLOR_AIRBORNE
        
        # Draw info panel background
        panel_height = 320
        panel_width = 380
        cv2.rectangle(frame, (5, 5), (panel_width, panel_height), COLOR_TEXT_BG, -1)
        cv2.rectangle(frame, (5, 5), (panel_width, panel_height), COLOR_TEXT, 1)
        
        # Draw text info
        y_offset = 25
        line_height = 18
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Frame number and overall state
        cv2.putText(frame, f"Frame: {frame_idx}", (15, y_offset), font, 0.5, COLOR_TEXT, 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Contact: {overall_state}", (15, y_offset), font, 0.7, overall_color, 2)
        y_offset += line_height + 5
        
        # Ground plane info
        cv2.putText(frame, f"Ground Plane: L={ground_plane.left_ground_y:.3f} R={ground_plane.right_ground_y:.3f}", 
                   (15, y_offset), font, 0.4, COLOR_GROUND_LINE, 1)
        y_offset += line_height - 2
        cv2.putText(frame, f"Slope: {ground_plane.slope:.4f}", (15, y_offset), font, 0.4, COLOR_GROUND_LINE, 1)
        y_offset += line_height + 2
        
        # Left foot details
        left_color = COLOR_GROUNDED if left_grounded else COLOR_AIRBORNE
        cv2.putText(frame, f"LEFT FOOT: {left_state.contact_type.upper()}", (15, y_offset), font, 0.5, left_color, 1)
        y_offset += line_height
        
        for joint in [left_state.toe, left_state.ball, left_state.ankle]:
            joint_color = COLOR_GROUNDED if joint.grounded else COLOR_AIRBORNE
            status = "GND" if joint.grounded else "AIR"
            cv2.putText(frame, f"  {joint.name.capitalize():6s}: {status}", (15, y_offset), font, 0.4, joint_color, 1)
            cv2.putText(frame, f"dist={joint.distance:+.3f} vel={joint.velocity:.3f}", (140, y_offset), font, 0.35, COLOR_GRAY, 1)
            y_offset += line_height - 2
        
        y_offset += 5
        
        # Right foot details
        right_color = COLOR_GROUNDED if right_grounded else COLOR_AIRBORNE
        cv2.putText(frame, f"RIGHT FOOT: {right_state.contact_type.upper()}", (15, y_offset), font, 0.5, right_color, 1)
        y_offset += line_height
        
        for joint in [right_state.toe, right_state.ball, right_state.ankle]:
            joint_color = COLOR_GROUNDED if joint.grounded else COLOR_AIRBORNE
            status = "GND" if joint.grounded else "AIR"
            cv2.putText(frame, f"  {joint.name.capitalize():6s}: {status}", (15, y_offset), font, 0.4, joint_color, 1)
            cv2.putText(frame, f"dist={joint.distance:+.3f} vel={joint.velocity:.3f}", (140, y_offset), font, 0.35, COLOR_GRAY, 1)
            y_offset += line_height - 2
        
        y_offset += 5
        
        # Thresholds
        cv2.putText(frame, f"Thresholds: height={height_threshold:.3f}m  vel={velocity_threshold:.3f}m/f", 
                   (15, y_offset), font, 0.35, COLOR_GRAY, 1)
        y_offset += line_height - 2
        cv2.putText(frame, f"Min contact frames: {min_contact_frames}", (15, y_offset), font, 0.35, COLOR_GRAY, 1)
        
        # Draw ground plane line at bottom (visual reference)
        ground_line_y = int(h * 0.85)
        cv2.line(frame, (0, ground_line_y), (w, ground_line_y), COLOR_GROUND_LINE, 2)
        cv2.putText(frame, "Ground Plane Reference", (10, ground_line_y - 10), font, 0.4, COLOR_GROUND_LINE, 1)
        
        # Convert back to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame
    
    def test_detection(
        self,
        mesh_sequence: Dict,
        images: torch.Tensor,
        left_foot_ground_frame: int = 0,
        right_foot_ground_frame: int = 0,
        height_threshold: float = 0.05,
        velocity_threshold: float = 0.03,
        min_contact_frames: int = 1,
    ) -> Tuple[torch.Tensor, Dict, str]:
        """Test foot contact detection and generate debug overlay."""
        
        # Get frames data
        frames_data = mesh_sequence.get("frames", {})
        if isinstance(frames_data, dict):
            frame_keys = sorted(frames_data.keys())
            frames = [frames_data[k] for k in frame_keys]
        else:
            frames = frames_data
        
        num_frames = len(frames)
        log.info(f"Processing {num_frames} frames with ground plane detection")
        
        # Validate frame indices
        max_frame = num_frames - 1
        if left_foot_ground_frame > max_frame:
            raise ValueError(f"left_foot_ground_frame={left_foot_ground_frame} out of range (max {max_frame})")
        if right_foot_ground_frame > max_frame:
            raise ValueError(f"right_foot_ground_frame={right_foot_ground_frame} out of range (max {max_frame})")
        
        # Calibrate ground plane
        ground_plane = self._calibrate_ground_plane(frames, left_foot_ground_frame, right_foot_ground_frame)
        
        log.info(f"Thresholds: height={height_threshold}m, velocity={velocity_threshold}m/frame, min_frames={min_contact_frames}")
        
        # Convert images to numpy
        images_np = images.cpu().numpy()
        if images_np.max() <= 1.0:
            images_np = (images_np * 255).astype(np.uint8)
        else:
            images_np = images_np.astype(np.uint8)
        
        # Process all frames
        all_states = []  # List of (left_state, right_state) per frame
        prev_positions = {}
        
        for i, frame_data in enumerate(frames):
            joint_coords = frame_data.get("joint_coords")
            
            if joint_coords is None:
                # No data - assume airborne
                dummy_joint = JointState("none", 0, 0, 0, 1.0, 0, False)
                left_state = FootState(dummy_joint, dummy_joint, dummy_joint)
                right_state = FootState(dummy_joint, dummy_joint, dummy_joint)
                all_states.append((left_state, right_state))
                continue
            
            is_first_frame = (i == 0)
            
            # Detect left foot
            left_state, left_prev = self._detect_foot_contact(
                joint_coords, ground_plane, prev_positions, "left",
                height_threshold, velocity_threshold, is_first_frame
            )
            
            # Detect right foot
            right_state, right_prev = self._detect_foot_contact(
                joint_coords, ground_plane, prev_positions, "right",
                height_threshold, velocity_threshold, is_first_frame
            )
            
            all_states.append((left_state, right_state))
            prev_positions.update(left_prev)
            prev_positions.update(right_prev)
        
        # Apply min_contact_frames filter if needed
        if min_contact_frames > 1:
            # Extract contact histories
            left_history = [s[0].any_grounded for s in all_states]
            right_history = [s[1].any_grounded for s in all_states]
            
            # Apply filter
            left_filtered = self._apply_min_contact_frames(left_history, min_contact_frames)
            right_filtered = self._apply_min_contact_frames(right_history, min_contact_frames)
            
            # Note: We keep individual joint states but overall grounded status is filtered
            # This is for display purposes - the filtered status overrides joint-level decisions
        else:
            left_filtered = [s[0].any_grounded for s in all_states]
            right_filtered = [s[1].any_grounded for s in all_states]
        
        # Generate output frames and statistics
        output_frames = []
        contact_states = []
        
        for i, (left_state, right_state) in enumerate(all_states):
            # Use filtered grounded status for overall contact
            left_grounded = left_filtered[i]
            right_grounded = right_filtered[i]
            
            if left_grounded and right_grounded:
                contact_state = "both"
            elif left_grounded:
                contact_state = "left"
            elif right_grounded:
                contact_state = "right"
            else:
                contact_state = "none"
            
            contact_states.append(contact_state)
            
            # Draw debug overlay
            if i < len(images_np):
                overlay = self._draw_debug_overlay(
                    image=images_np[i],
                    frame_idx=i,
                    left_state=left_state,
                    right_state=right_state,
                    ground_plane=ground_plane,
                    height_threshold=height_threshold,
                    velocity_threshold=velocity_threshold,
                    min_contact_frames=min_contact_frames,
                )
                output_frames.append(overlay)
        
        # Build statistics
        both_count = contact_states.count("both")
        left_count = contact_states.count("left")
        right_count = contact_states.count("right")
        none_count = contact_states.count("none")
        
        status = (
            f"Ground plane: L={ground_plane.left_ground_y:.3f} R={ground_plane.right_ground_y:.3f} | "
            f"Contacts: both={both_count}, left={left_count}, right={right_count}, air={none_count} "
            f"({num_frames} frames)"
        )
        log.info(status)
        
        # Add contact data to mesh_sequence for downstream nodes
        result_sequence = mesh_sequence.copy()
        result_sequence["foot_contact"] = {
            "contact_states": contact_states,
            "ground_plane": {
                "left_ground_y": ground_plane.left_ground_y,
                "right_ground_y": ground_plane.right_ground_y,
                "left_x": ground_plane.left_x,
                "right_x": ground_plane.right_x,
                "slope": ground_plane.slope,
            },
            "calibration_frames": {
                "left": left_foot_ground_frame,
                "right": right_foot_ground_frame,
            },
            "thresholds": {
                "height": height_threshold,
                "velocity": velocity_threshold,
                "min_frames": min_contact_frames,
            },
            "joint_indices": JOINT_INDICES,
            "per_frame": [
                {
                    "left": {
                        "grounded": left_filtered[i],
                        "contact_type": left_state.contact_type,
                        "toe": {"grounded": left_state.toe.grounded, "distance": left_state.toe.distance, "velocity": left_state.toe.velocity},
                        "ball": {"grounded": left_state.ball.grounded, "distance": left_state.ball.distance, "velocity": left_state.ball.velocity},
                        "ankle": {"grounded": left_state.ankle.grounded, "distance": left_state.ankle.distance, "velocity": left_state.ankle.velocity},
                    },
                    "right": {
                        "grounded": right_filtered[i],
                        "contact_type": right_state.contact_type,
                        "toe": {"grounded": right_state.toe.grounded, "distance": right_state.toe.distance, "velocity": right_state.toe.velocity},
                        "ball": {"grounded": right_state.ball.grounded, "distance": right_state.ball.distance, "velocity": right_state.ball.velocity},
                        "ankle": {"grounded": right_state.ankle.grounded, "distance": right_state.ankle.distance, "velocity": right_state.ankle.velocity},
                    },
                }
                for i, (left_state, right_state) in enumerate(all_states)
            ],
        }
        
        # Convert output to tensor
        output_array = np.stack(output_frames, axis=0)
        output_tensor = torch.from_numpy(output_array).float() / 255.0
        
        return (output_tensor, result_sequence, status)


NODE_CLASS_MAPPINGS = {
    "FootContactTest": FootContactTest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FootContactTest": "🦶 Foot Contact Test (Calibration)",
}
