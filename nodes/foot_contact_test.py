"""
Foot Contact Detection Test Node (v2 - Joint Based)
====================================================

Standalone node to test foot contact detection using skeleton joints.
Tracks ankle, ball, and toe joints for accurate contact detection.

Joint Indices (confirmed):
    Left:  ankle=17, ball=15, toe=16
    Right: ankle=14, ball=18, toe=19

Usage:
1. Connect mesh_sequence and images
2. Provide calibration frames for each joint (minimum 1 per foot)
3. View debug overlay showing contact states per joint
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
class JointContactState:
    """Contact state for a single joint."""
    grounded: bool
    distance: float
    velocity: float
    ground_y: float
    current_y: float


@dataclass
class FootContactState:
    """Contact state for one foot (3 joints)."""
    ankle: JointContactState
    ball: JointContactState
    toe: JointContactState
    
    @property
    def any_grounded(self) -> bool:
        """True if any joint is grounded."""
        return self.ankle.grounded or self.ball.grounded or self.toe.grounded
    
    @property
    def all_grounded(self) -> bool:
        """True if all joints are grounded."""
        return self.ankle.grounded and self.ball.grounded and self.toe.grounded
    
    @property
    def contact_type(self) -> str:
        """Return contact type: full, ankle, ball, toe, or none."""
        if self.all_grounded:
            return "full"
        elif self.ankle.grounded and not self.ball.grounded and not self.toe.grounded:
            return "ankle"
        elif self.ball.grounded and not self.ankle.grounded and not self.toe.grounded:
            return "ball"
        elif self.toe.grounded and not self.ankle.grounded and not self.ball.grounded:
            return "toe"
        elif self.ankle.grounded or self.ball.grounded or self.toe.grounded:
            return "partial"
        else:
            return "none"


class FootContactTest:
    """
    Test node for foot contact detection using skeleton joints.
    
    Tracks 3 joints per foot:
    - Ankle (joint 17 left, 14 right) - heel area
    - Ball (joint 15 left, 18 right) - mid-foot
    - Toe (joint 16 left, 19 right) - toe tip
    
    Contact = ANY of the 3 joints is grounded
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE",),
                "images": ("IMAGE",),
            },
            "optional": {
                # Left foot calibration
                "left_ankle_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "tooltip": "Frame where LEFT ANKLE touches ground (-1 = not provided)"
                }),
                "left_ball_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "tooltip": "Frame where LEFT BALL touches ground (-1 = not provided)"
                }),
                "left_toe_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "tooltip": "Frame where LEFT TOE touches ground (-1 = not provided)"
                }),
                # Right foot calibration
                "right_ankle_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "tooltip": "Frame where RIGHT ANKLE touches ground (-1 = not provided)"
                }),
                "right_ball_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "tooltip": "Frame where RIGHT BALL touches ground (-1 = not provided)"
                }),
                "right_toe_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "tooltip": "Frame where RIGHT TOE touches ground (-1 = not provided)"
                }),
                # Detection parameters
                "height_threshold": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.01,
                    "max": 0.3,
                    "step": 0.01,
                    "tooltip": "Distance from calibrated ground to count as contact (meters)"
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
    
    def _detect_joint_contact(
        self,
        current_y: float,
        ground_y: float,
        velocity: float,
        height_threshold: float,
        velocity_threshold: float,
        is_first_frame: bool = False,
    ) -> JointContactState:
        """Detect contact state for a single joint."""
        distance = current_y - ground_y
        
        # Height check
        height_ok = distance <= height_threshold
        
        # Velocity check (skip for first frame)
        if is_first_frame:
            velocity_ok = True
        else:
            velocity_ok = velocity <= velocity_threshold
        
        grounded = height_ok and velocity_ok
        
        return JointContactState(
            grounded=grounded,
            distance=distance,
            velocity=velocity,
            ground_y=ground_y,
            current_y=current_y,
        )
    
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
        
        for i, contact in enumerate(contact_history):
            if contact:
                consecutive += 1
            else:
                consecutive = 0
            
            # Only mark as grounded if we've had enough consecutive frames
            result.append(consecutive >= min_frames)
        
        return result
    
    def _draw_debug_overlay(
        self,
        image: np.ndarray,
        frame_idx: int,
        left_state: FootContactState,
        right_state: FootContactState,
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
        panel_height = 280
        panel_width = 350
        cv2.rectangle(frame, (5, 5), (panel_width, panel_height), COLOR_TEXT_BG, -1)
        cv2.rectangle(frame, (5, 5), (panel_width, panel_height), COLOR_TEXT, 1)
        
        # Draw text info
        y_offset = 25
        line_height = 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        
        # Frame number and overall state
        cv2.putText(frame, f"Frame: {frame_idx}", (15, y_offset),
                   font, font_scale, COLOR_TEXT, 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Contact: {overall_state}", (15, y_offset),
                   font, 0.7, overall_color, 2)
        y_offset += line_height + 5
        
        # Left foot details
        left_color = COLOR_GROUNDED if left_grounded else COLOR_AIRBORNE
        cv2.putText(frame, f"LEFT FOOT: {left_state.contact_type.upper()}", (15, y_offset),
                   font, font_scale, left_color, 1)
        y_offset += line_height
        
        # Left joints
        for name, state in [("Ankle", left_state.ankle), ("Ball", left_state.ball), ("Toe", left_state.toe)]:
            joint_color = COLOR_GROUNDED if state.grounded else COLOR_AIRBORNE
            status = "GND" if state.grounded else "AIR"
            cv2.putText(frame, f"  {name}: {status}", (15, y_offset),
                       font, 0.45, joint_color, 1)
            cv2.putText(frame, f"d={state.distance:.3f} v={state.velocity:.3f}", (120, y_offset),
                       font, 0.4, COLOR_GRAY, 1)
            y_offset += line_height - 2
        
        y_offset += 5
        
        # Right foot details
        right_color = COLOR_GROUNDED if right_grounded else COLOR_AIRBORNE
        cv2.putText(frame, f"RIGHT FOOT: {right_state.contact_type.upper()}", (15, y_offset),
                   font, font_scale, right_color, 1)
        y_offset += line_height
        
        # Right joints
        for name, state in [("Ankle", right_state.ankle), ("Ball", right_state.ball), ("Toe", right_state.toe)]:
            joint_color = COLOR_GROUNDED if state.grounded else COLOR_AIRBORNE
            status = "GND" if state.grounded else "AIR"
            cv2.putText(frame, f"  {name}: {status}", (15, y_offset),
                       font, 0.45, joint_color, 1)
            cv2.putText(frame, f"d={state.distance:.3f} v={state.velocity:.3f}", (120, y_offset),
                       font, 0.4, COLOR_GRAY, 1)
            y_offset += line_height - 2
        
        y_offset += 5
        
        # Thresholds
        cv2.putText(frame, f"Thresholds: h={height_threshold:.3f}m v={velocity_threshold:.3f}m/f", (15, y_offset),
                   font, 0.4, COLOR_GRAY, 1)
        y_offset += line_height - 2
        cv2.putText(frame, f"Min contact frames: {min_contact_frames}", (15, y_offset),
                   font, 0.4, COLOR_GRAY, 1)
        
        # Draw ground plane line at bottom
        ground_line_y = int(h * 0.85)
        cv2.line(frame, (0, ground_line_y), (w, ground_line_y), COLOR_GROUND_LINE, 2)
        cv2.putText(frame, "Ground Plane", (10, ground_line_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GROUND_LINE, 1)
        
        # Convert back to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame
    
    def test_detection(
        self,
        mesh_sequence: Dict,
        images: torch.Tensor,
        left_ankle_frame: int = -1,
        left_ball_frame: int = -1,
        left_toe_frame: int = -1,
        right_ankle_frame: int = -1,
        right_ball_frame: int = -1,
        right_toe_frame: int = -1,
        height_threshold: float = 0.05,
        velocity_threshold: float = 0.03,
        min_contact_frames: int = 1,
    ) -> Tuple[torch.Tensor, Dict, str]:
        """Test foot contact detection and generate debug overlay."""
        
        # Validate: minimum 1 calibration frame per foot
        left_provided = sum([
            left_ankle_frame >= 0,
            left_ball_frame >= 0,
            left_toe_frame >= 0,
        ])
        right_provided = sum([
            right_ankle_frame >= 0,
            right_ball_frame >= 0,
            right_toe_frame >= 0,
        ])
        
        if left_provided < 1 or right_provided < 1:
            raise ValueError(
                "Please provide at least 1 calibration frame per foot. "
                f"Got: left={left_provided}, right={right_provided}"
            )
        
        # Get frames data
        frames_data = mesh_sequence.get("frames", {})
        if isinstance(frames_data, dict):
            frame_keys = sorted(frames_data.keys())
            frames = [frames_data[k] for k in frame_keys]
        else:
            frames = frames_data
        
        num_frames = len(frames)
        log.info(f"Processing {num_frames} frames with joint-based detection")
        log.info(f"Left calibration: ankle={left_ankle_frame}, ball={left_ball_frame}, toe={left_toe_frame}")
        log.info(f"Right calibration: ankle={right_ankle_frame}, ball={right_ball_frame}, toe={right_toe_frame}")
        
        # Validate frame indices
        max_frame = num_frames - 1
        for name, idx in [
            ("left_ankle", left_ankle_frame),
            ("left_ball", left_ball_frame),
            ("left_toe", left_toe_frame),
            ("right_ankle", right_ankle_frame),
            ("right_ball", right_ball_frame),
            ("right_toe", right_toe_frame),
        ]:
            if idx > max_frame:
                raise ValueError(f"Calibration frame {name}={idx} out of range (max {max_frame})")
        
        # Extract calibration ground heights from joint_coords
        ground_heights = {
            "left_ankle": None,
            "left_ball": None,
            "left_toe": None,
            "right_ankle": None,
            "right_ball": None,
            "right_toe": None,
        }
        
        calibration_map = {
            "left_ankle": (left_ankle_frame, JOINT_INDICES["left_ankle"]),
            "left_ball": (left_ball_frame, JOINT_INDICES["left_ball"]),
            "left_toe": (left_toe_frame, JOINT_INDICES["left_toe"]),
            "right_ankle": (right_ankle_frame, JOINT_INDICES["right_ankle"]),
            "right_ball": (right_ball_frame, JOINT_INDICES["right_ball"]),
            "right_toe": (right_toe_frame, JOINT_INDICES["right_toe"]),
        }
        
        for joint_name, (frame_idx, joint_idx) in calibration_map.items():
            if frame_idx >= 0:
                joint_coords = frames[frame_idx].get("joint_coords")
                if joint_coords is not None:
                    pos = self._get_joint_position(joint_coords, joint_idx)
                    ground_heights[joint_name] = pos[1]  # Y coordinate
                    log.info(f"Calibrated {joint_name} ground Y = {pos[1]:.4f} from frame {frame_idx}")
        
        # Fill missing calibrations with available ones from same foot
        # Left foot: use first available
        left_available = [v for k, v in ground_heights.items() if k.startswith("left_") and v is not None]
        if left_available:
            left_default = left_available[0]
            for key in ["left_ankle", "left_ball", "left_toe"]:
                if ground_heights[key] is None:
                    ground_heights[key] = left_default
                    log.info(f"Using default for {key}: {left_default:.4f}")
        
        # Right foot: use first available
        right_available = [v for k, v in ground_heights.items() if k.startswith("right_") and v is not None]
        if right_available:
            right_default = right_available[0]
            for key in ["right_ankle", "right_ball", "right_toe"]:
                if ground_heights[key] is None:
                    ground_heights[key] = right_default
                    log.info(f"Using default for {key}: {right_default:.4f}")
        
        # Verify all calibrations are set
        for key, val in ground_heights.items():
            if val is None:
                raise ValueError(f"Could not calibrate {key}. Check joint_coords in calibration frames.")
        
        log.info(f"Thresholds: height={height_threshold}m, velocity={velocity_threshold}m/frame, min_frames={min_contact_frames}")
        
        # Convert images to numpy
        images_np = images.cpu().numpy()
        if images_np.max() <= 1.0:
            images_np = (images_np * 255).astype(np.uint8)
        else:
            images_np = images_np.astype(np.uint8)
        
        # Process all frames - first pass (raw detection)
        raw_contact_data = []  # List of (left_state, right_state) per frame
        prev_positions = {}  # joint_name -> previous position
        
        for i, frame_data in enumerate(frames):
            joint_coords = frame_data.get("joint_coords")
            
            if joint_coords is None:
                # No data - assume airborne
                dummy_state = JointContactState(False, 1.0, 0.0, 0.0, 1.0)
                left_state = FootContactState(dummy_state, dummy_state, dummy_state)
                right_state = FootContactState(dummy_state, dummy_state, dummy_state)
                raw_contact_data.append((left_state, right_state))
                continue
            
            is_first_frame = (i == 0)
            
            # Detect each joint
            joint_states = {}
            for joint_name, joint_idx in JOINT_INDICES.items():
                pos = self._get_joint_position(joint_coords, joint_idx)
                
                # Compute velocity
                if is_first_frame or joint_name not in prev_positions:
                    velocity = 0.0
                else:
                    velocity = self._compute_velocity(pos, prev_positions[joint_name])
                
                # Detect contact
                state = self._detect_joint_contact(
                    current_y=pos[1],
                    ground_y=ground_heights[joint_name],
                    velocity=velocity,
                    height_threshold=height_threshold,
                    velocity_threshold=velocity_threshold,
                    is_first_frame=is_first_frame,
                )
                
                joint_states[joint_name] = state
                prev_positions[joint_name] = pos
            
            # Build foot states
            left_state = FootContactState(
                ankle=joint_states["left_ankle"],
                ball=joint_states["left_ball"],
                toe=joint_states["left_toe"],
            )
            right_state = FootContactState(
                ankle=joint_states["right_ankle"],
                ball=joint_states["right_ball"],
                toe=joint_states["right_toe"],
            )
            
            raw_contact_data.append((left_state, right_state))
        
        # Apply min_contact_frames filter if needed
        if min_contact_frames > 1:
            # Extract per-joint contact history
            joint_histories = {name: [] for name in JOINT_INDICES.keys()}
            
            for left_state, right_state in raw_contact_data:
                joint_histories["left_ankle"].append(left_state.ankle.grounded)
                joint_histories["left_ball"].append(left_state.ball.grounded)
                joint_histories["left_toe"].append(left_state.toe.grounded)
                joint_histories["right_ankle"].append(right_state.ankle.grounded)
                joint_histories["right_ball"].append(right_state.ball.grounded)
                joint_histories["right_toe"].append(right_state.toe.grounded)
            
            # Apply filter
            filtered_histories = {
                name: self._apply_min_contact_frames(history, min_contact_frames)
                for name, history in joint_histories.items()
            }
            
            # Update contact data with filtered results
            for i, (left_state, right_state) in enumerate(raw_contact_data):
                left_state.ankle.grounded = filtered_histories["left_ankle"][i]
                left_state.ball.grounded = filtered_histories["left_ball"][i]
                left_state.toe.grounded = filtered_histories["left_toe"][i]
                right_state.ankle.grounded = filtered_histories["right_ankle"][i]
                right_state.ball.grounded = filtered_histories["right_ball"][i]
                right_state.toe.grounded = filtered_histories["right_toe"][i]
        
        # Generate output frames and statistics
        output_frames = []
        contact_states = []
        
        for i, (left_state, right_state) in enumerate(raw_contact_data):
            # Determine overall state
            left_grounded = left_state.any_grounded
            right_grounded = right_state.any_grounded
            
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
            f"Detected contacts: both={both_count}, left={left_count}, "
            f"right={right_count}, airborne={none_count} "
            f"(total {num_frames} frames)"
        )
        log.info(status)
        
        # Add contact data to mesh_sequence for downstream nodes
        result_sequence = mesh_sequence.copy()
        result_sequence["foot_contact"] = {
            "contact_states": contact_states,
            "calibration": ground_heights,
            "thresholds": {
                "height": height_threshold,
                "velocity": velocity_threshold,
                "min_frames": min_contact_frames,
            },
            "joint_indices": JOINT_INDICES,
            "per_frame": [
                {
                    "left": {
                        "grounded": left_state.any_grounded,
                        "contact_type": left_state.contact_type,
                        "ankle": {"grounded": left_state.ankle.grounded, "distance": left_state.ankle.distance, "velocity": left_state.ankle.velocity},
                        "ball": {"grounded": left_state.ball.grounded, "distance": left_state.ball.distance, "velocity": left_state.ball.velocity},
                        "toe": {"grounded": left_state.toe.grounded, "distance": left_state.toe.distance, "velocity": left_state.toe.velocity},
                    },
                    "right": {
                        "grounded": right_state.any_grounded,
                        "contact_type": right_state.contact_type,
                        "ankle": {"grounded": right_state.ankle.grounded, "distance": right_state.ankle.distance, "velocity": right_state.ankle.velocity},
                        "ball": {"grounded": right_state.ball.grounded, "distance": right_state.ball.distance, "velocity": right_state.ball.velocity},
                        "toe": {"grounded": right_state.toe.grounded, "distance": right_state.toe.distance, "velocity": right_state.toe.velocity},
                    },
                }
                for left_state, right_state in raw_contact_data
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
