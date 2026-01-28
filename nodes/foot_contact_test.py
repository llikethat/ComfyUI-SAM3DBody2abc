"""
Foot Contact Detection Test Node
================================

Standalone node to test foot contact detection logic with user calibration.
Does NOT modify mesh_sequence - only outputs debug visualization.

Usage:
1. Connect mesh_sequence and images
2. Provide at least 2 calibration frames (any combination of both/left/right)
3. View debug overlay showing ground plane and contact states
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Optional, Tuple

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


class FootContactTest:
    """
    Test node for foot contact detection with user calibration.
    
    Requires minimum 2 calibration frames out of 3:
    - both_feet_frame: Frame where both feet are on ground
    - left_foot_frame: Frame where only left foot is on ground  
    - right_foot_frame: Frame where only right foot is on ground
    
    Detection uses:
    - Height check: foot Y <= calibrated_ground_Y + threshold
    - Velocity check: foot velocity <= velocity_threshold (skipped for frame 0)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE",),
                "images": ("IMAGE",),
            },
            "optional": {
                "both_feet_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "tooltip": "Frame where BOTH feet are on ground (-1 = not provided)"
                }),
                "left_foot_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "tooltip": "Frame where only LEFT foot is on ground (-1 = not provided)"
                }),
                "right_foot_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "tooltip": "Frame where only RIGHT foot is on ground (-1 = not provided)"
                }),
                "height_threshold": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.01,
                    "max": 0.3,
                    "step": 0.01,
                    "tooltip": "Distance from calibrated ground to count as contact (meters)"
                }),
                "velocity_threshold": ("FLOAT", {
                    "default": 0.02,
                    "min": 0.005,
                    "max": 0.1,
                    "step": 0.005,
                    "tooltip": "Max foot velocity to count as grounded (m/frame)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("debug_overlay", "status")
    FUNCTION = "test_detection"
    CATEGORY = "SAM3DBody2abc/Testing"
    
    def _get_foot_positions(self, vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract left and right foot positions from mesh vertices.
        
        Uses X-axis to split left/right:
        - X <= 0: Left side of body
        - X > 0: Right side of body
        
        Returns lowest Y point on each side as foot position.
        """
        x_values = vertices[:, 0]
        y_values = vertices[:, 1]
        z_values = vertices[:, 2]
        
        # Split by X position
        left_mask = x_values <= 0
        right_mask = x_values > 0
        
        # Find lowest point on each side
        if left_mask.sum() > 0:
            left_indices = np.where(left_mask)[0]
            left_lowest_idx = left_indices[np.argmin(y_values[left_mask])]
            left_foot_pos = vertices[left_lowest_idx]
        else:
            left_foot_pos = np.array([0, 0, 0])
        
        if right_mask.sum() > 0:
            right_indices = np.where(right_mask)[0]
            right_lowest_idx = right_indices[np.argmin(y_values[right_mask])]
            right_foot_pos = vertices[right_lowest_idx]
        else:
            right_foot_pos = np.array([0, 0, 0])
        
        return left_foot_pos, right_foot_pos
    
    def _compute_velocity(
        self, 
        curr_pos: np.ndarray, 
        prev_pos: np.ndarray
    ) -> float:
        """Compute velocity magnitude between two positions."""
        return np.linalg.norm(curr_pos - prev_pos)
    
    def _detect_contact(
        self,
        left_foot_y: float,
        right_foot_y: float,
        left_velocity: float,
        right_velocity: float,
        left_ground_y: float,
        right_ground_y: float,
        height_threshold: float,
        velocity_threshold: float,
        is_first_frame: bool = False,
    ) -> Tuple[str, bool, bool, float, float]:
        """
        Detect foot contact state.
        
        Returns:
            contact_state: "both", "left", "right", or "none"
            left_grounded: bool
            right_grounded: bool
            left_dist: distance from ground
            right_dist: distance from ground
        """
        # Distance from calibrated ground
        left_dist = left_foot_y - left_ground_y
        right_dist = right_foot_y - right_ground_y
        
        # Height check
        left_height_ok = left_dist <= height_threshold
        right_height_ok = right_dist <= height_threshold
        
        # Velocity check (skip for first frame)
        if is_first_frame:
            left_velocity_ok = True
            right_velocity_ok = True
        else:
            left_velocity_ok = left_velocity <= velocity_threshold
            right_velocity_ok = right_velocity <= velocity_threshold
        
        # Both conditions must be met
        left_grounded = left_height_ok and left_velocity_ok
        right_grounded = right_height_ok and right_velocity_ok
        
        # Determine state
        if left_grounded and right_grounded:
            contact_state = "both"
        elif left_grounded:
            contact_state = "left"
        elif right_grounded:
            contact_state = "right"
        else:
            contact_state = "none"
        
        return contact_state, left_grounded, right_grounded, left_dist, right_dist
    
    def _draw_debug_overlay(
        self,
        image: np.ndarray,
        frame_idx: int,
        contact_state: str,
        left_grounded: bool,
        right_grounded: bool,
        left_dist: float,
        right_dist: float,
        left_foot_pos: np.ndarray,
        right_foot_pos: np.ndarray,
        left_ground_y: float,
        right_ground_y: float,
        left_velocity: float,
        right_velocity: float,
        height_threshold: float,
        velocity_threshold: float,
    ) -> np.ndarray:
        """Draw debug visualization on frame."""
        
        # Make copy and convert to uint8 if needed
        if image.dtype == np.float32 or image.dtype == np.float64:
            frame = (image * 255).astype(np.uint8).copy()
        else:
            frame = image.copy()
        
        # Ensure BGR format for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Assume RGB, convert to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        h, w = frame.shape[:2]
        
        # Colors
        COLOR_GROUNDED = (0, 255, 0)      # Green
        COLOR_AIRBORNE = (0, 0, 255)      # Red
        COLOR_GROUND_LINE = (255, 255, 0) # Cyan
        COLOR_TEXT = (255, 255, 255)      # White
        COLOR_TEXT_BG = (0, 0, 0)         # Black
        
        # Contact state colors
        state_colors = {
            "both": (0, 255, 0),    # Green
            "left": (0, 255, 255),  # Yellow
            "right": (0, 255, 255), # Yellow
            "none": (0, 0, 255),    # Red
        }
        
        # Draw ground plane line (at bottom 20% of image as reference)
        ground_line_y = int(h * 0.85)
        cv2.line(frame, (0, ground_line_y), (w, ground_line_y), COLOR_GROUND_LINE, 2)
        cv2.putText(frame, "Ground Plane", (10, ground_line_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GROUND_LINE, 1)
        
        # Draw info panel background
        panel_height = 180
        cv2.rectangle(frame, (5, 5), (320, panel_height), COLOR_TEXT_BG, -1)
        cv2.rectangle(frame, (5, 5), (320, panel_height), COLOR_TEXT, 1)
        
        # Draw text info
        y_offset = 25
        line_height = 22
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        
        # Frame number
        cv2.putText(frame, f"Frame: {frame_idx}", (15, y_offset),
                   font, font_scale, COLOR_TEXT, 1)
        y_offset += line_height
        
        # Contact state
        state_color = state_colors.get(contact_state, COLOR_TEXT)
        cv2.putText(frame, f"Contact: {contact_state.upper()}", (15, y_offset),
                   font, font_scale, state_color, 2)
        y_offset += line_height
        
        # Left foot info
        left_color = COLOR_GROUNDED if left_grounded else COLOR_AIRBORNE
        left_status = "GROUND" if left_grounded else "AIR"
        cv2.putText(frame, f"Left:  {left_status}", (15, y_offset),
                   font, font_scale, left_color, 1)
        cv2.putText(frame, f"dist={left_dist:.3f}m vel={left_velocity:.3f}", (130, y_offset),
                   font, 0.4, COLOR_TEXT, 1)
        y_offset += line_height
        
        # Right foot info
        right_color = COLOR_GROUNDED if right_grounded else COLOR_AIRBORNE
        right_status = "GROUND" if right_grounded else "AIR"
        cv2.putText(frame, f"Right: {right_status}", (15, y_offset),
                   font, font_scale, right_color, 1)
        cv2.putText(frame, f"dist={right_dist:.3f}m vel={right_velocity:.3f}", (130, y_offset),
                   font, 0.4, COLOR_TEXT, 1)
        y_offset += line_height
        
        # Thresholds
        cv2.putText(frame, f"Height thresh: {height_threshold:.3f}m", (15, y_offset),
                   font, 0.45, (180, 180, 180), 1)
        y_offset += line_height - 4
        cv2.putText(frame, f"Velocity thresh: {velocity_threshold:.3f}m/f", (15, y_offset),
                   font, 0.45, (180, 180, 180), 1)
        y_offset += line_height - 4
        
        # Calibrated ground heights
        cv2.putText(frame, f"Ground Y - L:{left_ground_y:.3f} R:{right_ground_y:.3f}", (15, y_offset),
                   font, 0.45, COLOR_GROUND_LINE, 1)
        
        # Convert back to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame
    
    def test_detection(
        self,
        mesh_sequence: Dict,
        images: torch.Tensor,
        both_feet_frame: int = -1,
        left_foot_frame: int = -1,
        right_foot_frame: int = -1,
        height_threshold: float = 0.05,
        velocity_threshold: float = 0.02,
    ) -> Tuple[torch.Tensor, str]:
        """Test foot contact detection and generate debug overlay."""
        
        # Validate: minimum 2 calibration frames required
        provided_count = sum([
            both_feet_frame >= 0,
            left_foot_frame >= 0,
            right_foot_frame >= 0,
        ])
        
        if provided_count < 2:
            raise ValueError(
                "Please provide at least 2 calibration frames. "
                f"Got: both={both_feet_frame}, left={left_foot_frame}, right={right_foot_frame}"
            )
        
        # Get frames data
        frames_data = mesh_sequence.get("frames", {})
        if isinstance(frames_data, dict):
            frame_keys = sorted(frames_data.keys())
            frames = [frames_data[k] for k in frame_keys]
        else:
            frames = frames_data
        
        num_frames = len(frames)
        log.info(f"Processing {num_frames} frames")
        log.info(f"Calibration frames - both:{both_feet_frame}, left:{left_foot_frame}, right:{right_foot_frame}")
        
        # Validate frame indices
        max_frame = num_frames - 1
        if both_feet_frame > max_frame or left_foot_frame > max_frame or right_foot_frame > max_frame:
            raise ValueError(
                f"Calibration frame index out of range. Max frame: {max_frame}. "
                f"Got: both={both_feet_frame}, left={left_foot_frame}, right={right_foot_frame}"
            )
        
        # Extract calibration data
        left_ground_y = None
        right_ground_y = None
        
        # Process both_feet_frame
        if both_feet_frame >= 0:
            vertices = frames[both_feet_frame].get("vertices")
            if vertices is not None:
                if vertices.ndim == 3:
                    vertices = vertices[0]
                left_pos, right_pos = self._get_foot_positions(vertices)
                left_ground_y = left_pos[1]
                right_ground_y = right_pos[1]
                log.info(f"Both feet frame {both_feet_frame}: L_ground={left_ground_y:.4f}, R_ground={right_ground_y:.4f}")
        
        # Process left_foot_frame (overrides left from both if provided)
        if left_foot_frame >= 0:
            vertices = frames[left_foot_frame].get("vertices")
            if vertices is not None:
                if vertices.ndim == 3:
                    vertices = vertices[0]
                left_pos, _ = self._get_foot_positions(vertices)
                left_ground_y = left_pos[1]
                log.info(f"Left foot frame {left_foot_frame}: L_ground={left_ground_y:.4f}")
        
        # Process right_foot_frame (overrides right from both if provided)
        if right_foot_frame >= 0:
            vertices = frames[right_foot_frame].get("vertices")
            if vertices is not None:
                if vertices.ndim == 3:
                    vertices = vertices[0]
                _, right_pos = self._get_foot_positions(vertices)
                right_ground_y = right_pos[1]
                log.info(f"Right foot frame {right_foot_frame}: R_ground={right_ground_y:.4f}")
        
        # Ensure we have both ground values
        if left_ground_y is None or right_ground_y is None:
            raise ValueError(
                f"Could not extract ground heights. L={left_ground_y}, R={right_ground_y}. "
                "Check that calibration frames have valid vertex data."
            )
        
        log.info(f"Calibrated ground: Left Y={left_ground_y:.4f}, Right Y={right_ground_y:.4f}")
        log.info(f"Thresholds: height={height_threshold}m, velocity={velocity_threshold}m/frame")
        
        # Process all frames
        contact_states = []
        prev_left_pos = None
        prev_right_pos = None
        
        # Convert images to numpy
        images_np = images.cpu().numpy()
        if images_np.max() <= 1.0:
            images_np = (images_np * 255).astype(np.uint8)
        else:
            images_np = images_np.astype(np.uint8)
        
        output_frames = []
        
        for i, frame_data in enumerate(frames):
            vertices = frame_data.get("vertices")
            
            if vertices is None:
                contact_states.append("none")
                # Just copy original frame
                if i < len(images_np):
                    output_frames.append(images_np[i])
                continue
            
            if vertices.ndim == 3:
                vertices = vertices[0]
            
            # Get foot positions
            left_foot_pos, right_foot_pos = self._get_foot_positions(vertices)
            
            # Compute velocities
            is_first_frame = (prev_left_pos is None)
            if is_first_frame:
                left_velocity = 0.0
                right_velocity = 0.0
            else:
                left_velocity = self._compute_velocity(left_foot_pos, prev_left_pos)
                right_velocity = self._compute_velocity(right_foot_pos, prev_right_pos)
            
            # Detect contact
            contact_state, left_grounded, right_grounded, left_dist, right_dist = self._detect_contact(
                left_foot_y=left_foot_pos[1],
                right_foot_y=right_foot_pos[1],
                left_velocity=left_velocity,
                right_velocity=right_velocity,
                left_ground_y=left_ground_y,
                right_ground_y=right_ground_y,
                height_threshold=height_threshold,
                velocity_threshold=velocity_threshold,
                is_first_frame=is_first_frame,
            )
            
            contact_states.append(contact_state)
            
            # Draw debug overlay
            if i < len(images_np):
                overlay = self._draw_debug_overlay(
                    image=images_np[i],
                    frame_idx=i,
                    contact_state=contact_state,
                    left_grounded=left_grounded,
                    right_grounded=right_grounded,
                    left_dist=left_dist,
                    right_dist=right_dist,
                    left_foot_pos=left_foot_pos,
                    right_foot_pos=right_foot_pos,
                    left_ground_y=left_ground_y,
                    right_ground_y=right_ground_y,
                    left_velocity=left_velocity,
                    right_velocity=right_velocity,
                    height_threshold=height_threshold,
                    velocity_threshold=velocity_threshold,
                )
                output_frames.append(overlay)
            
            # Store for next iteration
            prev_left_pos = left_foot_pos.copy()
            prev_right_pos = right_foot_pos.copy()
        
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
        
        # Convert output to tensor
        output_array = np.stack(output_frames, axis=0)
        output_tensor = torch.from_numpy(output_array).float() / 255.0
        
        return (output_tensor, status)


NODE_CLASS_MAPPINGS = {
    "FootContactTest": FootContactTest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FootContactTest": "🦶 Foot Contact Test (Calibration)",
}
