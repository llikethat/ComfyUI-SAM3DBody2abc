"""
Ground Contact Enforcement for SAM3DBody2abc
=============================================

Prevents foot skating by detecting foot contacts and pinning feet to ground.

Key Features:
- Automatic foot contact detection (height + velocity based)
- Ground plane estimation
- Foot pinning via root translation adjustment
- Smooth transitions in/out of contacts
- Support for MHR 127-joint skeleton

The Problem:
    SAM3DBody estimates body pose per-frame independently. This causes
    feet to "slide" or "skate" during contact phases where they should
    be stationary.

The Solution:
    1. Detect when feet are in contact (low height + low velocity)
    2. During contact, lock foot position
    3. Adjust root translation to maintain foot position
    4. Blend transitions to avoid pops

Usage:
    from ground_contact import enforce_foot_contacts, FootContactEnforcer
    
    # Quick usage
    fixed_sequence = enforce_foot_contacts(mesh_sequence)
    
    # Advanced usage
    enforcer = FootContactEnforcer(config)
    fixed_sequence = enforcer.process(mesh_sequence)

Author: SAM3DBody2abc
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt


# =============================================================================
# MHR/SMPL Joint Indices
# =============================================================================

# Standard MHR 127-joint indices for feet
MHR_JOINT_INDICES = {
    # Body
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine3": 9,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "left_collar": 13,
    "right_collar": 14,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
    
    # Toe tips (approximate - MHR specific)
    "left_toe": 10,   # Same as left_foot in basic skeleton
    "right_toe": 11,  # Same as right_foot in basic skeleton
}

# SMPL 24-joint indices (if using SMPL instead of MHR)
SMPL_JOINT_INDICES = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine3": 9,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "left_collar": 13,
    "right_collar": 14,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
    "left_hand": 22,
    "right_hand": 23,
}


@dataclass
class FootContactConfig:
    """Configuration for foot contact detection and enforcement."""
    
    # Detection thresholds
    height_threshold: float = 0.05      # Meters - foot considered on ground if below this
    velocity_threshold: float = 0.02    # Meters/frame - foot considered stationary below this
    
    # Joint indices (can be customized for different skeletons)
    left_ankle_idx: int = 7
    right_ankle_idx: int = 8
    left_foot_idx: int = 10
    right_foot_idx: int = 11
    pelvis_idx: int = 0
    
    # Contact detection
    min_contact_frames: int = 3         # Minimum frames for valid contact
    contact_smoothing: int = 3          # Median filter size for contact detection
    
    # Ground plane
    auto_ground_plane: bool = True      # Automatically estimate ground height
    ground_height: float = 0.0          # Manual ground height (if not auto)
    
    # Blending
    blend_frames: int = 3               # Frames to blend in/out of contacts
    
    # Processing
    use_toe_for_contact: bool = True    # Use toe (foot) joint instead of ankle
    both_feet_for_ground: bool = True   # Use both feet to estimate ground


@dataclass 
class FootContact:
    """Represents a detected foot contact phase."""
    foot: str                           # "left" or "right"
    start_frame: int
    end_frame: int
    ground_position: np.ndarray         # [x, y, z] where foot should be pinned
    confidence: float = 1.0


class FootContactEnforcer:
    """
    Detects foot contacts and adjusts motion to prevent skating.
    
    Pipeline:
    1. Extract foot positions from joint data
    2. Detect contact phases (low height + low velocity)
    3. Estimate ground plane
    4. During contacts, pin feet by adjusting root translation
    5. Blend transitions smoothly
    """
    
    def __init__(self, config: Optional[FootContactConfig] = None):
        """
        Initialize foot contact enforcer.
        
        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or FootContactConfig()
    
    def process(
        self,
        mesh_sequence: Dict,
        verbose: bool = True
    ) -> Dict:
        """
        Process mesh sequence to enforce foot contacts.
        
        Args:
            mesh_sequence: MESH_SEQUENCE from SAM3DBody
            verbose: Print progress information
        
        Returns:
            Modified mesh_sequence with adjusted root translations
        """
        frames = mesh_sequence.get("frames", [])
        if not frames:
            return mesh_sequence
        
        num_frames = len(frames)
        
        # 1. Extract joint positions
        joint_positions = self._extract_joint_positions(frames)
        if joint_positions is None:
            if verbose:
                print("[Foot Contact] No joint data found")
            return mesh_sequence
        
        # 2. Extract camera translations (pred_cam_t)
        root_translations = self._extract_root_translations(frames)
        
        # 3. Compute world positions (joints + translation)
        world_positions = joint_positions + root_translations[:, np.newaxis, :]
        
        # 4. Detect foot contacts
        contacts = self._detect_contacts(world_positions, verbose)
        
        if verbose:
            left_contacts = [c for c in contacts if c.foot == "left"]
            right_contacts = [c for c in contacts if c.foot == "right"]
            print(f"[Foot Contact] Detected {len(left_contacts)} left, {len(right_contacts)} right contacts")
        
        # 5. Estimate ground plane
        ground_height = self._estimate_ground_plane(world_positions, contacts, verbose)
        
        # 6. Enforce contacts by adjusting root translation
        adjusted_translations = self._enforce_contacts(
            joint_positions,
            root_translations,
            contacts,
            ground_height,
            verbose
        )
        
        # 7. Update mesh sequence
        result = mesh_sequence.copy()
        result["frames"] = frames.copy()
        
        for i, frame in enumerate(result["frames"]):
            frame["pred_cam_t"] = adjusted_translations[i].tolist()
            frame["foot_contact_adjusted"] = True
        
        # Store contact info
        result["foot_contacts"] = {
            "contacts": [
                {
                    "foot": c.foot,
                    "start": c.start_frame,
                    "end": c.end_frame,
                    "position": c.ground_position.tolist(),
                }
                for c in contacts
            ],
            "ground_height": float(ground_height),
            "config": {
                "height_threshold": self.config.height_threshold,
                "velocity_threshold": self.config.velocity_threshold,
            }
        }
        
        return result
    
    def _extract_joint_positions(self, frames: List[Dict]) -> Optional[np.ndarray]:
        """Extract joint positions from frames. Returns [T, J, 3] array."""
        positions = []
        
        for frame in frames:
            # Try different key names
            joints = frame.get("joint_coords")
            if joints is None:
                joints = frame.get("joints")
            if joints is None:
                joints = frame.get("keypoints_3d")
            
            if joints is not None:
                positions.append(np.array(joints))
            else:
                # Use previous frame if missing
                if positions:
                    positions.append(positions[-1].copy())
                else:
                    return None
        
        return np.array(positions)
    
    def _extract_root_translations(self, frames: List[Dict]) -> np.ndarray:
        """Extract root translations (pred_cam_t) from frames."""
        translations = []
        
        for frame in frames:
            cam_t = frame.get("pred_cam_t")
            if cam_t is None:
                cam_t = frame.get("camera")
            if cam_t is None:
                cam_t = [0, 0, 2]  # Default
            
            translations.append(np.array(cam_t).flatten()[:3])
        
        return np.array(translations)
    
    def _detect_contacts(
        self,
        world_positions: np.ndarray,
        verbose: bool
    ) -> List[FootContact]:
        """
        Detect foot contact phases from world positions.
        
        Uses height threshold + velocity threshold to determine contacts.
        """
        cfg = self.config
        T = world_positions.shape[0]
        
        # Get foot joint indices
        left_idx = cfg.left_foot_idx if cfg.use_toe_for_contact else cfg.left_ankle_idx
        right_idx = cfg.right_foot_idx if cfg.use_toe_for_contact else cfg.right_ankle_idx
        
        # Extract foot positions
        left_foot = world_positions[:, left_idx, :]
        right_foot = world_positions[:, right_idx, :]
        
        # Compute velocities
        left_velocity = np.zeros(T)
        right_velocity = np.zeros(T)
        left_velocity[1:] = np.linalg.norm(np.diff(left_foot, axis=0), axis=1)
        right_velocity[1:] = np.linalg.norm(np.diff(right_foot, axis=0), axis=1)
        
        # Get heights (Y coordinate in Y-up system)
        # First estimate rough ground level
        all_foot_heights = np.concatenate([left_foot[:, 1], right_foot[:, 1]])
        rough_ground = np.percentile(all_foot_heights, 10)
        
        left_height = left_foot[:, 1] - rough_ground
        right_height = right_foot[:, 1] - rough_ground
        
        # Detect contacts: low height AND low velocity
        left_contact_raw = (left_height < cfg.height_threshold) & (left_velocity < cfg.velocity_threshold)
        right_contact_raw = (right_height < cfg.height_threshold) & (right_velocity < cfg.velocity_threshold)
        
        # Smooth contact detection (remove spurious contacts)
        if cfg.contact_smoothing > 1:
            left_contact_raw = medfilt(left_contact_raw.astype(float), cfg.contact_smoothing) > 0.5
            right_contact_raw = medfilt(right_contact_raw.astype(float), cfg.contact_smoothing) > 0.5
        
        # Convert to contact phases
        contacts = []
        
        for foot, contact_mask, foot_pos in [
            ("left", left_contact_raw, left_foot),
            ("right", right_contact_raw, right_foot)
        ]:
            # Find contiguous contact regions
            in_contact = False
            start_frame = 0
            
            for t in range(T):
                if contact_mask[t] and not in_contact:
                    # Start of contact
                    in_contact = True
                    start_frame = t
                elif not contact_mask[t] and in_contact:
                    # End of contact
                    in_contact = False
                    if t - start_frame >= cfg.min_contact_frames:
                        # Valid contact phase
                        # Pin position is average foot position during contact
                        contact_pos = np.mean(foot_pos[start_frame:t], axis=0)
                        contact_pos[1] = rough_ground  # Snap to ground
                        
                        contacts.append(FootContact(
                            foot=foot,
                            start_frame=start_frame,
                            end_frame=t - 1,
                            ground_position=contact_pos,
                        ))
            
            # Handle contact at end of sequence
            if in_contact and T - start_frame >= cfg.min_contact_frames:
                contact_pos = np.mean(foot_pos[start_frame:], axis=0)
                contact_pos[1] = rough_ground
                contacts.append(FootContact(
                    foot=foot,
                    start_frame=start_frame,
                    end_frame=T - 1,
                    ground_position=contact_pos,
                ))
        
        return contacts
    
    def _estimate_ground_plane(
        self,
        world_positions: np.ndarray,
        contacts: List[FootContact],
        verbose: bool
    ) -> float:
        """
        Estimate ground plane height from contact positions.
        """
        if not contacts:
            # Fallback: use minimum foot height
            cfg = self.config
            left_idx = cfg.left_foot_idx
            right_idx = cfg.right_foot_idx
            
            foot_heights = np.concatenate([
                world_positions[:, left_idx, 1],
                world_positions[:, right_idx, 1]
            ])
            ground_height = np.percentile(foot_heights, 5)
            
            if verbose:
                print(f"[Foot Contact] No contacts, estimated ground at y={ground_height:.3f}")
            return ground_height
        
        # Use contact positions to estimate ground
        contact_heights = [c.ground_position[1] for c in contacts]
        ground_height = np.median(contact_heights)
        
        if verbose:
            print(f"[Foot Contact] Ground plane at y={ground_height:.3f}")
        
        return ground_height
    
    def _enforce_contacts(
        self,
        joint_positions: np.ndarray,
        root_translations: np.ndarray,
        contacts: List[FootContact],
        ground_height: float,
        verbose: bool
    ) -> np.ndarray:
        """
        Adjust root translations to enforce foot contacts.
        
        During contact phases, adjust the root so the contact foot
        remains at its pinned position.
        """
        cfg = self.config
        T = joint_positions.shape[0]
        
        # Start with original translations
        adjusted = root_translations.copy()
        
        # Get foot joint indices
        foot_indices = {
            "left": cfg.left_foot_idx if cfg.use_toe_for_contact else cfg.left_ankle_idx,
            "right": cfg.right_foot_idx if cfg.use_toe_for_contact else cfg.right_ankle_idx,
        }
        
        # Track adjustments per frame
        adjustments = np.zeros((T, 3))
        adjustment_weights = np.zeros(T)
        
        for contact in contacts:
            foot_idx = foot_indices[contact.foot]
            
            for t in range(contact.start_frame, contact.end_frame + 1):
                # Current foot position (joint + translation)
                current_foot = joint_positions[t, foot_idx] + adjusted[t]
                
                # Desired foot position (pinned)
                desired_foot = contact.ground_position.copy()
                desired_foot[1] = ground_height  # Ensure on ground
                
                # Adjustment needed
                adjustment = desired_foot - current_foot
                
                # Compute blend weight (fade in/out at contact edges)
                blend = 1.0
                if t < contact.start_frame + cfg.blend_frames:
                    blend = (t - contact.start_frame + 1) / cfg.blend_frames
                elif t > contact.end_frame - cfg.blend_frames:
                    blend = (contact.end_frame - t + 1) / cfg.blend_frames
                
                # Accumulate weighted adjustment
                adjustments[t] += adjustment * blend
                adjustment_weights[t] += blend
        
        # Apply averaged adjustments
        for t in range(T):
            if adjustment_weights[t] > 0:
                adjusted[t] += adjustments[t] / adjustment_weights[t]
        
        # Smooth the adjusted translations to avoid pops
        adjusted = gaussian_filter1d(adjusted, sigma=1.0, axis=0)
        
        if verbose:
            original_skating = self._compute_skating(joint_positions, root_translations, foot_indices)
            adjusted_skating = self._compute_skating(joint_positions, adjusted, foot_indices)
            reduction = (1 - adjusted_skating / max(original_skating, 1e-8)) * 100
            print(f"[Foot Contact] Skating reduced by {reduction:.1f}%")
        
        return adjusted
    
    def _compute_skating(
        self,
        joint_positions: np.ndarray,
        translations: np.ndarray,
        foot_indices: Dict[str, int]
    ) -> float:
        """Compute total foot skating distance."""
        T = joint_positions.shape[0]
        total_skating = 0.0
        
        for foot, idx in foot_indices.items():
            foot_world = joint_positions[:, idx, :] + translations
            
            # Skating = movement when foot should be on ground
            foot_height = foot_world[:, 1]
            on_ground = foot_height < np.percentile(foot_height, 20)
            
            for t in range(1, T):
                if on_ground[t] and on_ground[t-1]:
                    skating = np.linalg.norm(foot_world[t] - foot_world[t-1])
                    total_skating += skating
        
        return total_skating


def enforce_foot_contacts(
    mesh_sequence: Dict,
    height_threshold: float = 0.05,
    velocity_threshold: float = 0.02,
    verbose: bool = True
) -> Dict:
    """
    Convenience function to enforce foot contacts on a mesh sequence.
    
    Args:
        mesh_sequence: MESH_SEQUENCE from SAM3DBody
        height_threshold: Height below which foot is considered on ground (meters)
        velocity_threshold: Velocity below which foot is considered stationary (m/frame)
        verbose: Print progress
    
    Returns:
        Modified mesh_sequence with foot skating reduced
    """
    config = FootContactConfig(
        height_threshold=height_threshold,
        velocity_threshold=velocity_threshold,
    )
    enforcer = FootContactEnforcer(config)
    return enforcer.process(mesh_sequence, verbose=verbose)


# =============================================================================
# ComfyUI Node
# =============================================================================

class FootContactNode:
    """
    ComfyUI node for enforcing foot ground contacts.
    
    Reduces foot skating by detecting contacts and pinning feet.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE",),
            },
            "optional": {
                "height_threshold": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.01,
                    "max": 0.2,
                    "step": 0.01,
                    "tooltip": "Height threshold for ground contact (meters)."
                }),
                "velocity_threshold": ("FLOAT", {
                    "default": 0.02,
                    "min": 0.005,
                    "max": 0.1,
                    "step": 0.005,
                    "tooltip": "Velocity threshold for stationary foot (m/frame)."
                }),
                "min_contact_frames": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Minimum frames for valid contact detection."
                }),
                "blend_frames": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Frames to blend in/out of contacts."
                }),
                "use_toe_joint": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use toe joint instead of ankle for contact detection."
                }),
                "log_level": (["Normal", "Verbose", "Silent"], {
                    "default": "Normal",
                }),
            }
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE", "STRING")
    RETURN_NAMES = ("mesh_sequence", "status")
    FUNCTION = "enforce"
    CATEGORY = "SAM3DBody2abc/Processing"
    
    def enforce(
        self,
        mesh_sequence: Dict,
        height_threshold: float = 0.05,
        velocity_threshold: float = 0.02,
        min_contact_frames: int = 3,
        blend_frames: int = 3,
        use_toe_joint: bool = True,
        log_level: str = "Normal",
    ) -> Tuple[Dict, str]:
        """Enforce foot contacts on mesh sequence."""
        verbose = log_level != "Silent"
        
        config = FootContactConfig(
            height_threshold=height_threshold,
            velocity_threshold=velocity_threshold,
            min_contact_frames=min_contact_frames,
            blend_frames=blend_frames,
            use_toe_for_contact=use_toe_joint,
        )
        
        enforcer = FootContactEnforcer(config)
        result = enforcer.process(mesh_sequence, verbose=verbose)
        
        # Build status
        contacts = result.get("foot_contacts", {}).get("contacts", [])
        left_count = len([c for c in contacts if c["foot"] == "left"])
        right_count = len([c for c in contacts if c["foot"] == "right"])
        
        status = f"Enforced {left_count} left, {right_count} right foot contacts"
        
        return (result, status)


NODE_CLASS_MAPPINGS = {
    "FootContactEnforcer": FootContactNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FootContactEnforcer": "🦶 Foot Contact Enforcer",
}
