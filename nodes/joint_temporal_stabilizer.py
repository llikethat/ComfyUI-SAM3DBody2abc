"""
Joint Temporal Stabilizer for SAM3DBody2abc
============================================

Reduce jitter in 3D joints with separate thresholds for FEET vs BODY.

Key Feature: 
- FEET: Strong stabilization (prevents sliding)
- BODY: Light stabilization (preserves natural movement)

Placement in workflow:
    SAM3DBody2abc_VideoBatchProcessor → JointTemporalStabilizer → KinematicContact → ExportAnimatedFBX
    
    OR (replace KinematicContact entirely):
    SAM3DBody2abc_VideoBatchProcessor → JointTemporalStabilizer → ExportAnimatedFBX

Author: Claude (Anthropic)
Version: 1.1.0
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Optional, Any
from copy import deepcopy

# Try to import scipy for Savitzky-Golay filter
try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# Skeleton Configurations (MHR 127 joints / SMPL 24 joints)
# =============================================================================

SKELETON_CONFIGS = {
    "MHR_127": {
        "num_joints": 127,
        "feet_indices": [15, 16, 17, 18, 19, 20],
        "leg_indices": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "pelvis_idx": 0,
    },
    "MHR_70": {
        "num_joints": 70,
        "feet_indices": [15, 16, 17, 18, 19, 20],
        "leg_indices": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "pelvis_idx": 0,
    },
    "SMPL_24": {
        "num_joints": 24,
        "feet_indices": [7, 8, 10, 11],
        "leg_indices": [1, 2, 4, 5, 7, 8, 10, 11],
        "pelvis_idx": 0,
    },
    "HALPE_26": {
        "num_joints": 26,
        "feet_indices": [15, 16, 20, 21, 22, 23, 24, 25],
        "leg_indices": [11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25],
        "pelvis_idx": 19,
    },
    "AUTO": {
        "num_joints": -1,
        "feet_indices": [],
        "leg_indices": [],
        "pelvis_idx": 0,
    },
}


# =============================================================================
# Visualization Colors (BGR for OpenCV)
# =============================================================================

COLORS = {
    "original_joint": (0, 0, 255),      # Red - original position
    "stabilized_joint": (0, 255, 0),    # Green - stabilized position
    "feet_original": (0, 100, 255),     # Orange - original feet
    "feet_stabilized": (0, 255, 100),   # Light green - stabilized feet
    "ground_contact": (255, 255, 0),    # Cyan - grounded
    "trajectory_original": (0, 0, 200), # Dark red
    "trajectory_stabilized": (0, 200, 0), # Dark green
    "connection": (200, 200, 200),      # Gray - bone connections
    "text_bg": (0, 0, 0),               # Black
    "text_fg": (255, 255, 255),         # White
}


class JointTemporalStabilizer:
    """
    Temporal stabilization for 3D joints in MESH_SEQUENCE.
    
    Features:
    - Separate thresholds for FEET (strong pinning) vs BODY (natural movement)
    - Ground contact detection for automatic foot pinning
    - Debug overlay showing before/after comparison
    - Works with MHR (127), SMPL (24), or custom skeletons
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Video frames for overlay visualization"
                }),
                "mesh_sequence": ("MESH_SEQUENCE", {
                    "tooltip": "Mesh sequence from Video Batch Processor"
                }),
            },
            "optional": {
                # Skeleton format
                "skeleton_format": (list(SKELETON_CONFIGS.keys()), {
                    "default": "AUTO",
                    "tooltip": "Skeleton format for joint indices"
                }),
                "custom_feet_indices": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated joint indices for feet (e.g., '15,16,17,18')"
                }),
                
                # === FEET STABILIZATION (Strong) ===
                "feet_ema_alpha": ("FLOAT", {
                    "default": 0.15, "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": "EMA for feet. LOWER = stronger smoothing (more stable)"
                }),
                "feet_hysteresis_px": ("FLOAT", {
                    "default": 3.0, "min": 0.5, "max": 20.0, "step": 0.5,
                    "tooltip": "Movement threshold for feet. HIGHER = more pinned"
                }),
                "feet_velocity_damping": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 0.95, "step": 0.05,
                    "tooltip": "Velocity damping for feet. HIGHER = slower movement"
                }),
                
                # === BODY STABILIZATION (Light) ===
                "body_ema_alpha": ("FLOAT", {
                    "default": 0.4, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "EMA for body. HIGHER = more responsive"
                }),
                "body_hysteresis_px": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Movement threshold for body. LOWER = more natural"
                }),
                "body_velocity_damping": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 0.9, "step": 0.05,
                    "tooltip": "Velocity damping for body. LOWER = natural movement"
                }),
                
                # === GROUND CONTACT ===
                "enable_ground_contact": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-detect grounded feet for extra pinning"
                }),
                "ground_velocity_threshold": ("FLOAT", {
                    "default": 0.02, "min": 0.005, "max": 0.1, "step": 0.005,
                    "tooltip": "Max velocity to consider foot grounded"
                }),
                "ground_contact_ema": ("FLOAT", {
                    "default": 0.05, "min": 0.01, "max": 0.2, "step": 0.01,
                    "tooltip": "EMA when grounded (very strong smoothing)"
                }),
                
                # === SMOOTHING ===
                "enable_savgol": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply Savitzky-Golay filter (requires scipy)"
                }),
                "savgol_window": ("INT", {
                    "default": 5, "min": 3, "max": 15, "step": 2,
                    "tooltip": "Savitzky-Golay window (must be odd)"
                }),
                "savgol_order": ("INT", {
                    "default": 2, "min": 1, "max": 4,
                    "tooltip": "Savitzky-Golay polynomial order"
                }),
                
                # === VISUALIZATION ===
                "show_original": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show original joints (red)"
                }),
                "show_stabilized": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show stabilized joints (green)"
                }),
                "show_trajectories": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show joint trajectories over time"
                }),
                "trajectory_length": ("INT", {
                    "default": 15, "min": 5, "max": 60,
                    "tooltip": "Number of frames for trajectory trail"
                }),
                "show_ground_contact": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Highlight grounded feet"
                }),
                "joint_radius": ("INT", {
                    "default": 4, "min": 2, "max": 10,
                    "tooltip": "Radius of joint circles"
                }),
                
                # === DEBUG ===
                "log_level": (["silent", "normal", "verbose"], {
                    "default": "normal"
                }),
            },
        }

    RETURN_TYPES = ("MESH_SEQUENCE", "IMAGE", "STRING")
    RETURN_NAMES = ("stabilized_sequence", "debug_overlay", "debug_info")
    FUNCTION = "process"
    CATEGORY = "SAM3DBody2abc/Processing"

    def process(
        self,
        images,
        mesh_sequence: List[Dict],
        skeleton_format: str = "AUTO",
        custom_feet_indices: str = "",
        # Feet
        feet_ema_alpha: float = 0.15,
        feet_hysteresis_px: float = 3.0,
        feet_velocity_damping: float = 0.8,
        # Body
        body_ema_alpha: float = 0.4,
        body_hysteresis_px: float = 1.0,
        body_velocity_damping: float = 0.3,
        # Ground contact
        enable_ground_contact: bool = True,
        ground_velocity_threshold: float = 0.02,
        ground_contact_ema: float = 0.05,
        # Smoothing
        enable_savgol: bool = False,
        savgol_window: int = 5,
        savgol_order: int = 2,
        # Visualization
        show_original: bool = True,
        show_stabilized: bool = True,
        show_trajectories: bool = True,
        trajectory_length: int = 15,
        show_ground_contact: bool = True,
        joint_radius: int = 4,
        # Debug
        log_level: str = "normal",
    ):
        """Apply temporal stabilization with separate feet/body thresholds."""
        
        verbose = log_level == "verbose"
        silent = log_level == "silent"
        
        if not silent:
            print(f"[JointStabilizer] Processing {len(mesh_sequence)} frames")
        
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
        
        # Deep copy to avoid modifying original
        output_sequence = deepcopy(mesh_sequence)
        
        # Extract all joints and camera data
        joints_list = []
        cameras_list = []
        focal_lengths = []
        image_sizes = []
        valid_indices = []
        
        for i, frame in enumerate(output_sequence):
            if frame.get("valid", False) and frame.get("joints") is not None:
                joints = frame["joints"]
                if isinstance(joints, torch.Tensor):
                    joints = joints.cpu().numpy()
                joints_list.append(joints)
                
                # Get camera and focal length for projection
                camera = frame.get("camera")
                if camera is not None and isinstance(camera, torch.Tensor):
                    camera = camera.cpu().numpy()
                cameras_list.append(camera)
                focal_lengths.append(frame.get("focal_length"))
                image_sizes.append(frame.get("image_size", (images_np.shape[2], images_np.shape[1])))
                valid_indices.append(i)
        
        if len(joints_list) == 0:
            if not silent:
                print(f"[JointStabilizer] No valid joints found")
            empty_overlay = torch.from_numpy(images_np.astype(np.float32) / 255.0)
            return (output_sequence, empty_overlay, "No valid joints found")
        
        # Stack into (N, J, 3) array
        joints_array = np.stack(joints_list, axis=0)
        original_joints = joints_array.copy()  # Keep original for visualization
        N, J, D = joints_array.shape
        
        if not silent:
            print(f"[JointStabilizer] Processing {N} valid frames, {J} joints")
        
        # Get feet indices
        feet_indices, body_indices = self._get_joint_groups(
            skeleton_format, custom_feet_indices, J, verbose
        )
        
        if not silent:
            print(f"[JointStabilizer] Feet joints: {feet_indices}")
            print(f"[JointStabilizer] Body joints: {len(body_indices)} joints")
        
        # Stabilize joints
        stabilized, ground_contact = self._stabilize_joints(
            joints_array,
            feet_indices=feet_indices,
            body_indices=body_indices,
            feet_ema_alpha=feet_ema_alpha,
            feet_hysteresis=feet_hysteresis_px,
            feet_damping=feet_velocity_damping,
            body_ema_alpha=body_ema_alpha,
            body_hysteresis=body_hysteresis_px,
            body_damping=body_velocity_damping,
            enable_ground_contact=enable_ground_contact,
            ground_velocity_threshold=ground_velocity_threshold,
            ground_contact_ema=ground_contact_ema,
            enable_savgol=enable_savgol,
            savgol_window=savgol_window,
            savgol_order=savgol_order,
            verbose=verbose,
        )
        
        # Put stabilized joints back into sequence
        for i, frame_idx in enumerate(valid_indices):
            output_sequence[frame_idx]["joints"] = stabilized[i]
        
        # Generate debug overlay
        overlay = self._render_debug_overlay(
            images_np=images_np,
            original_joints=original_joints,
            stabilized_joints=stabilized,
            ground_contact=ground_contact,
            cameras=cameras_list,
            focal_lengths=focal_lengths,
            image_sizes=image_sizes,
            valid_indices=valid_indices,
            feet_indices=feet_indices,
            show_original=show_original,
            show_stabilized=show_stabilized,
            show_trajectories=show_trajectories,
            trajectory_length=trajectory_length,
            show_ground_contact=show_ground_contact,
            joint_radius=joint_radius,
        )
        
        # Convert to tensor
        overlay_tensor = torch.from_numpy(overlay.astype(np.float32) / 255.0)
        
        # Compute statistics for debug info
        displacement = np.linalg.norm(stabilized - original_joints, axis=-1)
        feet_disp = displacement[:, feet_indices].mean() if len(feet_indices) > 0 else 0
        body_disp = displacement[:, body_indices].mean() if len(body_indices) > 0 else 0
        
        ground_frames = 0
        if ground_contact is not None:
            ground_frames = int((ground_contact[:, feet_indices] > 0.5).any(axis=1).sum())
        
        debug_info = (
            f"=== Joint Temporal Stabilizer ===\n"
            f"Frames: {N} valid / {len(mesh_sequence)} total\n"
            f"Joints: {J} ({len(feet_indices)} feet, {len(body_indices)} body)\n"
            f"\n"
            f"=== Settings ===\n"
            f"Feet:  EMA={feet_ema_alpha}, hyst={feet_hysteresis_px}, damp={feet_velocity_damping}\n"
            f"Body:  EMA={body_ema_alpha}, hyst={body_hysteresis_px}, damp={body_velocity_damping}\n"
            f"Ground contact: {'enabled' if enable_ground_contact else 'disabled'}\n"
            f"\n"
            f"=== Results ===\n"
            f"Mean displacement (feet):  {feet_disp:.4f} units\n"
            f"Mean displacement (body):  {body_disp:.4f} units\n"
            f"Ground contact frames: {ground_frames}\n"
        )
        
        if not silent:
            print(f"[JointStabilizer] Complete! Mean disp: feet={feet_disp:.4f}, body={body_disp:.4f}")
        
        return (output_sequence, overlay_tensor, debug_info)

    def _get_joint_groups(
        self, 
        skeleton_format: str, 
        custom_feet: str, 
        num_joints: int,
        verbose: bool
    ) -> Tuple[List[int], List[int]]:
        """Get feet and body joint indices."""
        
        if custom_feet.strip():
            feet_indices = [int(i.strip()) for i in custom_feet.split(",") if i.strip().isdigit()]
            feet_indices = [i for i in feet_indices if i < num_joints]
        elif skeleton_format in SKELETON_CONFIGS and skeleton_format != "AUTO":
            config = SKELETON_CONFIGS[skeleton_format]
            feet_indices = [i for i in config["feet_indices"] if i < num_joints]
        else:
            if num_joints >= 100:
                feet_indices = [15, 16, 17, 18, 19, 20]
            elif num_joints >= 24:
                feet_indices = [7, 8, 10, 11]
            else:
                feet_indices = list(range(max(0, num_joints - 6), num_joints))
            feet_indices = [i for i in feet_indices if i < num_joints]
        
        body_indices = [i for i in range(num_joints) if i not in feet_indices]
        
        return feet_indices, body_indices

    def _stabilize_joints(
        self,
        joints: np.ndarray,
        feet_indices: List[int],
        body_indices: List[int],
        feet_ema_alpha: float,
        feet_hysteresis: float,
        feet_damping: float,
        body_ema_alpha: float,
        body_hysteresis: float,
        body_damping: float,
        enable_ground_contact: bool,
        ground_velocity_threshold: float,
        ground_contact_ema: float,
        enable_savgol: bool,
        savgol_window: int,
        savgol_order: int,
        verbose: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply multi-stage stabilization. Returns (stabilized_joints, ground_contact)."""
        
        N, J, D = joints.shape
        result = joints.copy()
        
        # Stage 0: Savitzky-Golay pre-smoothing
        if enable_savgol and SCIPY_AVAILABLE and N > savgol_window:
            for j in range(J):
                for d in range(D):
                    result[:, j, d] = savgol_filter(result[:, j, d], savgol_window, savgol_order)
        
        # Stage 1: Compute velocities
        velocities = np.zeros_like(result)
        velocities[1:] = result[1:] - result[:-1]
        velocity_magnitude = np.linalg.norm(velocities, axis=-1)
        
        # Stage 2: Ground contact detection
        ground_contact = np.zeros((N, J), dtype=np.float32)
        
        if enable_ground_contact:
            for j in feet_indices:
                contact_raw = (velocity_magnitude[:, j] < ground_velocity_threshold).astype(np.float32)
                ground_contact[:, j] = self._smooth_1d(contact_raw, alpha=0.3)
        
        # Stage 3: Apply stabilization per joint group
        
        # FEET
        for j in feet_indices:
            for d in range(D):
                if enable_ground_contact:
                    result[:, j, d] = self._adaptive_ema(
                        result[:, j, d], feet_ema_alpha, ground_contact_ema, ground_contact[:, j]
                    )
                else:
                    result[:, j, d] = self._bidirectional_ema(result[:, j, d], feet_ema_alpha)
            
            if enable_ground_contact:
                result[:, j, :] = self._adaptive_velocity_damping(
                    result[:, j, :], feet_damping, 0.95, ground_contact[:, j]
                )
            else:
                result[:, j, :] = self._velocity_damping(result[:, j, :], feet_damping)
            
            if enable_ground_contact:
                result[:, j, :] = self._adaptive_hysteresis(
                    result[:, j, :], feet_hysteresis, feet_hysteresis * 3, ground_contact[:, j]
                )
            else:
                result[:, j, :] = self._hysteresis(result[:, j, :], feet_hysteresis)
        
        # BODY
        for j in body_indices:
            for d in range(D):
                result[:, j, d] = self._bidirectional_ema(result[:, j, d], body_ema_alpha)
            result[:, j, :] = self._velocity_damping(result[:, j, :], body_damping)
            result[:, j, :] = self._hysteresis(result[:, j, :], body_hysteresis)
        
        return result, ground_contact

    def _render_debug_overlay(
        self,
        images_np: np.ndarray,
        original_joints: np.ndarray,
        stabilized_joints: np.ndarray,
        ground_contact: np.ndarray,
        cameras: List,
        focal_lengths: List,
        image_sizes: List,
        valid_indices: List[int],
        feet_indices: List[int],
        show_original: bool,
        show_stabilized: bool,
        show_trajectories: bool,
        trajectory_length: int,
        show_ground_contact: bool,
        joint_radius: int,
    ) -> np.ndarray:
        """Render debug overlay showing before/after comparison."""
        
        N_images = len(images_np)
        H, W = images_np.shape[1:3]
        overlay = images_np.copy()
        
        # Build frame index mapping
        valid_set = set(valid_indices)
        
        for img_idx in range(N_images):
            frame = overlay[img_idx]
            
            if img_idx not in valid_set:
                # No valid data for this frame
                cv2.putText(frame, "No joints", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["text_fg"], 2)
                continue
            
            # Find index in valid arrays
            valid_idx = valid_indices.index(img_idx)
            
            orig = original_joints[valid_idx]
            stab = stabilized_joints[valid_idx]
            gc = ground_contact[valid_idx] if ground_contact is not None else None
            camera = cameras[valid_idx]
            focal = focal_lengths[valid_idx]
            img_size = image_sizes[valid_idx]
            
            # Project 3D joints to 2D
            orig_2d = self._project_joints(orig, camera, focal, img_size)
            stab_2d = self._project_joints(stab, camera, focal, img_size)
            
            if orig_2d is None or stab_2d is None:
                cv2.putText(frame, "Projection failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["text_fg"], 2)
                continue
            
            # Draw trajectories
            if show_trajectories:
                start_idx = max(0, valid_idx - trajectory_length)
                for t in range(start_idx, valid_idx):
                    t_frame = valid_indices.index(valid_indices[t]) if valid_indices[t] in valid_indices else None
                    if t_frame is None:
                        continue
                    
                    prev_orig = self._project_joints(original_joints[t_frame], cameras[t_frame], focal_lengths[t_frame], image_sizes[t_frame])
                    prev_stab = self._project_joints(stabilized_joints[t_frame], cameras[t_frame], focal_lengths[t_frame], image_sizes[t_frame])
                    
                    if prev_orig is not None and prev_stab is not None:
                        alpha = (t - start_idx + 1) / (valid_idx - start_idx + 1)
                        
                        # Draw trajectory lines for feet only
                        for j in feet_indices:
                            if show_original:
                                pt1 = tuple(prev_orig[j].astype(int))
                                pt2 = tuple(orig_2d[j].astype(int))
                                cv2.line(frame, pt1, pt2, COLORS["trajectory_original"], 1)
                            
                            if show_stabilized:
                                pt1 = tuple(prev_stab[j].astype(int))
                                pt2 = tuple(stab_2d[j].astype(int))
                                cv2.line(frame, pt1, pt2, COLORS["trajectory_stabilized"], 1)
            
            # Draw joints
            J = orig.shape[0]
            for j in range(J):
                is_foot = j in feet_indices
                is_grounded = gc is not None and gc[j] > 0.5 if is_foot else False
                
                ox, oy = int(orig_2d[j, 0]), int(orig_2d[j, 1])
                sx, sy = int(stab_2d[j, 0]), int(stab_2d[j, 1])
                
                # Draw original (red)
                if show_original:
                    color = COLORS["feet_original"] if is_foot else COLORS["original_joint"]
                    cv2.circle(frame, (ox, oy), joint_radius, color, -1)
                    cv2.circle(frame, (ox, oy), joint_radius, (0, 0, 0), 1)  # Black outline
                
                # Draw stabilized (green)
                if show_stabilized:
                    color = COLORS["feet_stabilized"] if is_foot else COLORS["stabilized_joint"]
                    if is_grounded and show_ground_contact:
                        color = COLORS["ground_contact"]
                    cv2.circle(frame, (sx, sy), joint_radius, color, -1)
                    cv2.circle(frame, (sx, sy), joint_radius, (0, 0, 0), 1)
                
                # Draw line between original and stabilized
                if show_original and show_stabilized:
                    displacement = np.sqrt((ox - sx)**2 + (oy - sy)**2)
                    if displacement > 2:
                        cv2.line(frame, (ox, oy), (sx, sy), (255, 255, 255), 1)
            
            # Draw legend
            y_offset = 20
            if show_original:
                cv2.circle(frame, (15, y_offset), 6, COLORS["original_joint"], -1)
                cv2.putText(frame, "Original", (30, y_offset + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text_fg"], 1)
                y_offset += 20
            
            if show_stabilized:
                cv2.circle(frame, (15, y_offset), 6, COLORS["stabilized_joint"], -1)
                cv2.putText(frame, "Stabilized", (30, y_offset + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text_fg"], 1)
                y_offset += 20
            
            if show_ground_contact and gc is not None:
                grounded = any(gc[j] > 0.5 for j in feet_indices)
                if grounded:
                    cv2.circle(frame, (15, y_offset), 6, COLORS["ground_contact"], -1)
                    cv2.putText(frame, "Grounded", (30, y_offset + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text_fg"], 1)
            
            # Frame number
            cv2.putText(frame, f"Frame {img_idx}", (W - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text_fg"], 1)
        
        return overlay

    def _project_joints(
        self, 
        joints_3d: np.ndarray, 
        camera: Optional[np.ndarray], 
        focal_length: Optional[float],
        image_size: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """Project 3D joints to 2D image coordinates."""
        
        if joints_3d is None:
            return None
        
        W, H = image_size
        cx, cy = W / 2, H / 2
        
        # Get focal length
        if focal_length is not None:
            if hasattr(focal_length, '__len__'):
                f = float(np.array(focal_length).flatten()[0])
            else:
                f = float(focal_length)
        else:
            f = max(W, H)  # Fallback
        
        # Simple perspective projection
        # Assuming joints are in camera space (z forward)
        z = joints_3d[:, 2:3]
        z = np.clip(z, 0.1, None)  # Avoid division by zero
        
        x = joints_3d[:, 0] * f / z.flatten() + cx
        y = joints_3d[:, 1] * f / z.flatten() + cy
        
        # Clip to image bounds
        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)
        
        return np.stack([x, y], axis=-1)

    # =========================================================================
    # Filter Methods
    # =========================================================================
    
    def _smooth_1d(self, signal: np.ndarray, alpha: float) -> np.ndarray:
        result = signal.copy()
        for t in range(1, len(signal)):
            result[t] = alpha * signal[t] + (1 - alpha) * result[t-1]
        return result
    
    def _bidirectional_ema(self, signal: np.ndarray, alpha: float) -> np.ndarray:
        N = len(signal)
        forward = signal.copy()
        for t in range(1, N):
            forward[t] = alpha * signal[t] + (1 - alpha) * forward[t-1]
        backward = signal.copy()
        for t in range(N - 2, -1, -1):
            backward[t] = alpha * signal[t] + (1 - alpha) * backward[t+1]
        return 0.5 * (forward + backward)
    
    def _adaptive_ema(self, signal, alpha_base, alpha_grounded, ground_contact):
        N = len(signal)
        result = signal.copy()
        for t in range(1, N):
            alpha = alpha_grounded * ground_contact[t] + alpha_base * (1 - ground_contact[t])
            result[t] = alpha * signal[t] + (1 - alpha) * result[t-1]
        return result
    
    def _velocity_damping(self, joints: np.ndarray, damping: float) -> np.ndarray:
        N = len(joints)
        result = joints.copy()
        for t in range(1, N):
            velocity = result[t] - result[t-1]
            result[t] = result[t-1] + velocity * (1 - damping)
        return result
    
    def _adaptive_velocity_damping(self, joints, damping_base, damping_grounded, ground_contact):
        N = len(joints)
        result = joints.copy()
        for t in range(1, N):
            damping = damping_grounded * ground_contact[t] + damping_base * (1 - ground_contact[t])
            velocity = result[t] - result[t-1]
            result[t] = result[t-1] + velocity * (1 - damping)
        return result
    
    def _hysteresis(self, joints: np.ndarray, threshold: float) -> np.ndarray:
        N = len(joints)
        result = joints.copy()
        for t in range(1, N):
            if np.linalg.norm(joints[t] - result[t-1]) < threshold:
                result[t] = result[t-1]
        return result
    
    def _adaptive_hysteresis(self, joints, threshold_base, threshold_grounded, ground_contact):
        N = len(joints)
        result = joints.copy()
        for t in range(1, N):
            threshold = threshold_grounded * ground_contact[t] + threshold_base * (1 - ground_contact[t])
            if np.linalg.norm(joints[t] - result[t-1]) < threshold:
                result[t] = result[t-1]
        return result


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "JointTemporalStabilizer": JointTemporalStabilizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JointTemporalStabilizer": "🦴 Joint Temporal Stabilizer",
}
