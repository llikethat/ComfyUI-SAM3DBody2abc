"""
Joint Temporal Stabilizer for SAM3DBody2abc
============================================
Version: 2.0.0 - FIXED: Smooth in local space, not camera space
"""

import numpy as np
import torch
import cv2
import sys
from typing import Dict, List, Tuple, Optional, Any
from copy import deepcopy

try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def log(msg):
    print(f"[JointStabilizer] {msg}", flush=True)


SKELETON_CONFIGS = {
    "MHR_127": {"feet_indices": [14, 15, 16, 17, 18, 19]},
    "SMPL_24": {"feet_indices": [7, 8, 10, 11]},
    "AUTO": {"feet_indices": []},
}

COLORS = {
    "original": (0, 0, 255),         # Red
    "stabilized": (0, 255, 0),       # Green  
    "grounded": (0, 255, 255),       # Yellow
    "text": (255, 255, 255),
}


class JointTemporalStabilizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mesh_sequence": ("MESH_SEQUENCE",),
            },
            "optional": {
                "skeleton_format": (["AUTO", "MHR_127", "SMPL_24"], {"default": "AUTO"}),
                "custom_feet_indices": ("STRING", {"default": ""}),
                
                # Smoothing - REDUCED defaults
                "smoothing_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0=no smoothing, 1=max smoothing. Start low!"}),
                "feet_extra_smoothing": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.5, "step": 0.05,
                    "tooltip": "Extra smoothing for feet (added to base)"}),
                
                # Ground contact
                "enable_ground_contact": ("BOOLEAN", {"default": True}),
                "ground_velocity_threshold": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.05, "step": 0.001}),
                
                # Visualization
                "show_feet_only": ("BOOLEAN", {"default": True}),
                "show_original": ("BOOLEAN", {"default": True}),
                "show_stabilized": ("BOOLEAN", {"default": True}),
                "joint_radius": ("INT", {"default": 4, "min": 1, "max": 10}),
            },
        }

    RETURN_TYPES = ("MESH_SEQUENCE", "IMAGE", "STRING")
    RETURN_NAMES = ("stabilized_sequence", "debug_overlay", "debug_info")
    FUNCTION = "process"
    CATEGORY = "SAM3DBody2abc/Processing"

    def process(
        self,
        images,
        mesh_sequence,
        skeleton_format: str = "AUTO",
        custom_feet_indices: str = "",
        smoothing_strength: float = 0.3,
        feet_extra_smoothing: float = 0.2,
        enable_ground_contact: bool = True,
        ground_velocity_threshold: float = 0.01,
        show_feet_only: bool = True,
        show_original: bool = True,
        show_stabilized: bool = True,
        joint_radius: int = 4,
    ):
        log("=" * 50)
        log("Joint Temporal Stabilizer v2.0")
        log("=" * 50)
        
        # Extract frames
        frames = []
        frame_keys = None
        
        if isinstance(mesh_sequence, dict) and "frames" in mesh_sequence:
            frames_data = mesh_sequence["frames"]
            if isinstance(frames_data, dict):
                frame_keys = sorted(frames_data.keys())
                frames = [frames_data[k] for k in frame_keys]
            else:
                frames = list(frames_data)
        elif isinstance(mesh_sequence, list):
            frames = mesh_sequence
        
        log(f"Frames: {len(frames)}")
        
        # Convert images
        if torch.is_tensor(images):
            images_np = images.cpu().numpy()
        else:
            images_np = np.array(images)
        if images_np.dtype != np.uint8:
            if images_np.max() <= 1.0:
                images_np = (images_np * 255).astype(np.uint8)
            else:
                images_np = images_np.astype(np.uint8)
        
        N_images, H, W = images_np.shape[:3]
        log(f"Images: {N_images} x {W}x{H}")
        
        if not frames:
            empty = torch.from_numpy(images_np.astype(np.float32) / 255.0)
            return (mesh_sequence, empty, "No frames")
        
        # =====================================================================
        # Extract joints in LOCAL SPACE (not camera space!)
        # =====================================================================
        local_joints_list = []  # Joint coords in local/canonical space
        cam_t_list = []         # Camera translations (for projection only)
        focal_list = []
        cx_list = []
        cy_list = []
        valid_indices = []
        
        for i, frame in enumerate(frames):
            if not isinstance(frame, dict):
                continue
            
            # Get local joint coordinates
            joints = None
            for key in ["joint_coords", "joints"]:
                if key in frame and frame[key] is not None:
                    joints = frame[key]
                    break
            
            if joints is None:
                continue
            
            if isinstance(joints, torch.Tensor):
                joints = joints.cpu().numpy()
            joints = np.array(joints).copy()
            
            # Get camera translation (for projection only, NOT for smoothing)
            cam_t = frame.get("pred_cam_t")
            if cam_t is not None:
                if isinstance(cam_t, torch.Tensor):
                    cam_t = cam_t.cpu().numpy()
                cam_t = np.array(cam_t).flatten()
            else:
                cam_t = np.zeros(3)
            
            local_joints_list.append(joints)
            cam_t_list.append(cam_t)
            focal_list.append(frame.get("focal_length", max(W, H)))
            cx_list.append(frame.get("cx", W / 2))
            cy_list.append(frame.get("cy", H / 2))
            valid_indices.append(i)
        
        if len(local_joints_list) == 0:
            empty = torch.from_numpy(images_np.astype(np.float32) / 255.0)
            return (mesh_sequence, empty, "No joints found")
        
        # Stack: (N, J, 3) in LOCAL space
        local_joints = np.stack(local_joints_list, axis=0)
        original_local = local_joints.copy()
        cam_t_array = np.stack(cam_t_list, axis=0)
        
        N, J, D = local_joints.shape
        log(f"Joints: {N} frames, {J} joints (LOCAL space)")
        
        # Get feet indices
        feet_indices = self._get_feet_indices(skeleton_format, custom_feet_indices, J)
        body_indices = [i for i in range(J) if i not in feet_indices]
        log(f"Feet indices: {feet_indices}")
        
        # =====================================================================
        # Smooth in LOCAL SPACE (this is the key fix!)
        # =====================================================================
        stabilized_local = local_joints.copy()
        
        # Convert strength to EMA alpha (higher alpha = less smoothing)
        body_alpha = 1.0 - smoothing_strength * 0.5  # 0.3 strength -> 0.85 alpha
        feet_alpha = 1.0 - (smoothing_strength + feet_extra_smoothing) * 0.5
        feet_alpha = max(0.3, feet_alpha)  # Don't go too low
        
        log(f"EMA alpha: body={body_alpha:.2f}, feet={feet_alpha:.2f}")
        
        # Ground contact detection (based on velocity in local space)
        ground_contact = np.zeros((N, J), dtype=np.float32)
        if enable_ground_contact and N > 1:
            vel = np.zeros_like(local_joints)
            vel[1:] = local_joints[1:] - local_joints[:-1]
            vel_mag = np.linalg.norm(vel, axis=-1)
            
            for j in feet_indices:
                ground_contact[:, j] = (vel_mag[:, j] < ground_velocity_threshold).astype(np.float32)
                # Smooth the contact signal
                for t in range(1, N):
                    ground_contact[t, j] = 0.7 * ground_contact[t, j] + 0.3 * ground_contact[t-1, j]
        
        # Apply bidirectional EMA smoothing
        for j in range(J):
            alpha = feet_alpha if j in feet_indices else body_alpha
            
            for d in range(D):
                signal = stabilized_local[:, j, d]
                
                # Forward pass
                forward = signal.copy()
                for t in range(1, N):
                    forward[t] = alpha * signal[t] + (1 - alpha) * forward[t-1]
                
                # Backward pass
                backward = signal.copy()
                for t in range(N - 2, -1, -1):
                    backward[t] = alpha * signal[t] + (1 - alpha) * backward[t+1]
                
                # Average
                stabilized_local[:, j, d] = 0.5 * (forward + backward)
        
        log("Stabilization complete (LOCAL space)")
        
        # =====================================================================
        # Update frames with stabilized LOCAL joints
        # =====================================================================
        for i, frame_idx in enumerate(valid_indices):
            for key in ["joint_coords", "joints"]:
                if key in frames[frame_idx] and frames[frame_idx][key] is not None:
                    frames[frame_idx][key] = stabilized_local[i]
                    break
        
        # Rebuild output
        output_sequence = deepcopy(mesh_sequence) if isinstance(mesh_sequence, dict) else {}
        if frame_keys is not None:
            output_sequence["frames"] = {k: frames[i] for i, k in enumerate(frame_keys)}
        elif isinstance(mesh_sequence, dict):
            output_sequence["frames"] = frames
        
        # =====================================================================
        # Render overlay (project to 2D using camera params)
        # =====================================================================
        log("Rendering overlay...")
        overlay = images_np.copy()
        
        # Decide which joints to visualize
        vis_joints = feet_indices if show_feet_only else list(range(J))
        
        for vi, frame_idx in enumerate(valid_indices):
            if frame_idx >= N_images:
                continue
            
            img = overlay[frame_idx]
            
            # Get projection params
            focal = focal_list[vi]
            if hasattr(focal, '__len__'):
                focal = float(np.array(focal).flatten()[0])
            else:
                focal = float(focal) if focal else max(W, H)
            cx = float(cx_list[vi]) if cx_list[vi] else W / 2
            cy = float(cy_list[vi]) if cy_list[vi] else H / 2
            cam_t = cam_t_array[vi]
            
            # Project LOCAL joints to 2D (add cam_t for camera space, then project)
            orig_cam = original_local[vi] + cam_t.reshape(1, 3)
            stab_cam = stabilized_local[vi] + cam_t.reshape(1, 3)
            
            orig_2d = self._project(orig_cam, focal, cx, cy, W, H)
            stab_2d = self._project(stab_cam, focal, cx, cy, W, H)
            
            gc = ground_contact[vi]
            
            # Draw joints
            for j in vis_joints:
                ox, oy = int(orig_2d[j, 0]), int(orig_2d[j, 1])
                sx, sy = int(stab_2d[j, 0]), int(stab_2d[j, 1])
                is_grounded = j in feet_indices and gc[j] > 0.5
                
                # Original (red)
                if show_original:
                    cv2.circle(img, (ox, oy), joint_radius, COLORS["original"], -1)
                    cv2.circle(img, (ox, oy), joint_radius, (0,0,0), 1)
                
                # Stabilized (green or yellow if grounded)
                if show_stabilized:
                    color = COLORS["grounded"] if is_grounded else COLORS["stabilized"]
                    cv2.circle(img, (sx, sy), joint_radius, color, -1)
                    cv2.circle(img, (sx, sy), joint_radius, (0,0,0), 1)
                
                # Displacement line
                if show_original and show_stabilized:
                    dist = np.sqrt((ox-sx)**2 + (oy-sy)**2)
                    if dist > 2:
                        cv2.line(img, (ox, oy), (sx, sy), (255,255,255), 1)
            
            # Legend
            cv2.rectangle(img, (5, 5), (110, 65), (0,0,0), -1)
            cv2.circle(img, (15, 18), 5, COLORS["original"], -1)
            cv2.putText(img, "Original", (28, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["text"], 1)
            cv2.circle(img, (15, 36), 5, COLORS["stabilized"], -1)
            cv2.putText(img, "Stabilized", (28, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["text"], 1)
            cv2.circle(img, (15, 54), 5, COLORS["grounded"], -1)
            cv2.putText(img, "Grounded", (28, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["text"], 1)
        
        overlay_tensor = torch.from_numpy(overlay.astype(np.float32) / 255.0)
        
        # Stats
        disp = np.linalg.norm(stabilized_local - original_local, axis=-1)
        feet_disp = disp[:, feet_indices].mean() if feet_indices else 0
        body_disp = disp[:, body_indices].mean() if body_indices else 0
        
        debug_info = (
            f"=== Joint Temporal Stabilizer v2.0 ===\n"
            f"Frames: {N}, Joints: {J}\n"
            f"Smoothing: strength={smoothing_strength}, feet_extra={feet_extra_smoothing}\n"
            f"EMA alpha: body={body_alpha:.2f}, feet={feet_alpha:.2f}\n"
            f"Mean displacement: feet={feet_disp:.4f}, body={body_disp:.4f}\n"
            f"(Lower displacement = closer to original)"
        )
        
        log(f"Done! disp: feet={feet_disp:.4f}, body={body_disp:.4f}")
        
        return (output_sequence, overlay_tensor, debug_info)

    def _get_feet_indices(self, skeleton_format, custom_feet, num_joints):
        if custom_feet.strip():
            return [int(i.strip()) for i in custom_feet.split(",") if i.strip().isdigit() and int(i.strip()) < num_joints]
        if skeleton_format in SKELETON_CONFIGS:
            return [i for i in SKELETON_CONFIGS[skeleton_format]["feet_indices"] if i < num_joints]
        # AUTO
        if num_joints >= 100:
            return [14, 15, 16, 17, 18, 19]
        elif num_joints >= 24:
            return [7, 8, 10, 11]
        return list(range(max(0, num_joints - 6), num_joints))

    def _project(self, joints_3d, focal, cx, cy, W, H):
        z = np.clip(joints_3d[:, 2:3], 0.1, None)
        x = joints_3d[:, 0] * focal / z.flatten() + cx
        y = joints_3d[:, 1] * focal / z.flatten() + cy
        return np.stack([np.clip(x, 0, W-1), np.clip(y, 0, H-1)], axis=-1)


NODE_CLASS_MAPPINGS = {"JointTemporalStabilizer": JointTemporalStabilizer}
NODE_DISPLAY_NAME_MAPPINGS = {"JointTemporalStabilizer": "🦴 Joint Temporal Stabilizer"}
