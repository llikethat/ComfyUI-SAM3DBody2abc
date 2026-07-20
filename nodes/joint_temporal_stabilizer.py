"""
Joint Temporal Stabilizer for SAM3DBody2abc
============================================
Version: 1.3.0 - Debug mesh_sequence structure
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union
from copy import deepcopy

try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


SKELETON_CONFIGS = {
    "MHR_127": {"feet_indices": [15, 16, 17, 18, 19, 20]},
    "MHR_70": {"feet_indices": [15, 16, 17, 18, 19, 20]},
    "SMPL_24": {"feet_indices": [7, 8, 10, 11]},
    "HALPE_26": {"feet_indices": [15, 16, 20, 21, 22, 23, 24, 25]},
    "AUTO": {"feet_indices": []},
}

COLORS = {
    "original_joint": (0, 0, 255),
    "stabilized_joint": (0, 255, 0),
    "feet_original": (0, 100, 255),
    "feet_stabilized": (0, 255, 100),
    "ground_contact": (255, 255, 0),
    "trajectory_original": (0, 0, 200),
    "trajectory_stabilized": (0, 200, 0),
    "text_fg": (255, 255, 255),
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
                "skeleton_format": (list(SKELETON_CONFIGS.keys()), {"default": "AUTO"}),
                "custom_feet_indices": ("STRING", {"default": ""}),
                "feet_ema_alpha": ("FLOAT", {"default": 0.15, "min": 0.05, "max": 1.0, "step": 0.05}),
                "feet_hysteresis_px": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 20.0, "step": 0.5}),
                "feet_velocity_damping": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 0.95, "step": 0.05}),
                "body_ema_alpha": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 1.0, "step": 0.05}),
                "body_hysteresis_px": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "body_velocity_damping": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.9, "step": 0.05}),
                "enable_ground_contact": ("BOOLEAN", {"default": True}),
                "ground_velocity_threshold": ("FLOAT", {"default": 0.02, "min": 0.005, "max": 0.1, "step": 0.005}),
                "ground_contact_ema": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.2, "step": 0.01}),
                "enable_savgol": ("BOOLEAN", {"default": False}),
                "savgol_window": ("INT", {"default": 5, "min": 3, "max": 15, "step": 2}),
                "savgol_order": ("INT", {"default": 2, "min": 1, "max": 4}),
                "show_original": ("BOOLEAN", {"default": True}),
                "show_stabilized": ("BOOLEAN", {"default": True}),
                "show_trajectories": ("BOOLEAN", {"default": True}),
                "trajectory_length": ("INT", {"default": 15, "min": 5, "max": 60}),
                "show_ground_contact": ("BOOLEAN", {"default": True}),
                "joint_radius": ("INT", {"default": 4, "min": 2, "max": 10}),
                "log_level": (["silent", "normal", "verbose"], {"default": "verbose"}),
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
        feet_ema_alpha: float = 0.15,
        feet_hysteresis_px: float = 3.0,
        feet_velocity_damping: float = 0.8,
        body_ema_alpha: float = 0.4,
        body_hysteresis_px: float = 1.0,
        body_velocity_damping: float = 0.3,
        enable_ground_contact: bool = True,
        ground_velocity_threshold: float = 0.02,
        ground_contact_ema: float = 0.05,
        enable_savgol: bool = False,
        savgol_window: int = 5,
        savgol_order: int = 2,
        show_original: bool = True,
        show_stabilized: bool = True,
        show_trajectories: bool = True,
        trajectory_length: int = 15,
        show_ground_contact: bool = True,
        joint_radius: int = 4,
        log_level: str = "verbose",
    ):
        verbose = log_level == "verbose"
        silent = log_level == "silent"
        
        # =====================================================================
        # DEBUG: Print mesh_sequence structure
        # =====================================================================
        print(f"[JointStabilizer] === DEBUG: mesh_sequence structure ===")
        print(f"[JointStabilizer] Type: {type(mesh_sequence)}")
        
        if isinstance(mesh_sequence, dict):
            print(f"[JointStabilizer] Keys: {list(mesh_sequence.keys())}")
            for key in list(mesh_sequence.keys())[:5]:
                val = mesh_sequence[key]
                print(f"[JointStabilizer]   '{key}': {type(val)}")
                if isinstance(val, dict):
                    print(f"[JointStabilizer]     Sub-keys: {list(val.keys())[:5]}")
                elif isinstance(val, list):
                    print(f"[JointStabilizer]     List len: {len(val)}, first elem type: {type(val[0]) if val else 'empty'}")
        elif isinstance(mesh_sequence, list):
            print(f"[JointStabilizer] List length: {len(mesh_sequence)}")
            if mesh_sequence:
                print(f"[JointStabilizer] First element type: {type(mesh_sequence[0])}")
                if isinstance(mesh_sequence[0], dict):
                    print(f"[JointStabilizer] First element keys: {list(mesh_sequence[0].keys())[:10]}")
        else:
            print(f"[JointStabilizer] Unknown type: {mesh_sequence}")
        
        print(f"[JointStabilizer] === END DEBUG ===")
        
        # =====================================================================
        # Extract frames - handle multiple possible structures
        # =====================================================================
        frames = []
        frame_keys = None
        
        if isinstance(mesh_sequence, dict):
            # Check for "frames" key
            if "frames" in mesh_sequence:
                frames_data = mesh_sequence["frames"]
                if isinstance(frames_data, dict):
                    frame_keys = sorted(frames_data.keys())
                    frames = [frames_data[k] for k in frame_keys]
                    print(f"[JointStabilizer] Extracted {len(frames)} frames from mesh_sequence['frames'] (dict)")
                elif isinstance(frames_data, list):
                    frames = frames_data
                    print(f"[JointStabilizer] Extracted {len(frames)} frames from mesh_sequence['frames'] (list)")
            else:
                # Maybe mesh_sequence IS the frames dict directly
                # Check if keys look like frame indices
                keys = list(mesh_sequence.keys())
                if keys and all(isinstance(k, int) or (isinstance(k, str) and k.isdigit()) for k in keys[:5]):
                    frame_keys = sorted(mesh_sequence.keys())
                    frames = [mesh_sequence[k] for k in frame_keys]
                    print(f"[JointStabilizer] Extracted {len(frames)} frames directly from mesh_sequence keys")
        elif isinstance(mesh_sequence, list):
            frames = mesh_sequence
            print(f"[JointStabilizer] mesh_sequence is list with {len(frames)} items")
        
        if not frames:
            print(f"[JointStabilizer] ERROR: Could not extract frames from mesh_sequence")
            if torch.is_tensor(images):
                images_np = images.cpu().numpy()
            else:
                images_np = np.array(images)
            if images_np.max() <= 1.0:
                images_np = (images_np * 255).astype(np.uint8)
            empty_overlay = torch.from_numpy(images_np.astype(np.float32) / 255.0)
            return (mesh_sequence, empty_overlay, f"Could not extract frames. Type: {type(mesh_sequence)}")
        
        # Debug first frame
        print(f"[JointStabilizer] First frame type: {type(frames[0])}")
        if isinstance(frames[0], dict):
            print(f"[JointStabilizer] First frame keys: {list(frames[0].keys())[:10]}")
        
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
        
        # Extract joints
        joints_list = []
        cameras_list = []
        focal_lengths = []
        image_sizes = []
        valid_frame_indices = []
        
        for i, frame in enumerate(frames):
            # Skip non-dict frames
            if not isinstance(frame, dict):
                print(f"[JointStabilizer] Frame {i} is not a dict: {type(frame)}")
                continue
            
            is_valid = frame.get("valid", True)  # Default to True if not specified
            joints = frame.get("joints")
            
            if joints is not None:
                if isinstance(joints, torch.Tensor):
                    joints = joints.cpu().numpy()
                joints_list.append(joints)
                
                camera = frame.get("camera")
                if camera is not None and isinstance(camera, torch.Tensor):
                    camera = camera.cpu().numpy()
                cameras_list.append(camera)
                focal_lengths.append(frame.get("focal_length"))
                image_sizes.append(frame.get("image_size", (images_np.shape[2], images_np.shape[1])))
                valid_frame_indices.append(i)
        
        print(f"[JointStabilizer] Found {len(joints_list)} frames with joints")
        
        if len(joints_list) == 0:
            print(f"[JointStabilizer] No joints found!")
            empty_overlay = torch.from_numpy(images_np.astype(np.float32) / 255.0)
            return (mesh_sequence, empty_overlay, "No joints found in frames")
        
        # Stack joints
        joints_array = np.stack(joints_list, axis=0)
        original_joints = joints_array.copy()
        N, J, D = joints_array.shape
        
        print(f"[JointStabilizer] Processing {N} frames, {J} joints, {D}D")
        
        # Get joint groups
        feet_indices, body_indices = self._get_joint_groups(skeleton_format, custom_feet_indices, J)
        print(f"[JointStabilizer] Feet: {feet_indices}, Body: {len(body_indices)} joints")
        
        # Stabilize
        stabilized, ground_contact = self._stabilize_joints(
            joints_array, feet_indices, body_indices,
            feet_ema_alpha, feet_hysteresis_px, feet_velocity_damping,
            body_ema_alpha, body_hysteresis_px, body_velocity_damping,
            enable_ground_contact, ground_velocity_threshold, ground_contact_ema,
            enable_savgol, savgol_window, savgol_order
        )
        
        # Update frames
        for i, frame_idx in enumerate(valid_frame_indices):
            frames[frame_idx]["joints"] = stabilized[i]
        
        # Rebuild output
        output_sequence = deepcopy(mesh_sequence) if isinstance(mesh_sequence, dict) else {}
        if frame_keys is not None:
            output_sequence["frames"] = {k: frames[i] for i, k in enumerate(frame_keys)}
        else:
            output_sequence["frames"] = frames
        
        # Render overlay
        overlay = self._render_debug_overlay(
            images_np, original_joints, stabilized, ground_contact,
            cameras_list, focal_lengths, image_sizes, valid_frame_indices,
            feet_indices, show_original, show_stabilized, show_trajectories,
            trajectory_length, show_ground_contact, joint_radius
        )
        overlay_tensor = torch.from_numpy(overlay.astype(np.float32) / 255.0)
        
        # Stats
        displacement = np.linalg.norm(stabilized - original_joints, axis=-1)
        feet_disp = displacement[:, feet_indices].mean() if feet_indices else 0
        body_disp = displacement[:, body_indices].mean() if body_indices else 0
        
        debug_info = (
            f"Frames: {N}, Joints: {J}\n"
            f"Feet indices: {feet_indices}\n"
            f"Mean disp - feet: {feet_disp:.4f}, body: {body_disp:.4f}"
        )
        
        print(f"[JointStabilizer] Done!")
        return (output_sequence, overlay_tensor, debug_info)

    def _get_joint_groups(self, skeleton_format, custom_feet, num_joints):
        if custom_feet.strip():
            feet = [int(i.strip()) for i in custom_feet.split(",") if i.strip().isdigit()]
            feet = [i for i in feet if i < num_joints]
        elif skeleton_format in SKELETON_CONFIGS and skeleton_format != "AUTO":
            feet = [i for i in SKELETON_CONFIGS[skeleton_format]["feet_indices"] if i < num_joints]
        else:
            if num_joints >= 100:
                feet = [15, 16, 17, 18, 19, 20]
            elif num_joints >= 24:
                feet = [7, 8, 10, 11]
            else:
                feet = list(range(max(0, num_joints - 6), num_joints))
            feet = [i for i in feet if i < num_joints]
        body = [i for i in range(num_joints) if i not in feet]
        return feet, body

    def _stabilize_joints(self, joints, feet_idx, body_idx,
                          f_ema, f_hyst, f_damp, b_ema, b_hyst, b_damp,
                          en_gc, gc_th, gc_ema, en_sav, sav_w, sav_o):
        N, J, D = joints.shape
        result = joints.copy()
        
        if en_sav and SCIPY_AVAILABLE and N > sav_w:
            for j in range(J):
                for d in range(D):
                    result[:, j, d] = savgol_filter(result[:, j, d], sav_w, sav_o)
        
        vel = np.zeros_like(result)
        vel[1:] = result[1:] - result[:-1]
        vel_mag = np.linalg.norm(vel, axis=-1)
        
        gc = np.zeros((N, J), dtype=np.float32)
        if en_gc:
            for j in feet_idx:
                raw = (vel_mag[:, j] < gc_th).astype(np.float32)
                gc[:, j] = self._smooth_1d(raw, 0.3)
        
        for j in feet_idx:
            for d in range(D):
                if en_gc:
                    result[:, j, d] = self._adaptive_ema(result[:, j, d], f_ema, gc_ema, gc[:, j])
                else:
                    result[:, j, d] = self._bidir_ema(result[:, j, d], f_ema)
            if en_gc:
                result[:, j, :] = self._adaptive_vel_damp(result[:, j, :], f_damp, 0.95, gc[:, j])
            else:
                result[:, j, :] = self._vel_damp(result[:, j, :], f_damp)
            if en_gc:
                result[:, j, :] = self._adaptive_hyst(result[:, j, :], f_hyst, f_hyst * 3, gc[:, j])
            else:
                result[:, j, :] = self._hyst(result[:, j, :], f_hyst)
        
        for j in body_idx:
            for d in range(D):
                result[:, j, d] = self._bidir_ema(result[:, j, d], b_ema)
            result[:, j, :] = self._vel_damp(result[:, j, :], b_damp)
            result[:, j, :] = self._hyst(result[:, j, :], b_hyst)
        
        return result, gc

    def _render_debug_overlay(self, images_np, original, stabilized, gc,
                               cameras, focals, sizes, valid_idx, feet_idx,
                               show_o, show_s, show_t, t_len, show_gc, radius):
        N_img = len(images_np)
        H, W = images_np.shape[1:3]
        overlay = images_np.copy()
        valid_set = set(valid_idx)
        
        for img_i in range(N_img):
            frame = overlay[img_i]
            if img_i not in valid_set:
                cv2.putText(frame, "No joints", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["text_fg"], 2)
                continue
            
            vi = valid_idx.index(img_i)
            orig_2d = self._project(original[vi], focals[vi], sizes[vi])
            stab_2d = self._project(stabilized[vi], focals[vi], sizes[vi])
            if orig_2d is None:
                continue
            
            gc_frame = gc[vi] if gc is not None else None
            
            for j in range(original[vi].shape[0]):
                is_foot = j in feet_idx
                grounded = gc_frame is not None and gc_frame[j] > 0.5 if is_foot else False
                ox, oy = int(orig_2d[j, 0]), int(orig_2d[j, 1])
                sx, sy = int(stab_2d[j, 0]), int(stab_2d[j, 1])
                
                if show_o:
                    c = COLORS["feet_original"] if is_foot else COLORS["original_joint"]
                    cv2.circle(frame, (ox, oy), radius, c, -1)
                if show_s:
                    c = COLORS["ground_contact"] if (grounded and show_gc) else (COLORS["feet_stabilized"] if is_foot else COLORS["stabilized_joint"])
                    cv2.circle(frame, (sx, sy), radius, c, -1)
            
            cv2.putText(frame, f"F{img_i}", (W-50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text_fg"], 1)
        
        return overlay

    def _project(self, j3d, focal, size):
        if j3d is None: return None
        W, H = size
        f = float(np.array(focal).flatten()[0]) if focal is not None and hasattr(focal, '__len__') else (float(focal) if focal else max(W, H))
        z = np.clip(j3d[:, 2:3], 0.1, None)
        x = j3d[:, 0] * f / z.flatten() + W / 2
        y = j3d[:, 1] * f / z.flatten() + H / 2
        return np.stack([np.clip(x, 0, W-1), np.clip(y, 0, H-1)], axis=-1)

    def _smooth_1d(self, s, a):
        r = s.copy()
        for t in range(1, len(s)): r[t] = a * s[t] + (1-a) * r[t-1]
        return r
    def _bidir_ema(self, s, a):
        f = s.copy()
        for t in range(1, len(s)): f[t] = a * s[t] + (1-a) * f[t-1]
        b = s.copy()
        for t in range(len(s)-2, -1, -1): b[t] = a * s[t] + (1-a) * b[t+1]
        return 0.5 * (f + b)
    def _adaptive_ema(self, s, a_b, a_g, gc):
        r = s.copy()
        for t in range(1, len(s)):
            a = a_g * gc[t] + a_b * (1 - gc[t])
            r[t] = a * s[t] + (1-a) * r[t-1]
        return r
    def _vel_damp(self, j, d):
        r = j.copy()
        for t in range(1, len(j)): r[t] = r[t-1] + (r[t] - r[t-1]) * (1-d)
        return r
    def _adaptive_vel_damp(self, j, d_b, d_g, gc):
        r = j.copy()
        for t in range(1, len(j)):
            d = d_g * gc[t] + d_b * (1 - gc[t])
            r[t] = r[t-1] + (r[t] - r[t-1]) * (1-d)
        return r
    def _hyst(self, j, th):
        r = j.copy()
        for t in range(1, len(j)):
            if np.linalg.norm(j[t] - r[t-1]) < th: r[t] = r[t-1]
        return r
    def _adaptive_hyst(self, j, th_b, th_g, gc):
        r = j.copy()
        for t in range(1, len(j)):
            th = th_g * gc[t] + th_b * (1 - gc[t])
            if np.linalg.norm(j[t] - r[t-1]) < th: r[t] = r[t-1]
        return r


NODE_CLASS_MAPPINGS = {"JointTemporalStabilizer": JointTemporalStabilizer}
NODE_DISPLAY_NAME_MAPPINGS = {"JointTemporalStabilizer": "🦴 Joint Temporal Stabilizer"}
