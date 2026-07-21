"""
Joint Temporal Stabilizer for SAM3DBody2abc
============================================
Version: 1.7.0 - Fixed: transform joints to camera space using pred_cam_t
"""

import numpy as np
import torch
import cv2
import sys
from typing import Dict, List, Tuple, Optional, Any, Union
from copy import deepcopy

try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def log(msg):
    print(f"[JointStabilizer] {msg}", flush=True)
    sys.stdout.flush()


SKELETON_CONFIGS = {
    "MHR_127": {"feet_indices": [14, 15, 16, 17, 18, 19]},
    "MHR_70": {"feet_indices": [14, 15, 16, 17, 18, 19]},
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
                "show_ground_contact": ("BOOLEAN", {"default": True}),
                "joint_radius": ("INT", {"default": 6, "min": 2, "max": 15}),
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
        show_ground_contact: bool = True,
        joint_radius: int = 6,
    ):
        log("=" * 50)
        log("STARTING Joint Temporal Stabilizer")
        log("=" * 50)
        
        # Extract frames
        frames = []
        frame_keys = None
        
        if isinstance(mesh_sequence, dict):
            if "frames" in mesh_sequence:
                frames_data = mesh_sequence["frames"]
                if isinstance(frames_data, dict):
                    frame_keys = sorted(frames_data.keys())
                    frames = [frames_data[k] for k in frame_keys]
                elif isinstance(frames_data, list):
                    frames = frames_data
        elif isinstance(mesh_sequence, list):
            frames = mesh_sequence
        
        log(f"Extracted {len(frames)} frames from mesh_sequence")
        
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
        log(f"Images: {N_images} frames, {W}x{H}")
        
        if not frames:
            log("ERROR: No frames found!")
            empty_overlay = torch.from_numpy(images_np.astype(np.float32) / 255.0)
            return (mesh_sequence, empty_overlay, "No frames found")
        
        # Debug first frame structure
        if frames and isinstance(frames[0], dict):
            log(f"Frame 0 keys: {list(frames[0].keys())}")
        
        # =====================================================================
        # Extract joints - need to transform to camera space
        # =====================================================================
        joints_list = []
        cam_t_list = []
        focal_lengths = []
        cx_list = []
        cy_list = []
        valid_frame_indices = []
        
        for i, frame in enumerate(frames):
            if not isinstance(frame, dict):
                continue
            
            # Get joint coordinates (in local/canonical space)
            joints = None
            joint_key_used = None
            
            # Try different keys
            for key in ["joint_coords", "joints", "pred_keypoints_3d"]:
                if key in frame and frame[key] is not None:
                    joints = frame[key]
                    joint_key_used = key
                    break
            
            if joints is None:
                continue
            
            if isinstance(joints, torch.Tensor):
                joints = joints.cpu().numpy()
            joints = np.array(joints)
            
            # Get camera translation for transforming to camera space
            cam_t = frame.get("pred_cam_t")
            if cam_t is not None:
                if isinstance(cam_t, torch.Tensor):
                    cam_t = cam_t.cpu().numpy()
                cam_t = np.array(cam_t).flatten()
            
            if i == 0:
                log(f"Using joint key: '{joint_key_used}'")
                log(f"Joint shape: {joints.shape}")
                log(f"pred_cam_t: {cam_t}")
                log(f"Sample joint (local): {joints[0]}")
            
            # Transform joints to camera space: joints_cam = joints + cam_t
            if cam_t is not None and len(cam_t) >= 3:
                joints_camera = joints + cam_t.reshape(1, 3)
            else:
                joints_camera = joints
            
            if i == 0:
                log(f"Sample joint (camera): {joints_camera[0]}")
            
            joints_list.append(joints_camera)
            cam_t_list.append(cam_t)
            focal_lengths.append(frame.get("focal_length"))
            cx_list.append(frame.get("cx", W / 2))
            cy_list.append(frame.get("cy", H / 2))
            valid_frame_indices.append(i)
        
        log(f"Found {len(joints_list)} frames with valid joints")
        
        if len(joints_list) == 0:
            log("ERROR: No joints found in any frame!")
            empty_overlay = torch.from_numpy(images_np.astype(np.float32) / 255.0)
            return (mesh_sequence, empty_overlay, "No joints found")
        
        # Stack joints (now in camera space)
        joints_array = np.stack(joints_list, axis=0)
        original_joints = joints_array.copy()
        N, J, D = joints_array.shape
        
        log(f"Joint array shape: {N} frames, {J} joints, {D}D")
        log(f"Joint range X: [{joints_array[:,:,0].min():.2f}, {joints_array[:,:,0].max():.2f}]")
        log(f"Joint range Y: [{joints_array[:,:,1].min():.2f}, {joints_array[:,:,1].max():.2f}]")
        log(f"Joint range Z: [{joints_array[:,:,2].min():.2f}, {joints_array[:,:,2].max():.2f}]")
        
        # Get joint groups
        feet_indices, body_indices = self._get_joint_groups(skeleton_format, custom_feet_indices, J)
        log(f"Feet indices: {feet_indices}")
        
        # Stabilize
        log("Starting stabilization...")
        stabilized, ground_contact = self._stabilize_joints(
            joints_array, feet_indices, body_indices,
            feet_ema_alpha, feet_hysteresis_px, feet_velocity_damping,
            body_ema_alpha, body_hysteresis_px, body_velocity_damping,
            enable_ground_contact, ground_velocity_threshold, ground_contact_ema,
            enable_savgol, savgol_window, savgol_order
        )
        log("Stabilization complete")
        
        # =====================================================================
        # Update frames - need to convert back to local space
        # =====================================================================
        for i, frame_idx in enumerate(valid_frame_indices):
            frame = frames[frame_idx]
            cam_t = cam_t_list[i]
            
            # Convert stabilized joints back to local space
            if cam_t is not None and len(cam_t) >= 3:
                joints_local = stabilized[i] - cam_t.reshape(1, 3)
            else:
                joints_local = stabilized[i]
            
            # Update the frame
            for key in ["joint_coords", "joints", "pred_keypoints_3d"]:
                if key in frame and frame[key] is not None:
                    frames[frame_idx][key] = joints_local
                    break
        
        # Rebuild output
        output_sequence = deepcopy(mesh_sequence) if isinstance(mesh_sequence, dict) else {}
        if frame_keys is not None:
            output_sequence["frames"] = {k: frames[i] for i, k in enumerate(frame_keys)}
        else:
            if isinstance(mesh_sequence, dict) and "frames" in mesh_sequence:
                output_sequence["frames"] = frames
            else:
                output_sequence = frames
        
        # =====================================================================
        # Render overlay
        # =====================================================================
        log("Rendering debug overlay...")
        overlay = images_np.copy()
        
        # Debug first frame projection
        if len(valid_frame_indices) > 0:
            vi = 0
            focal = focal_lengths[vi]
            cx = cx_list[vi] if cx_list[vi] is not None else W / 2
            cy = cy_list[vi] if cy_list[vi] is not None else H / 2
            
            if focal is None:
                focal = max(W, H)
            elif hasattr(focal, '__len__'):
                focal = float(np.array(focal).flatten()[0])
            else:
                focal = float(focal)
            
            log(f"Frame 0 projection: focal={focal:.1f}, cx={cx:.1f}, cy={cy:.1f}")
            
            test_joint = original_joints[0, 0]
            log(f"Frame 0 joint[0] 3D (camera space): [{test_joint[0]:.4f}, {test_joint[1]:.4f}, {test_joint[2]:.4f}]")
            
            z = max(test_joint[2], 0.1)
            proj_x = test_joint[0] * focal / z + cx
            proj_y = test_joint[1] * focal / z + cy
            log(f"Frame 0 joint[0] 2D: [{proj_x:.1f}, {proj_y:.1f}]")
        
        for vi, frame_idx in enumerate(valid_frame_indices):
            img_idx = frame_idx
            if img_idx >= N_images:
                continue
            
            frame_img = overlay[img_idx]
            
            focal = focal_lengths[vi]
            cx = cx_list[vi] if cx_list[vi] is not None else W / 2
            cy = cy_list[vi] if cy_list[vi] is not None else H / 2
            
            if focal is None:
                focal = max(W, H)
            elif hasattr(focal, '__len__'):
                focal = float(np.array(focal).flatten()[0])
            else:
                focal = float(focal)
            
            orig_3d = original_joints[vi]
            stab_3d = stabilized[vi]
            gc_frame = ground_contact[vi] if ground_contact is not None else None
            
            orig_2d = self._project_joints(orig_3d, focal, cx, cy, W, H)
            stab_2d = self._project_joints(stab_3d, focal, cx, cy, W, H)
            
            # Draw joints
            for j in range(J):
                is_foot = j in feet_indices
                grounded = gc_frame is not None and j < len(gc_frame) and gc_frame[j] > 0.5 if is_foot else False
                
                ox, oy = int(orig_2d[j, 0]), int(orig_2d[j, 1])
                sx, sy = int(stab_2d[j, 0]), int(stab_2d[j, 1])
                
                if show_original:
                    color = COLORS["feet_original"] if is_foot else COLORS["original_joint"]
                    cv2.circle(frame_img, (ox, oy), joint_radius, color, -1)
                    cv2.circle(frame_img, (ox, oy), joint_radius, (0, 0, 0), 1)
                
                if show_stabilized:
                    if grounded and show_ground_contact:
                        color = COLORS["ground_contact"]
                    else:
                        color = COLORS["feet_stabilized"] if is_foot else COLORS["stabilized_joint"]
                    cv2.circle(frame_img, (sx, sy), joint_radius, color, -1)
                    cv2.circle(frame_img, (sx, sy), joint_radius, (0, 0, 0), 1)
                
                if show_original and show_stabilized:
                    disp = np.sqrt((ox - sx)**2 + (oy - sy)**2)
                    if disp > 2:
                        cv2.line(frame_img, (ox, oy), (sx, sy), (255, 255, 255), 1)
            
            # Legend
            cv2.rectangle(frame_img, (5, 5), (150, 90), (0, 0, 0), -1)
            y = 25
            if show_original:
                cv2.circle(frame_img, (20, y), 8, COLORS["original_joint"], -1)
                cv2.putText(frame_img, "Original", (35, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text_fg"], 1)
                y += 22
            if show_stabilized:
                cv2.circle(frame_img, (20, y), 8, COLORS["stabilized_joint"], -1)
                cv2.putText(frame_img, "Stabilized", (35, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text_fg"], 1)
                y += 22
            if show_ground_contact:
                cv2.circle(frame_img, (20, y), 8, COLORS["ground_contact"], -1)
                cv2.putText(frame_img, "Grounded", (35, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text_fg"], 1)
            
            cv2.putText(frame_img, f"F{img_idx}", (W - 60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["text_fg"], 2)
        
        log(f"Overlay rendered for {len(valid_frame_indices)} frames")
        
        overlay_tensor = torch.from_numpy(overlay.astype(np.float32) / 255.0)
        
        # Stats
        displacement = np.linalg.norm(stabilized - original_joints, axis=-1)
        feet_disp = displacement[:, feet_indices].mean() if feet_indices else 0
        body_disp = displacement[:, body_indices].mean() if body_indices else 0
        gc_frames = int((ground_contact[:, feet_indices] > 0.5).any(axis=1).sum()) if feet_indices else 0
        
        debug_info = (
            f"=== Joint Temporal Stabilizer ===\n"
            f"Frames: {N}, Joints: {J}\n"
            f"Feet indices: {feet_indices}\n"
            f"Feet: EMA={feet_ema_alpha}, hyst={feet_hysteresis_px}, damp={feet_velocity_damping}\n"
            f"Body: EMA={body_ema_alpha}, hyst={body_hysteresis_px}, damp={body_velocity_damping}\n"
            f"Ground contact: {'ON' if enable_ground_contact else 'OFF'}, frames={gc_frames}\n"
            f"Mean disp - feet: {feet_disp:.4f}, body: {body_disp:.4f}"
        )
        
        log(f"DONE! feet_disp={feet_disp:.4f}, body_disp={body_disp:.4f}")
        log("=" * 50)
        
        return (output_sequence, overlay_tensor, debug_info)

    def _project_joints(self, joints_3d, focal, cx, cy, W, H):
        z = joints_3d[:, 2:3]
        z = np.clip(z, 0.1, None)
        x = joints_3d[:, 0] * focal / z.flatten() + cx
        y = joints_3d[:, 1] * focal / z.flatten() + cy
        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)
        return np.stack([x, y], axis=-1)

    def _get_joint_groups(self, skeleton_format, custom_feet, num_joints):
        if custom_feet.strip():
            feet = [int(i.strip()) for i in custom_feet.split(",") if i.strip().isdigit()]
            feet = [i for i in feet if i < num_joints]
        elif skeleton_format in SKELETON_CONFIGS and skeleton_format != "AUTO":
            feet = [i for i in SKELETON_CONFIGS[skeleton_format]["feet_indices"] if i < num_joints]
        else:
            if num_joints >= 100:
                feet = [14, 15, 16, 17, 18, 19]
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
