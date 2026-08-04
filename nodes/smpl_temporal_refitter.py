"""
SMPL Temporal Refitter for SAM3DBody2abc
=========================================
Version: 2.0.0

Applies temporal smoothing to mesh_sequence using tracked 2D keypoints.
Works even without SMPL model - directly smooths 3D joints.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import copy


def log(msg):
    print(f"[SMPLRefitter] {msg}", flush=True)


class SMPLTemporalRefitter:
    """
    Apply temporal smoothing to mesh sequence using tracked 2D keypoints.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE",),
                "tracked_keypoints_2d": ("KEYPOINTS_2D",),
                "images": ("IMAGE",),
            },
            "optional": {
                "smooth_factor": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Smoothing factor (0=no smoothing, 1=maximum smoothing)"
                }),
                "keypoint_blend": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Blend between original (0) and tracked (1) keypoints"
                }),
                "window_size": ("INT", {
                    "default": 7,
                    "min": 1,
                    "max": 15,
                    "step": 2,
                    "tooltip": "Smoothing window size (odd number)"
                }),
            },
        }

    RETURN_TYPES = ("MESH_SEQUENCE", "IMAGE", "STRING")
    RETURN_NAMES = ("refined_mesh_sequence", "debug_video", "status")
    FUNCTION = "process"
    CATEGORY = "SAM3DBody2abc/Refinement"

    def process(
        self,
        mesh_sequence: Dict,
        tracked_keypoints_2d: Dict,
        images: torch.Tensor,
        smooth_factor: float = 0.85,
        keypoint_blend: float = 0.7,
        window_size: int = 7,
    ) -> Tuple[Dict, torch.Tensor, str]:
        
        log("=" * 60)
        log("SMPL Temporal Refitter v2.0")
        log("=" * 60)
        
        # Extract data
        frames = mesh_sequence.get("frames", {})
        
        # Get tracked keypoints
        tracked_kp = None
        log(f"tracked_keypoints_2d type: {type(tracked_keypoints_2d)}")
        
        if isinstance(tracked_keypoints_2d, dict):
            tracked_kp = tracked_keypoints_2d.get("keypoints")
            if tracked_kp is not None:
                log(f"Tracked keypoints shape: {tracked_kp.shape}")
        
        if not frames:
            log("ERROR: No frames in mesh_sequence")
            empty_debug = torch.zeros(1, 64, 64, 3)
            return (mesh_sequence, empty_debug, "Error: No frames")
        
        frame_indices = sorted(frames.keys())
        num_frames = len(frame_indices)
        log(f"Processing {num_frames} frames")
        log(f"Settings: smooth={smooth_factor}, blend={keypoint_blend}, window={window_size}")
        
        # Create refined mesh sequence
        refined_sequence = copy.deepcopy(mesh_sequence)
        refined_frames = refined_sequence.get("frames", {})
        
        # =====================================================================
        # Step 1: Collect all 3D joints into array for batch smoothing
        # =====================================================================
        log("Step 1: Collecting 3D joints...")
        
        all_joints = []
        all_cam_t = []
        valid_frames = []
        
        for frame_idx in frame_indices:
            frame_data = frames[frame_idx]
            joints = frame_data.get("joint_coords")
            cam_t = frame_data.get("pred_cam_t")
            
            if joints is not None:
                if isinstance(joints, torch.Tensor):
                    joints = joints.cpu().numpy()
                all_joints.append(joints)
                
                if cam_t is not None:
                    if isinstance(cam_t, torch.Tensor):
                        cam_t = cam_t.cpu().numpy()
                    all_cam_t.append(cam_t)
                else:
                    # Use default camera translation
                    all_cam_t.append(np.array([0.0, 0.0, 5.0], dtype=np.float32))
                
                valid_frames.append(frame_idx)
        
        if len(all_joints) == 0:
            log("ERROR: No valid joints found")
            empty_debug = torch.zeros(1, 64, 64, 3)
            return (mesh_sequence, empty_debug, "Error: No joints found")
        
        joints_array = np.stack(all_joints, axis=0)  # (N, J, 3)
        cam_t_array = np.stack(all_cam_t, axis=0)    # (N, 3)
        
        log(f"Joints array shape: {joints_array.shape}")
        log(f"Camera translation shape: {cam_t_array.shape}")
        
        # =====================================================================
        # Step 2: Apply temporal smoothing to joints
        # =====================================================================
        log("Step 2: Applying temporal smoothing...")
        
        smoothed_joints = self._temporal_smooth(joints_array, window_size, smooth_factor)
        smoothed_cam_t = self._temporal_smooth(cam_t_array, window_size, smooth_factor)
        
        # =====================================================================
        # Step 3: Blend with tracked 2D keypoints (if available)
        # =====================================================================
        if tracked_kp is not None and keypoint_blend > 0:
            log("Step 3: Blending with tracked 2D keypoints...")
            
            for i, frame_idx in enumerate(valid_frames):
                if i >= len(tracked_kp):
                    continue
                
                frame_data = refined_frames[frame_idx]
                target_2d = tracked_kp[i]  # (K, 2)
                
                # Get camera params
                focal = frame_data.get("focal_length") or 2000.0
                img_size = frame_data.get("image_size", (1920, 1080))
                cx = frame_data.get("cx") or img_size[0] / 2
                cy = frame_data.get("cy") or img_size[1] / 2
                
                # Project smoothed 3D joints to 2D
                joints_3d = smoothed_joints[i]
                cam_t = smoothed_cam_t[i]
                
                # Adjust joints to better match tracked 2D
                adjusted_joints = self._adjust_to_2d(
                    joints_3d, cam_t, target_2d, 
                    focal, cx, cy, keypoint_blend
                )
                
                smoothed_joints[i] = adjusted_joints
        else:
            log("Step 3: Skipping 2D blend (no tracked keypoints)")
        
        # =====================================================================
        # Step 4: Update refined mesh sequence
        # =====================================================================
        log("Step 4: Updating mesh sequence...")
        
        for i, frame_idx in enumerate(valid_frames):
            refined_frames[frame_idx]["joint_coords"] = smoothed_joints[i]
            refined_frames[frame_idx]["pred_cam_t"] = smoothed_cam_t[i]
            refined_frames[frame_idx]["refined"] = True
        
        # =====================================================================
        # Step 5: Create debug video
        # =====================================================================
        log("Step 5: Creating debug video...")
        
        debug_video = self._create_debug_video(
            images, refined_frames, tracked_kp, frame_indices, valid_frames
        )
        
        status = f"Refined {len(valid_frames)} frames with smooth={smooth_factor}"
        log(status)
        log("=" * 60)
        
        return (refined_sequence, debug_video, status)

    def _temporal_smooth(
        self, 
        data: np.ndarray, 
        window_size: int, 
        smooth_factor: float
    ) -> np.ndarray:
        """Apply temporal smoothing using moving average."""
        
        if smooth_factor <= 0:
            return data.copy()
        
        # Ensure odd window size
        window_size = window_size | 1
        
        # Pad data
        pad_size = window_size // 2
        if data.ndim == 2:  # (N, 3) for camera
            padded = np.pad(data, ((pad_size, pad_size), (0, 0)), mode='edge')
        else:  # (N, J, 3) for joints
            padded = np.pad(data, ((pad_size, pad_size), (0, 0), (0, 0)), mode='edge')
        
        # Moving average
        smoothed = np.zeros_like(data)
        for i in range(len(data)):
            window = padded[i:i + window_size]
            smoothed[i] = window.mean(axis=0)
        
        # Blend original and smoothed
        result = (1 - smooth_factor) * data + smooth_factor * smoothed
        
        return result

    def _adjust_to_2d(
        self,
        joints_3d: np.ndarray,
        cam_t: np.ndarray,
        target_2d: np.ndarray,
        focal: float,
        cx: float,
        cy: float,
        blend: float,
    ) -> np.ndarray:
        """Adjust 3D joints to better match target 2D keypoints."""
        
        # Project current 3D to 2D
        joints_cam = joints_3d + cam_t.reshape(1, 3)
        z = np.maximum(joints_cam[:, 2:3], 0.1)
        proj_x = joints_cam[:, 0:1] * focal / z + cx
        proj_y = joints_cam[:, 1:2] * focal / z + cy
        current_2d = np.concatenate([proj_x, proj_y], axis=1)
        
        # Compute 2D error
        num_match = min(len(current_2d), len(target_2d))
        diff_2d = target_2d[:num_match] - current_2d[:num_match]
        
        # Convert 2D error to 3D adjustment
        # This is approximate - we adjust x,y in camera space
        dx_3d = diff_2d[:, 0:1] * z[:num_match] / focal
        dy_3d = diff_2d[:, 1:2] * z[:num_match] / focal
        
        # Apply adjustment with blend factor
        adjusted = joints_3d.copy()
        adjusted[:num_match, 0:1] += blend * dx_3d
        adjusted[:num_match, 1:2] += blend * dy_3d
        
        return adjusted

    def _create_debug_video(
        self,
        images: torch.Tensor,
        refined_frames: Dict,
        tracked_kp: Optional[np.ndarray],
        frame_indices: List[int],
        valid_frames: List[int],
    ) -> torch.Tensor:
        """Create side-by-side debug video: Left=TAPIR tracked 2D, Right=Refined 3D projected."""
        import cv2
        
        num_frames = len(images)
        debug_frames = []
        
        # Colors (BGR)
        COLOR_TRACKED = (0, 255, 0)    # Green - TAPIR tracked 2D
        COLOR_REFINED = (255, 255, 0)  # Teal/Cyan - Refined 3D projected
        COLOR_ORIGINAL = (0, 0, 255)   # Red - Original per-frame 2D (for comparison)
        
        for idx in range(num_frames):
            # Get frame
            frame = images[idx].cpu().numpy()
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
            
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img_h, img_w = frame_bgr.shape[:2]
            
            # Create two copies
            left_frame = frame_bgr.copy()
            right_frame = frame_bgr.copy()
            
            # Get frame data
            frame_idx = frame_indices[idx] if idx < len(frame_indices) else idx
            frame_data = refined_frames.get(frame_idx, {})
            
            # === LEFT PANE: TAPIR Tracked 2D keypoints (Green) ===
            # These are already in pixel coordinates from TAPIR
            if tracked_kp is not None and idx < len(tracked_kp):
                for kp in tracked_kp[idx]:
                    x, y = int(kp[0]), int(kp[1])
                    if 0 <= x < img_w and 0 <= y < img_h:
                        cv2.circle(left_frame, (x, y), 5, COLOR_TRACKED, -1)
            
            # === RIGHT PANE: Refined/Smoothed joints ===
            # Option 1: Use pred_keypoints_2d if available (already in pixel coords)
            # Option 2: Project smoothed 3D joints using camera params
            
            # First try to use the original pred_keypoints_2d as baseline
            # Then overlay the projected smoothed 3D joints
            
            # Draw original 2D keypoints (small red dots) for reference
            original_2d = frame_data.get("pred_keypoints_2d")
            if original_2d is not None:
                if isinstance(original_2d, torch.Tensor):
                    original_2d = original_2d.cpu().numpy()
                original_2d = np.array(original_2d)
                for i, kp in enumerate(original_2d[:70]):  # First 70 keypoints
                    x, y = int(kp[0]), int(kp[1])
                    if 0 <= x < img_w and 0 <= y < img_h:
                        cv2.circle(right_frame, (x, y), 2, COLOR_ORIGINAL, -1)
            
            # Project smoothed 3D joints (teal circles)
            joints_3d = frame_data.get("joint_coords")
            cam_t = frame_data.get("pred_cam_t")
            
            if joints_3d is not None:
                if isinstance(joints_3d, torch.Tensor):
                    joints_3d = joints_3d.cpu().numpy()
                
                # Get camera params with proper defaults
                focal = frame_data.get("focal_length")
                if focal is None or focal == 0:
                    focal = frame_data.get("focal_length_sam3d", 2000.0)
                if focal is None or focal == 0:
                    focal = 2000.0
                    
                cx = frame_data.get("cx")
                cy = frame_data.get("cy")
                if cx is None:
                    cx = img_w / 2
                if cy is None:
                    cy = img_h / 2
                
                if cam_t is not None:
                    if isinstance(cam_t, torch.Tensor):
                        cam_t = cam_t.cpu().numpy()
                    
                    # Project: joints in local space + cam_t = camera space
                    joints_cam = joints_3d + cam_t.reshape(1, 3)
                    z = np.maximum(joints_cam[:, 2], 0.1)
                    proj_x = (joints_cam[:, 0] * focal / z + cx).astype(int)
                    proj_y = (joints_cam[:, 1] * focal / z + cy).astype(int)
                    
                    # Draw projected joints (teal)
                    for i in range(min(70, len(proj_x))):
                        x, y = proj_x[i], proj_y[i]
                        if 0 <= x < img_w and 0 <= y < img_h:
                            cv2.circle(right_frame, (x, y), 5, COLOR_REFINED, -1)
                else:
                    # No camera translation - use pred_keypoints_2d directly if available
                    log(f"Frame {idx}: No cam_t, using pred_keypoints_2d") if idx == 0 else None
            
            # Add labels
            cv2.putText(left_frame, "TAPIR Tracked 2D (Green)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TRACKED, 2)
            cv2.putText(left_frame, f"Frame {idx}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(right_frame, "Smoothed 3D Proj (Teal)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_REFINED, 2)
            cv2.putText(right_frame, "Original 2D (Red dots)", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ORIGINAL, 1)
            cv2.putText(right_frame, f"Frame {idx}", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Combine side by side with divider
            cv2.line(left_frame, (img_w - 2, 0), (img_w - 2, img_h), (255, 255, 255), 2)
            combined = np.concatenate([left_frame, right_frame], axis=1)
            
            frame_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            debug_frames.append(frame_rgb)
        
        debug_video = np.stack(debug_frames, axis=0)
        debug_video = torch.from_numpy(debug_video).float() / 255.0
        
        log(f"Debug video shape: {debug_video.shape} (side-by-side)")
        return debug_video


NODE_CLASS_MAPPINGS = {
    "SAM3DBody2abc_SMPLTemporalRefitter": SMPLTemporalRefitter
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBody2abc_SMPLTemporalRefitter": "🔄 SMPL Temporal Refitter"
}
