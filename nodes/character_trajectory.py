"""
Character Trajectory Tracker for SAM3DBody2abc

Combines TAPIR point tracking with Depth Anything V2 for accurate 3D character trajectory.

Pipeline:
1. TAPIR tracks 2D points on the character (using SAM3 mask)
2. Depth Anything V2 provides per-frame depth maps
3. This node fuses 2D tracks + depth = 3D trajectory

This solves the problem where SAM3DBody's pred_cam_t depth estimation is inaccurate
for large depth changes (character moving toward/away from camera).

Author: Claude (Anthropic)
Version: 1.0.0
License: Apache 2.0
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
import cv2
from scipy import ndimage
from scipy.signal import savgol_filter

# Try to import TAPIR
TAPIR_AVAILABLE = False
try:
    from tapnet.torch import tapir_model
    TAPIR_AVAILABLE = True
    print("[CharacterTracker] TAPIR (tapnet) available")
except ImportError:
    print("[CharacterTracker] TAPIR not found - will use external TAPIR tracks or optical flow fallback")


class CharacterTrajectoryTracker:
    """
    Track character in 3D space using TAPIR (2D) + Depth Anything V2 (depth).
    
    This provides accurate 3D trajectory even when character moves toward/away
    from camera, which SAM3DBody's weak perspective model struggles with.
    """
    
    TRACKING_MODES = ["Pelvis (Center)", "Feet (Ground Contact)", "Full Body (Average)", "Custom Points"]
    DEPTH_MODES = ["Relative (Normalized)", "Metric (Estimated)", "Reference Scale (SAM3DBody)"]
    SMOOTHING_METHODS = ["None", "Gaussian", "Savitzky-Golay", "Moving Average"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input video frames"
                }),
                "character_mask": ("MASK", {
                    "tooltip": "Character mask from SAM3 segmentation"
                }),
                "depth_maps": ("IMAGE", {
                    "tooltip": "Depth maps from Depth Anything V2"
                }),
            },
            "optional": {
                # === From SAM3DBody ===
                "mesh_sequence": ("MESH_SEQUENCE", {
                    "tooltip": "SAM3DBody output - provides reference scale and pose data"
                }),
                
                # === Depth Settings ===
                "depth_is_inverted": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "True if CLOSER objects are BRIGHTER (white) in depth map"
                }),
                "reference_depth_m": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.5,
                    "max": 50.0,
                    "step": 0.1,
                    "tooltip": "Reference depth in meters for frame 0 (used to scale relative depth)"
                }),
                
                # === Tracking Settings ===
                "tracking_mode": (cls.TRACKING_MODES, {
                    "default": "Pelvis (Center)",
                    "tooltip": "Which body part(s) to track for trajectory"
                }),
                "num_track_points": ("INT", {
                    "default": 16,
                    "min": 4,
                    "max": 64,
                    "tooltip": "Number of points to track on character (for internal TAPIR)"
                }),
                
                # === Smoothing ===
                "smoothing_method": (cls.SMOOTHING_METHODS, {
                    "default": "Savitzky-Golay",
                    "tooltip": "Temporal smoothing for trajectory"
                }),
                "smoothing_window": ("INT", {
                    "default": 5,
                    "min": 3,
                    "max": 21,
                    "step": 2,
                    "tooltip": "Smoothing window size (odd number)"
                }),
                
                # === Output ===
                "output_debug_video": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Output visualization of tracked points and depth"
                }),
            }
        }
    
    RETURN_TYPES = ("TRAJECTORY_3D", "MESH_SEQUENCE", "IMAGE", "STRING")
    RETURN_NAMES = ("trajectory_3d", "mesh_sequence_updated", "debug_video", "info")
    FUNCTION = "track_character"
    CATEGORY = "SAM3DBody2abc/Tracking"
    
    def __init__(self):
        self.tapir_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_tapir(self) -> bool:
        """Load TAPIR model if available."""
        if not TAPIR_AVAILABLE:
            return False
            
        if self.tapir_model is not None:
            return True
        
        try:
            import os
            checkpoint_paths = [
                "/workspace/ComfyUI/models/tapir/bootstapir_checkpoint_v2.pt",
                "/workspace/models/tapir/bootstapir_checkpoint_v2.pt",
                os.path.expanduser("~/models/tapir/bootstapir_checkpoint_v2.pt"),
                "bootstapir_checkpoint_v2.pt",
            ]
            
            checkpoint_path = None
            for path in checkpoint_paths:
                if os.path.exists(path):
                    checkpoint_path = path
                    break
            
            if checkpoint_path is None:
                print("[CharacterTracker] TAPIR checkpoint not found")
                return False
            
            self.tapir_model = tapir_model.TAPIR(pyramid_level=1)
            self.tapir_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.tapir_model.to(self.device)
            self.tapir_model.eval()
            
            print(f"[CharacterTracker] TAPIR loaded from {checkpoint_path}")
            return True
            
        except Exception as e:
            print(f"[CharacterTracker] Error loading TAPIR: {e}")
            return False
    
    def _get_mask_centroid(self, mask: np.ndarray, tracking_mode: str) -> Tuple[float, float]:
        """
        Get centroid of mask region based on tracking mode.
        
        Args:
            mask: Binary mask [H, W]
            tracking_mode: Which body part to track
            
        Returns:
            (x, y) centroid coordinates
        """
        H, W = mask.shape
        
        coords = np.where(mask > 0.5)
        if len(coords[0]) == 0:
            return W / 2, H / 2
        
        y_coords, x_coords = coords
        y_min, y_max = y_coords.min(), y_coords.max()
        mask_height = y_max - y_min
        
        if tracking_mode == "Pelvis (Center)":
            # Focus on middle region (pelvis area)
            pelvis_y_min = y_min + int(mask_height * 0.35)
            pelvis_y_max = y_min + int(mask_height * 0.55)
            
            pelvis_mask = (y_coords >= pelvis_y_min) & (y_coords <= pelvis_y_max)
            if pelvis_mask.sum() > 0:
                x_coords = x_coords[pelvis_mask]
                y_coords = y_coords[pelvis_mask]
        
        elif tracking_mode == "Feet (Ground Contact)":
            # Focus on bottom region (feet)
            feet_y_min = y_min + int(mask_height * 0.85)
            
            feet_mask = y_coords >= feet_y_min
            if feet_mask.sum() > 0:
                x_coords = x_coords[feet_mask]
                y_coords = y_coords[feet_mask]
        
        # Return centroid
        return float(x_coords.mean()), float(y_coords.mean())
    
    def _sample_depth_at_mask(self, depth_map: np.ndarray, mask: np.ndarray, 
                               tracking_mode: str) -> float:
        """
        Sample depth value within the mask region.
        
        Args:
            depth_map: Depth map [H, W] (normalized 0-1)
            mask: Binary mask [H, W]
            tracking_mode: Which body part to sample
            
        Returns:
            Average depth value in the region
        """
        H, W = mask.shape
        
        # Resize depth map if needed
        if depth_map.shape[:2] != (H, W):
            depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
        coords = np.where(mask > 0.5)
        if len(coords[0]) == 0:
            return 0.5  # Default mid-depth
        
        y_coords, x_coords = coords
        y_min, y_max = y_coords.min(), y_coords.max()
        mask_height = y_max - y_min
        
        # Create region mask based on tracking mode
        if tracking_mode == "Pelvis (Center)":
            pelvis_y_min = y_min + int(mask_height * 0.35)
            pelvis_y_max = y_min + int(mask_height * 0.55)
            region_mask = mask.copy()
            region_mask[:pelvis_y_min, :] = 0
            region_mask[pelvis_y_max:, :] = 0
        elif tracking_mode == "Feet (Ground Contact)":
            feet_y_min = y_min + int(mask_height * 0.85)
            region_mask = mask.copy()
            region_mask[:feet_y_min, :] = 0
        else:
            region_mask = mask
        
        # Sample depth in region
        region_coords = np.where(region_mask > 0.5)
        if len(region_coords[0]) == 0:
            region_coords = coords
        
        depth_values = depth_map[region_coords]
        
        # Use median for robustness
        return float(np.median(depth_values))
    
    def _smooth_trajectory(self, trajectory: np.ndarray, method: str, window: int) -> np.ndarray:
        """
        Apply temporal smoothing to trajectory.
        
        Args:
            trajectory: [N, 3] array of (x, y, z) positions
            method: Smoothing method
            window: Window size
            
        Returns:
            Smoothed trajectory
        """
        if method == "None" or len(trajectory) < window:
            return trajectory
        
        smoothed = trajectory.copy()
        
        # Ensure odd window
        window = window if window % 2 == 1 else window + 1
        
        if method == "Gaussian":
            for i in range(3):
                smoothed[:, i] = ndimage.gaussian_filter1d(trajectory[:, i], sigma=window/4)
        
        elif method == "Savitzky-Golay":
            poly_order = min(3, window - 1)
            for i in range(3):
                smoothed[:, i] = savgol_filter(trajectory[:, i], window, poly_order)
        
        elif method == "Moving Average":
            kernel = np.ones(window) / window
            for i in range(3):
                smoothed[:, i] = np.convolve(trajectory[:, i], kernel, mode='same')
        
        return smoothed
    
    def _track_with_optical_flow(self, images: np.ndarray, masks: np.ndarray, 
                                  tracking_mode: str) -> np.ndarray:
        """
        Fallback tracking using optical flow when TAPIR not available.
        
        Args:
            images: [N, H, W, 3] video frames
            masks: [N, H, W] character masks
            tracking_mode: Tracking mode
            
        Returns:
            [N, 2] array of (x, y) tracked positions
        """
        N, H, W = masks.shape[:3]
        tracks = np.zeros((N, 2))
        
        # Initialize with first frame centroid
        tracks[0] = self._get_mask_centroid(masks[0], tracking_mode)
        
        # Track using Lucas-Kanade optical flow
        prev_gray = cv2.cvtColor((images[0] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        prev_pt = np.array([[tracks[0]]], dtype=np.float32)
        
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        for i in range(1, N):
            curr_gray = cv2.cvtColor((images[i] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Track point
            next_pt, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pt, None, **lk_params
            )
            
            if status[0][0] == 1:
                tracks[i] = next_pt[0][0]
            else:
                # Lost track - reinitialize from mask
                tracks[i] = self._get_mask_centroid(masks[i], tracking_mode)
            
            # Validate against mask - if point is outside mask, reinitialize
            x, y = int(tracks[i, 0]), int(tracks[i, 1])
            if 0 <= x < W and 0 <= y < H:
                if masks[i, y, x] < 0.5:
                    tracks[i] = self._get_mask_centroid(masks[i], tracking_mode)
            
            prev_gray = curr_gray
            prev_pt = np.array([[tracks[i]]], dtype=np.float32)
        
        return tracks
    
    def _track_with_tapir(self, images: np.ndarray, masks: np.ndarray,
                          tracking_mode: str, num_points: int) -> np.ndarray:
        """
        Track character using TAPIR.
        
        Args:
            images: [N, H, W, 3] video frames
            masks: [N, H, W] character masks
            tracking_mode: Tracking mode
            num_points: Number of points to track
            
        Returns:
            [N, 2] array of (x, y) averaged tracked positions
        """
        N, H, W, _ = images.shape
        
        # Get initial points from first frame mask
        init_points = self._sample_points_from_mask(masks[0], tracking_mode, num_points)
        
        # Prepare video for TAPIR: [B, T, H, W, 3], range [-1, 1]
        video = torch.from_numpy(images).float().permute(0, 1, 2, 3)  # Keep as [T, H, W, C]
        video = video.unsqueeze(0)  # [1, T, H, W, C]
        video = (video / 255.0) * 2 - 1  # Normalize to [-1, 1] if not already
        video = video.to(self.device)
        
        # Query points: [B, N, 3] where each point is [t, y, x]
        query_points = torch.zeros((1, len(init_points), 3), dtype=torch.float32, device=self.device)
        query_points[0, :, 0] = 0  # All points from frame 0
        query_points[0, :, 1] = torch.from_numpy(init_points[:, 1])  # y
        query_points[0, :, 2] = torch.from_numpy(init_points[:, 0])  # x
        
        # Run TAPIR
        with torch.no_grad():
            outputs = self.tapir_model(video, query_points)
        
        # outputs['tracks']: [B, N, T, 2] - (x, y) positions
        # outputs['occlusion']: [B, N, T] - occlusion logits
        tracks = outputs['tracks'][0].cpu().numpy()  # [N, T, 2]
        occlusion = torch.sigmoid(outputs['occlusion'][0]).cpu().numpy()  # [N, T]
        
        # Average visible points per frame
        result = np.zeros((N, 2))
        for t in range(N):
            visible_mask = occlusion[:, t] < 0.5  # Not occluded
            if visible_mask.sum() > 0:
                result[t] = tracks[visible_mask, t, :].mean(axis=0)
            else:
                result[t] = tracks[:, t, :].mean(axis=0)
        
        return result
    
    def _sample_points_from_mask(self, mask: np.ndarray, tracking_mode: str, 
                                  num_points: int) -> np.ndarray:
        """Sample tracking points from mask region."""
        H, W = mask.shape
        
        coords = np.where(mask > 0.5)
        if len(coords[0]) == 0:
            # Return center point
            return np.array([[W // 2, H // 2]])
        
        y_coords, x_coords = coords
        y_min, y_max = y_coords.min(), y_coords.max()
        mask_height = y_max - y_min
        
        # Filter by tracking mode
        if tracking_mode == "Pelvis (Center)":
            pelvis_y_min = y_min + int(mask_height * 0.35)
            pelvis_y_max = y_min + int(mask_height * 0.55)
            valid = (y_coords >= pelvis_y_min) & (y_coords <= pelvis_y_max)
        elif tracking_mode == "Feet (Ground Contact)":
            feet_y_min = y_min + int(mask_height * 0.85)
            valid = y_coords >= feet_y_min
        else:
            valid = np.ones(len(y_coords), dtype=bool)
        
        if valid.sum() < num_points:
            valid = np.ones(len(y_coords), dtype=bool)
        
        valid_x = x_coords[valid]
        valid_y = y_coords[valid]
        
        # Sample points
        indices = np.random.choice(len(valid_x), min(num_points, len(valid_x)), replace=False)
        
        points = np.stack([valid_x[indices], valid_y[indices]], axis=1)
        return points.astype(np.float32)
    
    def _create_debug_video(self, images: np.ndarray, masks: np.ndarray,
                            tracks_2d: np.ndarray, depths: np.ndarray,
                            trajectory_3d: np.ndarray) -> np.ndarray:
        """Create debug visualization video."""
        N, H, W, C = images.shape
        debug_frames = []
        
        for i in range(N):
            frame = (images[i] * 255).astype(np.uint8).copy()
            
            # Draw mask outline
            mask_uint8 = (masks[i] * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
            
            # Draw tracked point
            x, y = int(tracks_2d[i, 0]), int(tracks_2d[i, 1])
            cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)
            cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)
            
            # Draw trajectory trail (last 10 frames)
            for j in range(max(0, i - 10), i):
                x1, y1 = int(tracks_2d[j, 0]), int(tracks_2d[j, 1])
                x2, y2 = int(tracks_2d[j + 1, 0]), int(tracks_2d[j + 1, 1])
                alpha = (j - max(0, i - 10)) / 10
                color = (int(255 * alpha), int(100 * alpha), int(255 * (1 - alpha)))
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add text overlay
            depth_val = depths[i]
            pos_3d = trajectory_3d[i]
            
            cv2.putText(frame, f"Frame: {i}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Depth: {depth_val:.2f}m", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"3D: ({pos_3d[0]:.2f}, {pos_3d[1]:.2f}, {pos_3d[2]:.2f})", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            debug_frames.append(frame)
        
        return np.stack(debug_frames).astype(np.float32) / 255.0
    
    def track_character(self, images, character_mask, depth_maps,
                        mesh_sequence=None,
                        depth_is_inverted=False,
                        reference_depth_m=5.0,
                        tracking_mode="Pelvis (Center)",
                        num_track_points=16,
                        smoothing_method="Savitzky-Golay",
                        smoothing_window=5,
                        output_debug_video=True):
        """
        Main tracking function.
        
        Combines 2D tracking (TAPIR or optical flow) with depth maps to produce
        accurate 3D trajectory.
        """
        print(f"[CharacterTracker] Starting character trajectory tracking...")
        
        # Convert inputs to numpy
        if isinstance(images, torch.Tensor):
            images_np = images.cpu().numpy()
        else:
            images_np = np.array(images)
        
        if isinstance(character_mask, torch.Tensor):
            masks_np = character_mask.cpu().numpy()
        else:
            masks_np = np.array(character_mask)
        
        if isinstance(depth_maps, torch.Tensor):
            depth_np = depth_maps.cpu().numpy()
        else:
            depth_np = np.array(depth_maps)
        
        # Handle dimensions
        if len(images_np.shape) == 3:
            images_np = images_np[np.newaxis, ...]
        if len(masks_np.shape) == 2:
            masks_np = masks_np[np.newaxis, ...]
        if len(depth_np.shape) == 3:
            depth_np = depth_np[np.newaxis, ...]
        
        N = len(images_np)
        H, W = images_np.shape[1:3]
        
        print(f"[CharacterTracker] Processing {N} frames at {W}x{H}")
        print(f"[CharacterTracker] Tracking mode: {tracking_mode}")
        
        # === Step 1: Get reference depth from SAM3DBody or use provided ===
        if mesh_sequence is not None:
            frames = mesh_sequence.get("frames", [])
            if frames and "pred_cam_t" in frames[0]:
                ref_depth = abs(frames[0]["pred_cam_t"][2])
                print(f"[CharacterTracker] Reference depth from SAM3DBody: {ref_depth:.2f}m")
            else:
                ref_depth = reference_depth_m
                print(f"[CharacterTracker] Using provided reference depth: {ref_depth:.2f}m")
        else:
            ref_depth = reference_depth_m
            print(f"[CharacterTracker] Using provided reference depth: {ref_depth:.2f}m")
        
        # === Step 2: Track 2D position ===
        print(f"[CharacterTracker] Tracking 2D position...")
        
        tapir_loaded = self._load_tapir()
        
        if tapir_loaded:
            print(f"[CharacterTracker] Using TAPIR for tracking")
            tracks_2d = self._track_with_tapir(images_np, masks_np, tracking_mode, num_track_points)
        else:
            print(f"[CharacterTracker] Using optical flow fallback")
            tracks_2d = self._track_with_optical_flow(images_np, masks_np, tracking_mode)
        
        # === Step 3: Sample depth at tracked positions ===
        print(f"[CharacterTracker] Sampling depth values...")
        
        depths_raw = np.zeros(N)
        for i in range(N):
            # Get depth map for this frame (handle single channel or RGB)
            if len(depth_np.shape) == 4 and depth_np.shape[-1] == 3:
                depth_frame = depth_np[i, :, :, 0]  # Use first channel
            elif len(depth_np.shape) == 4:
                depth_frame = depth_np[i, :, :, 0]
            else:
                depth_frame = depth_np[i]
            
            # Sample depth at mask region
            if i < len(masks_np):
                depths_raw[i] = self._sample_depth_at_mask(depth_frame, masks_np[i], tracking_mode)
            else:
                depths_raw[i] = self._sample_depth_at_mask(depth_frame, masks_np[-1], tracking_mode)
        
        # Invert if needed (so that larger value = farther)
        if depth_is_inverted:
            depths_raw = 1.0 - depths_raw
        
        # === Step 4: Convert relative depth to metric depth ===
        # Use first frame as reference
        ref_depth_value = depths_raw[0] if depths_raw[0] > 0 else 0.5
        
        # Scale so that frame 0 depth = ref_depth meters
        # depth_metric = ref_depth * (depth_value / ref_depth_value)
        depths_metric = ref_depth * (depths_raw / ref_depth_value)
        
        print(f"[CharacterTracker] Depth range: {depths_metric.min():.2f}m to {depths_metric.max():.2f}m")
        
        # === Step 5: Convert 2D + depth to 3D ===
        # X, Y in screen space, Z from depth
        # Normalize X, Y to be relative to image center
        cx, cy = W / 2, H / 2
        
        trajectory_3d = np.zeros((N, 3))
        for i in range(N):
            # Normalized screen position (-1 to 1 range approximately)
            # Scale by depth to get world position
            screen_x = (tracks_2d[i, 0] - cx) / W  # Normalized X
            screen_y = (tracks_2d[i, 1] - cy) / H  # Normalized Y
            
            # World position (using similar triangles)
            # Assuming a reference focal length
            focal_factor = 1.5  # Approximate focal length factor
            
            trajectory_3d[i, 0] = screen_x * depths_metric[i] * focal_factor  # X
            trajectory_3d[i, 1] = -screen_y * depths_metric[i] * focal_factor  # Y (flip for Y-up)
            trajectory_3d[i, 2] = depths_metric[i]  # Z (depth)
        
        # === Step 6: Apply smoothing ===
        if smoothing_method != "None":
            print(f"[CharacterTracker] Applying {smoothing_method} smoothing (window={smoothing_window})")
            trajectory_3d = self._smooth_trajectory(trajectory_3d, smoothing_method, smoothing_window)
            depths_metric = self._smooth_trajectory(depths_metric.reshape(-1, 1), 
                                                     smoothing_method, smoothing_window).flatten()
        
        # === Step 7: Create output trajectory dict ===
        trajectory_output = {
            "frames": [],
            "reference_depth": ref_depth,
            "tracking_mode": tracking_mode,
            "smoothing": smoothing_method,
        }
        
        for i in range(N):
            trajectory_output["frames"].append({
                "frame": i,
                "position_2d": tracks_2d[i].tolist(),
                "position_3d": trajectory_3d[i].tolist(),
                "depth_metric": float(depths_metric[i]),
                "depth_relative": float(depths_raw[i]),
            })
        
        # === Step 8: Update mesh_sequence with new depth values ===
        mesh_sequence_updated = mesh_sequence
        if mesh_sequence is not None:
            mesh_sequence_updated = mesh_sequence.copy() if hasattr(mesh_sequence, 'copy') else dict(mesh_sequence)
            frames = mesh_sequence_updated.get("frames", [])
            
            for i, frame_data in enumerate(frames):
                if i < N:
                    # Update pred_cam_t with tracked depth
                    if "pred_cam_t" in frame_data:
                        old_tz = frame_data["pred_cam_t"][2]
                        new_tz = depths_metric[i]
                        frame_data["pred_cam_t_original"] = frame_data["pred_cam_t"].copy()
                        frame_data["pred_cam_t"][2] = new_tz
                    
                    # Add tracking data
                    frame_data["tracked_position_2d"] = tracks_2d[i].tolist()
                    frame_data["tracked_position_3d"] = trajectory_3d[i].tolist()
                    frame_data["tracked_depth"] = float(depths_metric[i])
        
        # === Step 9: Create debug video ===
        if output_debug_video:
            print(f"[CharacterTracker] Creating debug visualization...")
            debug_video = self._create_debug_video(images_np, masks_np, tracks_2d, 
                                                   depths_metric, trajectory_3d)
            debug_video = torch.from_numpy(debug_video)
        else:
            debug_video = images  # Return original
        
        # === Step 10: Create info string ===
        depth_change = depths_metric[-1] - depths_metric[0]
        depth_pct = (depth_change / ref_depth) * 100
        
        info = f"""Character Trajectory Tracking Results
=====================================
Frames: {N}
Tracking Mode: {tracking_mode}
Tracker: {'TAPIR' if tapir_loaded else 'Optical Flow'}
Smoothing: {smoothing_method} (window={smoothing_window})

Depth Analysis:
  Reference depth (frame 0): {ref_depth:.2f}m
  Final depth (frame {N-1}): {depths_metric[-1]:.2f}m
  Depth change: {depth_change:+.2f}m ({depth_pct:+.1f}%)
  Depth range: {depths_metric.min():.2f}m to {depths_metric.max():.2f}m

3D Trajectory:
  Start: ({trajectory_3d[0, 0]:.2f}, {trajectory_3d[0, 1]:.2f}, {trajectory_3d[0, 2]:.2f})
  End: ({trajectory_3d[-1, 0]:.2f}, {trajectory_3d[-1, 1]:.2f}, {trajectory_3d[-1, 2]:.2f})

Use the updated mesh_sequence for FBX export with corrected depth values.
"""
        
        print(f"[CharacterTracker] Tracking complete!")
        print(f"[CharacterTracker] Depth change: {depth_change:+.2f}m ({depth_pct:+.1f}%)")
        
        return (trajectory_output, mesh_sequence_updated, debug_video, info)


# Node registration
NODE_CLASS_MAPPINGS = {
    "CharacterTrajectoryTracker": CharacterTrajectoryTracker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CharacterTrajectoryTracker": "🏃 Character Trajectory Tracker",
}