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

# Import logger
try:
    from ..lib.logger import log, set_module
    set_module("Character Trajectory")
except ImportError:
    class _FallbackLog:
        def info(self, msg): print(f"[Character Trajectory] {msg}")
        def debug(self, msg): pass
        def warn(self, msg): print(f"[Character Trajectory] WARN: {msg}")
        def error(self, msg): print(f"[Character Trajectory] ERROR: {msg}")
        def progress(self, c, t, task="", interval=10): 
            if c == 0 or c == t - 1 or (c + 1) % interval == 0:
                print(f"[Character Trajectory] {task}: {c + 1}/{t}")
    log = _FallbackLog()

# Try to import TAPIR
TAPIR_AVAILABLE = False
try:
    from tapnet.torch import tapir_model
    TAPIR_AVAILABLE = True
    log.info("TAPIR (tapnet) available")
except ImportError:
    log.info("TAPIR not found - will use external TAPIR tracks or optical flow fallback")


class CharacterTrajectoryTracker:
    """
    Track character in 3D space using TAPIR (2D) + Depth Anything V2 (depth).
    
    This provides accurate 3D trajectory even when character moves toward/away
    from camera, which SAM3DBody's weak perspective model struggles with.
    
    v4.8.8: Added joint-based tracking using actual SAM3DBody joint positions.
    """
    
    # Joint definitions for reference
    COCO_JOINT_NAMES = [
        "0: Nose", "1: L_Eye", "2: R_Eye", "3: L_Ear", "4: R_Ear",
        "5: L_Shoulder", "6: R_Shoulder", "7: L_Elbow", "8: R_Elbow",
        "9: L_Wrist", "10: R_Wrist", "11: L_Hip (Pelvis Proxy)", "12: R_Hip",
        "13: L_Knee", "14: R_Knee", "15: L_Ankle", "16: R_Ankle"
    ]
    
    SMPLH_JOINT_NAMES = [
        "0: Pelvis", "1: L_Hip", "2: R_Hip", "3: Spine1", "4: L_Knee", "5: R_Knee",
        "6: Spine2", "7: L_Ankle", "8: R_Ankle", "9: Spine3", "10: L_Foot", "11: R_Foot",
        "12: Neck", "13: L_Collar", "14: R_Collar", "15: Head", "16: L_Shoulder", "17: R_Shoulder",
        "18: L_Elbow", "19: R_Elbow", "20: L_Wrist", "21: R_Wrist"
    ]
    
    TRACKING_MODES = [
        "Joint Position (Recommended)",  # NEW - uses actual joint
        "Pelvis Region (Legacy)",         # Old region-based
        "Feet Region (Ground Contact)", 
        "Full Body (Average)", 
    ]
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
                    "default": "Joint Position (Recommended)",
                    "tooltip": "Joint Position uses actual SAM3DBody joint. Legacy modes use mask region approximation."
                }),
                "skeleton_format": (["COCO (keypoints_2d)", "SMPL-H (joint_coords)"], {
                    "default": "COCO (keypoints_2d)",
                    "tooltip": "Which skeleton format to use for joint tracking"
                }),
                "reference_joint_coco": (cls.COCO_JOINT_NAMES, {
                    "default": "11: L_Hip (Pelvis Proxy)",
                    "tooltip": "Joint to track for COCO format. L_Hip (11) is closest to pelvis."
                }),
                "reference_joint_smplh": (cls.SMPLH_JOINT_NAMES, {
                    "default": "0: Pelvis",
                    "tooltip": "Joint to track for SMPL-H format. 0 is true pelvis."
                }),
                
                # === TAPIR Settings ===
                "tapir_grid_size": ("INT", {
                    "default": 4,
                    "min": 2,
                    "max": 8,
                    "tooltip": "Grid NxN points on character. 4=16pts, 6=36pts, 8=64pts. Lower = less VRAM."
                }),
                "tapir_batch_frames": ("INT", {
                    "default": 25,
                    "min": 5,
                    "max": 100,
                    "tooltip": "Process video in batches of N frames. Lower = less VRAM but may lose tracking continuity."
                }),
                "use_tapir": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use TAPIR for tracking. Disable to use optical flow fallback (faster, less VRAM)."
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
                log.info("TAPIR checkpoint not found")
                return False
            
            self.tapir_model = tapir_model.TAPIR(pyramid_level=1)
            self.tapir_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.tapir_model.to(self.device)
            self.tapir_model.eval()
            
            log.info(f"TAPIR loaded from {checkpoint_path}")
            return True
            
        except Exception as e:
            log.info(f"Error loading TAPIR: {e}")
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
        
        if tracking_mode == "Pelvis (Center)" or tracking_mode == "Pelvis Region (Legacy)":
            # Focus on middle region (pelvis area)
            pelvis_y_min = y_min + int(mask_height * 0.35)
            pelvis_y_max = y_min + int(mask_height * 0.55)
            
            pelvis_mask = (y_coords >= pelvis_y_min) & (y_coords <= pelvis_y_max)
            if pelvis_mask.sum() > 0:
                x_coords = x_coords[pelvis_mask]
                y_coords = y_coords[pelvis_mask]
        
        elif tracking_mode == "Feet (Ground Contact)" or tracking_mode == "Feet Region (Ground Contact)":
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
        if tracking_mode == "Pelvis (Center)" or tracking_mode == "Pelvis Region (Legacy)":
            pelvis_y_min = y_min + int(mask_height * 0.35)
            pelvis_y_max = y_min + int(mask_height * 0.55)
            region_mask = mask.copy()
            region_mask[:pelvis_y_min, :] = 0
            region_mask[pelvis_y_max:, :] = 0
        elif tracking_mode == "Feet (Ground Contact)" or tracking_mode == "Feet Region (Ground Contact)":
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
            trajectory: [N, D] array of positions (D can be 1, 2, or 3)
            method: Smoothing method
            window: Window size
            
        Returns:
            Smoothed trajectory
        """
        if method == "None" or len(trajectory) < window:
            return trajectory
        
        # Handle 1D arrays
        was_1d = False
        if len(trajectory.shape) == 1:
            was_1d = True
            trajectory = trajectory.reshape(-1, 1)
        
        smoothed = trajectory.copy()
        
        # Ensure odd window
        window = window if window % 2 == 1 else window + 1
        
        # Make sure window doesn't exceed data length
        window = min(window, len(trajectory))
        if window < 3:
            return trajectory.flatten() if was_1d else trajectory
        
        num_dims = trajectory.shape[1]
        
        if method == "Gaussian":
            for i in range(num_dims):
                smoothed[:, i] = ndimage.gaussian_filter1d(trajectory[:, i], sigma=window/4)
        
        elif method == "Savitzky-Golay":
            poly_order = min(3, window - 1)
            for i in range(num_dims):
                smoothed[:, i] = savgol_filter(trajectory[:, i], window, poly_order)
        
        elif method == "Moving Average":
            kernel = np.ones(window) / window
            for i in range(num_dims):
                smoothed[:, i] = np.convolve(trajectory[:, i], kernel, mode='same')
        
        if was_1d:
            return smoothed.flatten()
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
                          tracking_mode: str, grid_size: int = 4,
                          batch_frames: int = 25) -> np.ndarray:
        """
        Track character using TAPIR with memory-efficient batch processing.
        
        Args:
            images: [N, H, W, 3] video frames
            masks: [N, H, W] character masks
            tracking_mode: Tracking mode
            grid_size: NxN grid for point sampling (4=16pts, 6=36pts, 8=64pts)
            batch_frames: Process video in batches of N frames
            
        Returns:
            [N, 2] array of (x, y) averaged tracked positions
        """
        import gc
        
        N, H, W, _ = images.shape
        num_points = grid_size * grid_size
        
        log.info(f"TAPIR settings: grid={grid_size}x{grid_size}={num_points}pts, batch_frames={batch_frames}")
        
        # Get initial points from first frame mask
        init_points = self._sample_points_from_mask(masks[0], tracking_mode, grid_size)
        log.info(f"Tracking {len(init_points)} points (requested {num_points})")
        
        # Result array
        all_tracks = np.zeros((N, 2))
        
        # Process in batches to manage VRAM
        num_batches = (N + batch_frames - 1) // batch_frames
        
        for batch_idx in range(num_batches):
            start_frame = batch_idx * batch_frames
            end_frame = min(start_frame + batch_frames, N)
            batch_size = end_frame - start_frame
            
            log.info(f"Processing batch {batch_idx + 1}/{num_batches} (frames {start_frame}-{end_frame-1})")
            
            # Get batch of frames
            batch_images = images[start_frame:end_frame]
            
            # Prepare video for TAPIR: [B, T, H, W, 3], range [-1, 1]
            video = torch.from_numpy(batch_images).float()
            if video.max() > 1.0:
                video = video / 255.0
            video = video * 2 - 1  # Normalize to [-1, 1]
            video = video.unsqueeze(0)  # [1, T, H, W, C]
            video = video.to(self.device)
            
            # For subsequent batches, use last known position as starting point
            if batch_idx == 0:
                query_frame = 0
                query_pts = init_points
            else:
                # Re-initialize from mask at start of batch
                query_frame = 0  # Relative to batch
                query_pts = self._sample_points_from_mask(masks[start_frame], tracking_mode, grid_size)
            
            # Query points: [B, N, 3] where each point is [t, y, x]
            query_points = torch.zeros((1, len(query_pts), 3), dtype=torch.float32, device=self.device)
            query_points[0, :, 0] = query_frame
            query_points[0, :, 1] = torch.from_numpy(query_pts[:, 1].astype(np.float32))  # y
            query_points[0, :, 2] = torch.from_numpy(query_pts[:, 0].astype(np.float32))  # x
            
            # Run TAPIR with memory management
            try:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):  # Disable autocast to avoid BFloat16
                        outputs = self.tapir_model(video, query_points)
                
                # Extract results - convert to float32 explicitly
                tracks = outputs['tracks'][0].float().cpu().numpy()  # [N_points, T, 2]
                occlusion_logits = outputs['occlusion'][0].float()
                occlusion = torch.sigmoid(occlusion_logits).cpu().numpy()  # [N_points, T]
                
                # Average visible points per frame in this batch
                for t in range(batch_size):
                    visible_mask = occlusion[:, t] < 0.5  # Not occluded
                    if visible_mask.sum() > 0:
                        all_tracks[start_frame + t] = tracks[visible_mask, t, :].mean(axis=0)
                    else:
                        all_tracks[start_frame + t] = tracks[:, t, :].mean(axis=0)
                        
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log.info(f"WARNING: GPU OOM in batch {batch_idx + 1}. Try reducing:")
                    log.debug(f" - tapir_batch_frames (current: {batch_frames})")
                    log.debug(f" - tapir_grid_size (current: {grid_size}, points: {num_points})")
                    # Fall back to optical flow for this batch
                    log.info(f"Falling back to optical flow for this batch...")
                    batch_tracks = self._track_with_optical_flow(
                        batch_images, masks[start_frame:end_frame], tracking_mode
                    )
                    all_tracks[start_frame:end_frame] = batch_tracks
                else:
                    raise e
            finally:
                # Clean up GPU memory
                del video, query_points
                if 'outputs' in locals():
                    del outputs
                if 'tracks' in locals():
                    del tracks
                if 'occlusion_logits' in locals():
                    del occlusion_logits
                torch.cuda.empty_cache()
                gc.collect()
        
        log.info(f"TAPIR tracking complete for {N} frames")
        return all_tracks
    
    def _sample_points_from_mask(self, mask: np.ndarray, tracking_mode: str, 
                                  grid_size: int = 4) -> np.ndarray:
        """
        Sample tracking points from mask region using a grid pattern.
        
        Args:
            mask: Binary mask [H, W]
            tracking_mode: Which body region to sample
            grid_size: NxN grid (e.g., 4 = 16 points, 6 = 36 points)
            
        Returns:
            points: [N, 2] array of (x, y) coordinates
        """
        H, W = mask.shape
        
        coords = np.where(mask > 0.5)
        if len(coords[0]) == 0:
            # Empty mask - return center point
            return np.array([[W // 2, H // 2]], dtype=np.float32)
        
        y_coords, x_coords = coords
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()
        mask_height = y_max - y_min
        mask_width = x_max - x_min
        
        # Filter by tracking mode to get the region of interest
        if tracking_mode == "Pelvis (Center)" or tracking_mode == "Pelvis Region (Legacy)":
            # Focus on middle region (pelvis area)
            roi_y_min = y_min + int(mask_height * 0.35)
            roi_y_max = y_min + int(mask_height * 0.55)
            roi_x_min = x_min
            roi_x_max = x_max
        elif tracking_mode == "Feet (Ground Contact)" or tracking_mode == "Feet Region (Ground Contact)":
            # Focus on bottom region (feet)
            roi_y_min = y_min + int(mask_height * 0.85)
            roi_y_max = y_max
            roi_x_min = x_min
            roi_x_max = x_max
        else:  # Full Body
            roi_y_min = y_min
            roi_y_max = y_max
            roi_x_min = x_min
            roi_x_max = x_max
        
        # Create grid points within the ROI
        roi_height = roi_y_max - roi_y_min
        roi_width = roi_x_max - roi_x_min
        
        if roi_height < 2 or roi_width < 2:
            # ROI too small, use centroid
            return np.array([[(roi_x_min + roi_x_max) // 2, (roi_y_min + roi_y_max) // 2]], dtype=np.float32)
        
        # Generate grid coordinates
        grid_y = np.linspace(roi_y_min + roi_height * 0.1, roi_y_max - roi_height * 0.1, grid_size)
        grid_x = np.linspace(roi_x_min + roi_width * 0.1, roi_x_max - roi_width * 0.1, grid_size)
        
        # Create meshgrid
        xx, yy = np.meshgrid(grid_x, grid_y)
        grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1)
        
        # Filter to keep only points inside the mask
        valid_points = []
        for pt in grid_points:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < W and 0 <= y < H and mask[y, x] > 0.5:
                valid_points.append(pt)
        
        if len(valid_points) == 0:
            # No valid grid points, fall back to centroid
            centroid_x = (roi_x_min + roi_x_max) // 2
            centroid_y = (roi_y_min + roi_y_max) // 2
            return np.array([[centroid_x, centroid_y]], dtype=np.float32)
        
        return np.array(valid_points, dtype=np.float32)
    
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
                        tracking_mode="Joint Position (Recommended)",
                        skeleton_format="COCO (keypoints_2d)",
                        reference_joint_coco="11: L_Hip (Pelvis Proxy)",
                        reference_joint_smplh="0: Pelvis",
                        tapir_grid_size=4,
                        tapir_batch_frames=25,
                        use_tapir=True,
                        smoothing_method="Savitzky-Golay",
                        smoothing_window=5,
                        output_debug_video=True):
        """
        Main tracking function.
        
        Combines 2D tracking (TAPIR or optical flow) with depth maps to produce
        accurate 3D trajectory.
        
        v4.8.8: Added joint-based tracking using actual SAM3DBody joint positions.
        """
        log.info(f"Starting character trajectory tracking...")
        
        # Parse joint index from selection string
        if skeleton_format == "COCO (keypoints_2d)":
            joint_idx = int(reference_joint_coco.split(":")[0])
            joint_key = "pred_keypoints_2d"
            joint_name = reference_joint_coco.split(": ")[1] if ": " in reference_joint_coco else f"Joint {joint_idx}"
        else:
            joint_idx = int(reference_joint_smplh.split(":")[0])
            joint_key = "joints_2d"  # SMPL-H 2D projections
            joint_name = reference_joint_smplh.split(": ")[1] if ": " in reference_joint_smplh else f"Joint {joint_idx}"
        
        log.info(f"Tracking mode: {tracking_mode}")
        if "Joint" in tracking_mode:
            log.info(f"Using joint: {joint_idx} ({joint_name}) from {skeleton_format}")
        
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
        
        num_points = tapir_grid_size * tapir_grid_size
        
        log.info(f"Processing {N} frames at {W}x{H}")
        
        # === Step 1: Get reference depth from SAM3DBody or use provided ===
        if mesh_sequence is not None:
            frames = mesh_sequence.get("frames", {})
            # Handle both dict and list formats
            if isinstance(frames, dict):
                if frames:
                    first_key = sorted(frames.keys())[0]
                    first_frame = frames[first_key]
                    if "pred_cam_t" in first_frame:
                        ref_depth = abs(first_frame["pred_cam_t"][2])
                        log.info(f"Reference depth from SAM3DBody: {ref_depth:.2f}m")
                    else:
                        ref_depth = reference_depth_m
                        log.info(f"Using provided reference depth: {ref_depth:.2f}m")
                else:
                    ref_depth = reference_depth_m
                    log.info(f"Using provided reference depth: {ref_depth:.2f}m")
            else:
                if frames and "pred_cam_t" in frames[0]:
                    ref_depth = abs(frames[0]["pred_cam_t"][2])
                    log.info(f"Reference depth from SAM3DBody: {ref_depth:.2f}m")
                else:
                    ref_depth = reference_depth_m
                    log.info(f"Using provided reference depth: {ref_depth:.2f}m")
        else:
            ref_depth = reference_depth_m
            log.info(f"Using provided reference depth: {ref_depth:.2f}m")
        
        # === Step 2: Track 2D position ===
        log.info(f"Tracking 2D position...")
        
        # Check if we should use joint-based tracking
        use_joint_tracking = False
        joint_tracks_2d = None
        
        if "Joint" in tracking_mode and mesh_sequence is not None:
            # Try to extract joint positions from mesh_sequence
            frames = mesh_sequence.get("frames", {})
            if isinstance(frames, dict):
                frame_indices = sorted([int(k) for k in frames.keys() if str(k).isdigit()])
            else:
                frame_indices = list(range(len(frames)))
            
            joint_tracks_2d = []
            valid_joint_frames = 0
            
            for i in range(N):
                frame_key = frame_indices[i] if i < len(frame_indices) else frame_indices[-1]
                frame_data = frames.get(frame_key, frames.get(str(frame_key), {})) if isinstance(frames, dict) else (frames[i] if i < len(frames) else frames[-1])
                
                # Try to get joint 2D position
                joint_2d = None
                
                # Try pred_keypoints_2d first (COCO format)
                if joint_key == "pred_keypoints_2d":
                    kp_2d = frame_data.get("pred_keypoints_2d")
                    if kp_2d is not None:
                        kp_2d = np.array(kp_2d)
                        if kp_2d.ndim == 3:
                            kp_2d = kp_2d.squeeze(0)
                        if joint_idx < len(kp_2d):
                            joint_2d = kp_2d[joint_idx][:2]  # Get x, y
                
                # Try joints_2d (SMPL-H projected)
                if joint_2d is None and joint_key == "joints_2d":
                    j_2d = frame_data.get("joints_2d")
                    if j_2d is None:
                        j_2d = frame_data.get("pred_joint_coords_2d")
                    if j_2d is not None:
                        j_2d = np.array(j_2d)
                        if j_2d.ndim == 3:
                            j_2d = j_2d.squeeze(0)
                        if joint_idx < len(j_2d):
                            joint_2d = j_2d[joint_idx][:2]
                
                # Fallback: project 3D joint to 2D if we have camera params
                if joint_2d is None:
                    kp_3d = frame_data.get("pred_keypoints_3d")
                    if kp_3d is None:
                        kp_3d = frame_data.get("joint_coords")
                    cam_t = frame_data.get("pred_cam_t")
                    focal = frame_data.get("focal_length", 1000.0)
                    
                    if kp_3d is not None and cam_t is not None:
                        kp_3d = np.array(kp_3d)
                        if kp_3d.ndim == 3:
                            kp_3d = kp_3d.squeeze(0)
                        cam_t = np.array(cam_t).flatten()
                        
                        if joint_idx < len(kp_3d):
                            # Project 3D to 2D
                            joint_3d = kp_3d[joint_idx]
                            # Apply camera translation
                            joint_cam = joint_3d + cam_t[:3]
                            # Project
                            if isinstance(focal, (list, tuple)):
                                focal = focal[0]
                            focal = float(focal)
                            if joint_cam[2] > 0:
                                x_2d = focal * joint_cam[0] / joint_cam[2] + W / 2
                                y_2d = focal * joint_cam[1] / joint_cam[2] + H / 2
                                joint_2d = np.array([x_2d, y_2d])
                
                if joint_2d is not None:
                    joint_tracks_2d.append(joint_2d)
                    valid_joint_frames += 1
                else:
                    # Use last known position or center
                    if joint_tracks_2d:
                        joint_tracks_2d.append(joint_tracks_2d[-1])
                    else:
                        joint_tracks_2d.append(np.array([W / 2, H / 2]))
            
            if valid_joint_frames > N * 0.5:  # At least 50% valid
                joint_tracks_2d = np.array(joint_tracks_2d)
                use_joint_tracking = True
                log.info(f"✓ Using joint-based tracking: {valid_joint_frames}/{N} frames with valid joint data")
            else:
                log.warning(f"Only {valid_joint_frames}/{N} frames have valid joint data, falling back to mask tracking")
        
        if use_joint_tracking:
            tracks_2d = joint_tracks_2d
            tapir_loaded = False  # Not used for joint tracking
        else:
            # Fall back to TAPIR or optical flow
            tapir_loaded = self._load_tapir() if use_tapir else False
            
            if tapir_loaded and use_tapir:
                log.info(f"Using TAPIR for tracking")
                tracks_2d = self._track_with_tapir(
                    images_np, masks_np, tracking_mode,
                    grid_size=tapir_grid_size, batch_frames=tapir_batch_frames
                )
            else:
                if not use_tapir:
                    log.info(f"TAPIR disabled, using optical flow")
                else:
                    log.info(f"TAPIR not available, using optical flow fallback")
                tracks_2d = self._track_with_optical_flow(images_np, masks_np, tracking_mode)
        
        # === Step 3: Sample depth at tracked positions ===
        log.info(f"Sampling depth values...")
        
        depths_raw = np.zeros(N)
        for i in range(N):
            # Get depth map for this frame (handle single channel or RGB)
            if len(depth_np.shape) == 4 and depth_np.shape[-1] == 3:
                depth_frame = depth_np[i, :, :, 0]  # Use first channel
            elif len(depth_np.shape) == 4:
                depth_frame = depth_np[i, :, :, 0]
            else:
                depth_frame = depth_np[i]
            
            if use_joint_tracking:
                # Sample depth at joint 2D position
                x, y = int(tracks_2d[i, 0]), int(tracks_2d[i, 1])
                # Clamp to image bounds
                x = max(0, min(x, depth_frame.shape[1] - 1))
                y = max(0, min(y, depth_frame.shape[0] - 1))
                
                # Sample with small neighborhood for robustness
                y_min = max(0, y - 2)
                y_max = min(depth_frame.shape[0], y + 3)
                x_min = max(0, x - 2)
                x_max = min(depth_frame.shape[1], x + 3)
                
                neighborhood = depth_frame[y_min:y_max, x_min:x_max]
                if neighborhood.size > 0:
                    depths_raw[i] = float(np.median(neighborhood))
                else:
                    depths_raw[i] = float(depth_frame[y, x])
            else:
                # Sample depth at mask region (legacy)
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
        
        log.info(f"Depth range: {depths_metric.min():.2f}m to {depths_metric.max():.2f}m")
        
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
            log.info(f"Applying {smoothing_method} smoothing (window={smoothing_window})")
            trajectory_3d = self._smooth_trajectory(trajectory_3d, smoothing_method, smoothing_window)
            depths_metric = self._smooth_trajectory(depths_metric, smoothing_method, smoothing_window)
        
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
            try:
                # Deep copy to avoid modifying original
                import copy
                mesh_sequence_updated = copy.deepcopy(mesh_sequence)
                frames = mesh_sequence_updated.get("frames", {})
                
                # frames can be a dict (with int keys) or a list
                if isinstance(frames, dict):
                    frame_items = list(frames.items())  # [(idx, frame_data), ...]
                else:
                    frame_items = list(enumerate(frames))  # [(0, frame_data), (1, frame_data), ...]
                
                for frame_idx, frame_data in frame_items:
                    i = frame_idx if isinstance(frame_idx, int) else int(frame_idx)
                    if i >= N:
                        break
                    
                    # Check if frame_data is a dict
                    if not isinstance(frame_data, dict):
                        continue
                    
                    # Update pred_cam_t with tracked depth
                    if "pred_cam_t" in frame_data and isinstance(frame_data["pred_cam_t"], (list, np.ndarray)):
                        pred_cam_t = frame_data["pred_cam_t"]
                        if hasattr(pred_cam_t, '__len__') and len(pred_cam_t) >= 3:
                            # Save original
                            if isinstance(pred_cam_t, np.ndarray):
                                frame_data["pred_cam_t_original"] = pred_cam_t.copy().tolist()
                                frame_data["pred_cam_t"] = pred_cam_t.copy()
                                frame_data["pred_cam_t"][2] = float(depths_metric[i])
                                frame_data["pred_cam_t"] = frame_data["pred_cam_t"].tolist()
                            else:
                                frame_data["pred_cam_t_original"] = list(pred_cam_t)
                                frame_data["pred_cam_t"] = list(pred_cam_t)
                                frame_data["pred_cam_t"][2] = float(depths_metric[i])
                    
                    # Add tracking data
                    frame_data["tracked_position_2d"] = tracks_2d[i].tolist()
                    frame_data["tracked_position_3d"] = trajectory_3d[i].tolist()
                    frame_data["tracked_depth"] = float(depths_metric[i])
                    
                log.info(f"Updated {len(frame_items)} frames with depth data")
                    
            except Exception as e:
                log.info(f"Warning: Could not update mesh_sequence: {e}")
                import traceback
                traceback.print_exc()
                mesh_sequence_updated = mesh_sequence
        
        # === Step 9: Create debug video ===
        if output_debug_video:
            log.info(f"Creating debug visualization...")
            debug_video = self._create_debug_video(images_np, masks_np, tracks_2d, 
                                                   depths_metric, trajectory_3d)
            debug_video = torch.from_numpy(debug_video)
        else:
            debug_video = images  # Return original
        
        # === Step 10: Create info string ===
        depth_change = depths_metric[-1] - depths_metric[0]
        depth_pct = (depth_change / ref_depth) * 100
        
        # Determine tracker info
        if use_joint_tracking:
            tracker_info = f"Joint-Based ({joint_name})"
            tapir_settings = f"\n  Skeleton: {skeleton_format}\n  Joint Index: {joint_idx}"
        else:
            tracker_info = "TAPIR" if (tapir_loaded and use_tapir) else "Optical Flow"
            tapir_settings = f"\n  Grid: {tapir_grid_size}x{tapir_grid_size} = {num_points} points\n  Batch Frames: {tapir_batch_frames}" if (tapir_loaded and use_tapir) else ""
        
        info = f"""Character Trajectory Tracking Results
=====================================
Frames: {N}
Tracking Mode: {tracking_mode}
Tracker: {tracker_info}{tapir_settings}
Smoothing: {smoothing_method} (window={smoothing_window})

Depth Analysis:
  Reference depth (frame 0): {ref_depth:.2f}m
  Final depth (frame {N-1}): {depths_metric[-1]:.2f}m
  Depth change: {depth_change:+.2f}m ({depth_pct:+.1f}%)
  Depth range: {depths_metric.min():.2f}m to {depths_metric.max():.2f}m

3D Trajectory:
  Start: ({trajectory_3d[0, 0]:.2f}, {trajectory_3d[0, 1]:.2f}, {trajectory_3d[0, 2]:.2f})
  End: ({trajectory_3d[-1, 0]:.2f}, {trajectory_3d[-1, 1]:.2f}, {trajectory_3d[-1, 2]:.2f})

Tips for GPU Memory (if OOM):
  - Reduce tapir_grid_size: 4=16pts, 3=9pts, 2=4pts
  - Reduce tapir_batch_frames (current: {tapir_batch_frames})
  - Or disable use_tapir for optical flow fallback

Use the updated mesh_sequence for FBX export with corrected depth values.
"""
        
        # Store settings in mesh_sequence for FBX metadata
        if mesh_sequence_updated is not None:
            mesh_sequence_updated["character_trajectory_settings"] = {
                "tracking_mode": tracking_mode,
                "tracker_type": "joint" if use_joint_tracking else ("tapir" if (tapir_loaded and use_tapir) else "optical_flow"),
                "joint_index": joint_idx if use_joint_tracking else -1,
                "joint_name": joint_name if use_joint_tracking else "",
                "skeleton_format": skeleton_format if use_joint_tracking else "",
                "smoothing_method": smoothing_method,
                "smoothing_window": smoothing_window,
                "reference_depth_m": ref_depth,
            }
        else:
            log.info("WARNING: mesh_sequence input was not connected - mesh_sequence_updated output will be None!")
            log.info("         Connect a MESH_SEQUENCE from Video Processor or processing nodes to this input.")
        
        log.info(f"Tracking complete!")
        log.info(f"Depth change: {depth_change:+.2f}m ({depth_pct:+.1f}%)")
        
        return (trajectory_output, mesh_sequence_updated, debug_video, info)


# Node registration
NODE_CLASS_MAPPINGS = {
    "CharacterTrajectoryTracker": CharacterTrajectoryTracker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CharacterTrajectoryTracker": "🏃 Character Trajectory Tracker",
}