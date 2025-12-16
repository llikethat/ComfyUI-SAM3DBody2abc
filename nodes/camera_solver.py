"""
Camera Rotation Solver Node for SAM3DBody2abc

Estimates camera rotation (pan/tilt) from video frames using optical flow
on the background (excluding foreground masks from SAM3).

This solves the fundamental problem: pred_cam_t from SAM3DBody tells us
WHERE the body appears on screen, but not WHY (body movement vs camera rotation).

By analyzing background motion, we can determine the actual camera rotation.

Pipeline:
1. Invert SAM3 masks to get background
2. Compute optical flow (RAFT) between consecutive frames
3. Extract flow only from background regions
4. Estimate homography from background flow
5. Decompose homography to get camera rotation

Requirements:
- torchvision >= 0.14 (for RAFT)
- opencv-python
"""

import numpy as np
import torch
import cv2
from typing import Dict, Tuple, Any, Optional, List


class CameraRotationSolver:
    """
    Estimates camera rotation from video frames using background optical flow.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "foreground_masks": ("MASK",),
                "sam3_masks": ("SAM3_VIDEO_MASKS",),
                "focal_length_px": ("FLOAT", {
                    "default": 1000.0,
                    "min": 100.0,
                    "max": 5000.0,
                    "tooltip": "Focal length in pixels (from SAM3DBody or estimated). Used for proper rotation decomposition."
                }),
                "flow_threshold": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Minimum flow magnitude to consider (filters noise)"
                }),
                "ransac_threshold": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "RANSAC reprojection threshold for homography estimation"
                }),
                "smoothing": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 15,
                    "tooltip": "Temporal smoothing window for rotation values (0=none)"
                }),
            }
        }
    
    RETURN_TYPES = ("CAMERA_ROTATION_DATA",)
    RETURN_NAMES = ("camera_rotations",)
    FUNCTION = "solve_camera_rotation"
    CATEGORY = "SAM3DBody2abc/Camera"
    
    def __init__(self):
        self.raft_model = None
        self.device = None
    
    def load_raft(self):
        """Load RAFT model from torchvision."""
        if self.raft_model is not None:
            return
        
        try:
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[CameraSolver] Loading RAFT model on {self.device}...")
            
            weights = Raft_Large_Weights.DEFAULT
            self.raft_model = raft_large(weights=weights, progress=True)
            self.raft_model = self.raft_model.to(self.device)
            self.raft_model.eval()
            
            # Get transforms
            self.raft_transforms = weights.transforms()
            
            print(f"[CameraSolver] RAFT model loaded successfully")
            
        except ImportError as e:
            print(f"[CameraSolver] Error: torchvision >= 0.14 required for RAFT. {e}")
            raise
    
    def compute_optical_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Compute optical flow between two frames using RAFT.
        
        Args:
            frame1: First frame (H, W, 3) uint8
            frame2: Second frame (H, W, 3) uint8
            
        Returns:
            flow: Optical flow (H, W, 2) - dx, dy per pixel
        """
        self.load_raft()
        
        # Convert to torch tensors
        img1 = torch.from_numpy(frame1).permute(2, 0, 1).float()  # (3, H, W)
        img2 = torch.from_numpy(frame2).permute(2, 0, 1).float()
        
        # Apply transforms
        img1, img2 = self.raft_transforms(img1, img2)
        
        # Add batch dimension
        img1 = img1.unsqueeze(0).to(self.device)
        img2 = img2.unsqueeze(0).to(self.device)
        
        # Compute flow
        with torch.no_grad():
            flow_predictions = self.raft_model(img1, img2)
            # RAFT returns list of predictions, take the last (most refined)
            flow = flow_predictions[-1]
        
        # Convert to numpy (H, W, 2)
        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        
        return flow
    
    def estimate_homography_from_flow(
        self, 
        flow: np.ndarray, 
        mask: Optional[np.ndarray], 
        flow_threshold: float,
        ransac_threshold: float
    ) -> Optional[np.ndarray]:
        """
        Estimate homography from optical flow on background regions.
        
        Args:
            flow: Optical flow (H, W, 2)
            mask: Background mask (H, W) - 1 for background, 0 for foreground
            flow_threshold: Minimum flow magnitude to use
            ransac_threshold: RANSAC threshold
            
        Returns:
            H: 3x3 homography matrix, or None if estimation failed
        """
        H, W = flow.shape[:2]
        
        # Create grid of points
        y_coords, x_coords = np.mgrid[0:H:8, 0:W:8]  # Sample every 8 pixels
        
        # Flatten
        src_points = np.column_stack([x_coords.ravel(), y_coords.ravel()]).astype(np.float32)
        
        # Get flow at these points
        flow_x = flow[::8, ::8, 0].ravel()
        flow_y = flow[::8, ::8, 1].ravel()
        
        # Destination points
        dst_points = src_points + np.column_stack([flow_x, flow_y])
        
        # Apply mask if provided
        if mask is not None:
            mask_sampled = mask[::8, ::8].ravel()
            valid = mask_sampled > 0.5
            src_points = src_points[valid]
            dst_points = dst_points[valid]
            flow_x = flow_x[valid]
            flow_y = flow_y[valid]
        
        # Filter by flow magnitude
        flow_mag = np.sqrt(flow_x**2 + flow_y**2)
        valid = flow_mag > flow_threshold
        src_points = src_points[valid]
        dst_points = dst_points[valid]
        
        if len(src_points) < 10:
            print(f"[CameraSolver] Warning: Not enough background points ({len(src_points)})")
            return None
        
        # Estimate homography with RANSAC
        H, inliers = cv2.findHomography(
            src_points, 
            dst_points, 
            cv2.RANSAC, 
            ransac_threshold
        )
        
        if H is None:
            print(f"[CameraSolver] Warning: Homography estimation failed")
            return None
        
        inlier_ratio = np.sum(inliers) / len(inliers) if inliers is not None else 0
        if inlier_ratio < 0.3:
            print(f"[CameraSolver] Warning: Low inlier ratio ({inlier_ratio:.2f})")
        
        return H
    
    def decompose_homography_to_rotation(
        self, 
        H: np.ndarray, 
        focal_length: float,
        image_width: int,
        image_height: int
    ) -> Tuple[float, float, float]:
        """
        Decompose homography to extract camera rotation.
        
        For a pure rotation, H = K * R * K^-1
        where K is the camera intrinsic matrix.
        
        Args:
            H: 3x3 homography matrix
            focal_length: Focal length in pixels
            image_width, image_height: Image dimensions
            
        Returns:
            (pan, tilt, roll): Rotation angles in radians
        """
        # Camera intrinsic matrix
        cx, cy = image_width / 2, image_height / 2
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        K_inv = np.linalg.inv(K)
        
        # For pure rotation: H = K * R * K^-1
        # Therefore: R = K^-1 * H * K
        R = K_inv @ H @ K
        
        # The result may not be a perfect rotation matrix due to noise
        # Use SVD to find the closest rotation matrix
        U, S, Vt = np.linalg.svd(R)
        R_clean = U @ Vt
        
        # Ensure proper rotation (det = 1)
        if np.linalg.det(R_clean) < 0:
            R_clean = -R_clean
        
        # Extract Euler angles (assuming Y-up convention for Maya)
        # R = Ry(pan) * Rx(tilt) * Rz(roll)
        
        # Tilt (rotation around X axis)
        tilt = np.arctan2(-R_clean[1, 2], R_clean[2, 2])
        
        # Pan (rotation around Y axis)
        pan = np.arctan2(R_clean[0, 2], np.sqrt(R_clean[1, 2]**2 + R_clean[2, 2]**2))
        
        # Roll (rotation around Z axis)
        roll = np.arctan2(-R_clean[0, 1], R_clean[0, 0])
        
        return pan, tilt, roll
    
    def smooth_rotations(self, rotations: List[Tuple[float, float, float]], window: int) -> List[Tuple[float, float, float]]:
        """Apply moving average smoothing to rotation values."""
        if window <= 1 or len(rotations) < window:
            return rotations
        
        pans = [r[0] for r in rotations]
        tilts = [r[1] for r in rotations]
        rolls = [r[2] for r in rotations]
        
        def smooth_array(values):
            result = []
            half = window // 2
            for i in range(len(values)):
                start = max(0, i - half)
                end = min(len(values), i + half + 1)
                avg = sum(values[start:end]) / (end - start)
                result.append(avg)
            return result
        
        smoothed_pans = smooth_array(pans)
        smoothed_tilts = smooth_array(tilts)
        smoothed_rolls = smooth_array(rolls)
        
        return list(zip(smoothed_pans, smoothed_tilts, smoothed_rolls))
    
    def solve_camera_rotation(
        self,
        images: torch.Tensor,
        foreground_masks: Optional[torch.Tensor] = None,
        sam3_masks: Optional[Any] = None,
        focal_length_px: float = 1000.0,
        flow_threshold: float = 1.0,
        ransac_threshold: float = 3.0,
        smoothing: int = 5,
    ) -> Tuple[Dict]:
        """
        Solve for camera rotation from video frames.
        
        Args:
            images: Video frames (N, H, W, C)
            foreground_masks: Foreground masks to exclude (N, H, W)
            sam3_masks: SAM3 video masks (alternative to foreground_masks)
            focal_length_px: Focal length in pixels
            flow_threshold: Minimum flow magnitude
            ransac_threshold: RANSAC threshold
            smoothing: Temporal smoothing window
            
        Returns:
            camera_rotations: Dict with per-frame rotation data
        """
        print(f"[CameraSolver] Starting camera rotation solve...")
        print(f"[CameraSolver] Input: {images.shape[0]} frames")
        
        # Convert images to numpy
        frames = (images.cpu().numpy() * 255).astype(np.uint8)
        num_frames, H, W, C = frames.shape
        
        print(f"[CameraSolver] Frame size: {W}x{H}")
        print(f"[CameraSolver] Focal length: {focal_length_px}px")
        
        # Process masks
        bg_masks = None
        if foreground_masks is not None:
            # Invert foreground masks to get background
            fg = foreground_masks.cpu().numpy()
            bg_masks = 1.0 - fg
            print(f"[CameraSolver] Using provided foreground masks")
        elif sam3_masks is not None:
            # Handle SAM3 mask format
            bg_masks = self._process_sam3_masks(sam3_masks, num_frames, H, W)
            print(f"[CameraSolver] Using SAM3 masks")
        else:
            print(f"[CameraSolver] No masks provided - using full frame")
        
        # Compute per-frame rotations
        rotations = []
        cumulative_pan = 0.0
        cumulative_tilt = 0.0
        cumulative_roll = 0.0
        
        # First frame has no rotation (reference)
        rotations.append((0.0, 0.0, 0.0))
        
        for i in range(1, num_frames):
            frame1 = frames[i - 1]
            frame2 = frames[i]
            
            # Get background mask for this frame pair
            mask = None
            if bg_masks is not None:
                # Use average of both frames' masks
                if len(bg_masks.shape) == 3:
                    mask = (bg_masks[i-1] + bg_masks[i]) / 2
                else:
                    mask = bg_masks[i] if i < len(bg_masks) else None
            
            # Compute optical flow
            try:
                flow = self.compute_optical_flow(frame1, frame2)
            except Exception as e:
                print(f"[CameraSolver] Frame {i}: Flow computation failed - {e}")
                rotations.append((cumulative_pan, cumulative_tilt, cumulative_roll))
                continue
            
            # Estimate homography
            H = self.estimate_homography_from_flow(flow, mask, flow_threshold, ransac_threshold)
            
            if H is None:
                # No valid homography, assume no rotation
                rotations.append((cumulative_pan, cumulative_tilt, cumulative_roll))
                continue
            
            # Decompose to rotation
            delta_pan, delta_tilt, delta_roll = self.decompose_homography_to_rotation(
                H, focal_length_px, W, H
            )
            
            # Accumulate rotation
            cumulative_pan += delta_pan
            cumulative_tilt += delta_tilt
            cumulative_roll += delta_roll
            
            rotations.append((cumulative_pan, cumulative_tilt, cumulative_roll))
            
            if i % 10 == 0:
                print(f"[CameraSolver] Frame {i}/{num_frames}: pan={np.degrees(cumulative_pan):.2f}°, tilt={np.degrees(cumulative_tilt):.2f}°")
        
        # Apply smoothing
        if smoothing > 1:
            rotations = self.smooth_rotations(rotations, smoothing)
            print(f"[CameraSolver] Applied smoothing (window={smoothing})")
        
        # Build output
        camera_rotation_data = {
            "num_frames": num_frames,
            "image_width": W,
            "image_height": H,
            "focal_length_px": focal_length_px,
            "rotations": [
                {
                    "frame": i,
                    "pan": rot[0],      # radians
                    "tilt": rot[1],     # radians
                    "roll": rot[2],     # radians
                    "pan_deg": np.degrees(rot[0]),
                    "tilt_deg": np.degrees(rot[1]),
                    "roll_deg": np.degrees(rot[2]),
                }
                for i, rot in enumerate(rotations)
            ]
        }
        
        # Summary
        final_rot = rotations[-1]
        print(f"[CameraSolver] Solve complete!")
        print(f"[CameraSolver] Total rotation: pan={np.degrees(final_rot[0]):.2f}°, tilt={np.degrees(final_rot[1]):.2f}°, roll={np.degrees(final_rot[2]):.2f}°")
        
        return (camera_rotation_data,)
    
    def _process_sam3_masks(self, sam3_masks, num_frames, H, W) -> np.ndarray:
        """Convert SAM3 masks to background masks."""
        try:
            # Handle different SAM3 mask formats
            if isinstance(sam3_masks, dict):
                # Frame-indexed dict
                bg_masks = []
                for i in range(num_frames):
                    if i in sam3_masks:
                        fg = sam3_masks[i]
                        if isinstance(fg, torch.Tensor):
                            fg = fg.cpu().numpy()
                        if fg.shape[0] != H or fg.shape[1] != W:
                            fg = cv2.resize(fg.astype(np.float32), (W, H))
                        bg_masks.append(1.0 - fg)
                    else:
                        bg_masks.append(np.ones((H, W), dtype=np.float32))
                return np.array(bg_masks)
            
            elif isinstance(sam3_masks, torch.Tensor):
                fg = sam3_masks.cpu().numpy()
                if len(fg.shape) == 4:
                    fg = fg[:, 0]  # Take first channel
                return 1.0 - fg
            
            else:
                print(f"[CameraSolver] Unknown SAM3 mask format: {type(sam3_masks)}")
                return None
                
        except Exception as e:
            print(f"[CameraSolver] Error processing SAM3 masks: {e}")
            return None


# Node registration
NODE_CLASS_MAPPINGS = {
    "CameraRotationSolver": CameraRotationSolver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CameraRotationSolver": "Camera Rotation Solver",
}
