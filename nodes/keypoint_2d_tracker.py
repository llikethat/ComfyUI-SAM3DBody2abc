"""
Keypoint 2D Tracker for SAM3DBody2abc
=====================================
Version: 1.0.0

Detects 2D keypoints in frame 0, then uses TAPIR to track them
across all frames for temporal consistency.

This replaces per-frame detection with detect-once-track-all approach,
eliminating jitter at the source.
"""

import os
import sys
import gc
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any


def log(msg):
    print(f"[Keypoint2DTracker] {msg}", flush=True)


# TAPIR Import (same pattern as camera_solver_v2.py)
TAPIR_AVAILABLE = False
TAPIR_BACKEND = None

try:
    from tapnet.torch import tapir_model
    from tapnet.utils import transforms as tapir_transforms
    TAPIR_BACKEND = "tapnet"
    TAPIR_AVAILABLE = True
except ImportError:
    try:
        from tapir.torch import tapir_model
        from tapir.utils import transforms as tapir_transforms
        TAPIR_BACKEND = "tapir"
        TAPIR_AVAILABLE = True
    except ImportError:
        pass


class Keypoint2DTracker:
    """
    Detect 2D keypoints once, track with TAPIR across all frames.
    
    Pipeline:
        Frame 0: SAM3D → pred_keypoints_2d (70, 2)
        Frame 0-N: TAPIR tracks those 70 points
        Output: tracked_keypoints_2d (N, 70, 2)
    """
    
    DEFAULT_CHECKPOINT = "models/tapir/bootstapir_checkpoint_v2.pt"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "sam3d_model": ("SAM3D_MODEL",),
            },
            "optional": {
                "detection_frame": ("INT", {
                    "default": 0, 
                    "min": 0,
                    "tooltip": "Frame to detect keypoints (usually 0)"
                }),
                "mask": ("MASK", {
                    "tooltip": "Optional mask for person detection"
                }),
                "tapir_checkpoint": ("STRING", {
                    "default": "",
                    "tooltip": "Path to TAPIR checkpoint (leave empty for default)"
                }),
                "tracking_resolution": (["full", "half", "quarter"], {
                    "default": "half",
                    "tooltip": "Resolution for TAPIR tracking (lower = faster, less VRAM)"
                }),
            },
        }

    RETURN_TYPES = ("KEYPOINTS_2D", "TENSOR", "BBOX", "STRING")
    RETURN_NAMES = ("tracked_keypoints_2d", "tracking_confidence", "detection_bbox", "status")
    FUNCTION = "process"
    CATEGORY = "SAM3DBody2abc/Tracking"

    def __init__(self):
        self.tapir_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def process(
        self,
        images: torch.Tensor,
        sam3d_model: Dict,
        detection_frame: int = 0,
        mask: Optional[torch.Tensor] = None,
        tapir_checkpoint: str = "",
        tracking_resolution: str = "half",
    ) -> Tuple[Dict, torch.Tensor, List, str]:
        
        log("=" * 60)
        log("Starting Keypoint 2D Tracker")
        log("=" * 60)
        
        # Get dimensions
        num_frames = images.shape[0]
        H, W = images.shape[1], images.shape[2]
        log(f"Input: {num_frames} frames, {W}x{H}")
        
        if detection_frame >= num_frames:
            detection_frame = 0
            log(f"Detection frame out of range, using frame 0")
        
        # =====================================================================
        # Step 1: Detect 2D keypoints in detection_frame using SAM3D
        # =====================================================================
        log(f"Step 1: Detecting keypoints in frame {detection_frame}...")
        
        keypoints_2d, bbox = self._detect_keypoints(
            images, sam3d_model, detection_frame, mask
        )
        
        if keypoints_2d is None:
            log("ERROR: Failed to detect keypoints!")
            empty_kp = {
                "keypoints": np.zeros((num_frames, 70, 2), dtype=np.float32),
                "num_frames": num_frames,
                "num_keypoints": 70,
                "detection_frame": detection_frame,
            }
            empty_conf = torch.zeros(num_frames, 70)
            return (empty_kp, empty_conf, [], "Failed to detect keypoints")
        
        num_keypoints = keypoints_2d.shape[0]
        log(f"Detected {num_keypoints} keypoints in frame {detection_frame}")
        log(f"Keypoint range: x=[{keypoints_2d[:,0].min():.1f}, {keypoints_2d[:,0].max():.1f}], "
            f"y=[{keypoints_2d[:,1].min():.1f}, {keypoints_2d[:,1].max():.1f}]")
        
        # =====================================================================
        # Step 2: Track keypoints using TAPIR
        # =====================================================================
        log(f"Step 2: Loading TAPIR model...")
        
        if not self._load_tapir(tapir_checkpoint):
            log("ERROR: TAPIR not available, using static keypoints")
            # Fallback: repeat detection frame keypoints for all frames
            static_kp = np.tile(keypoints_2d[np.newaxis, :, :], (num_frames, 1, 1))
            output = {
                "keypoints": static_kp,
                "num_frames": num_frames,
                "num_keypoints": num_keypoints,
                "detection_frame": detection_frame,
            }
            confidence = torch.ones(num_frames, num_keypoints)
            return (output, confidence, bbox, "TAPIR unavailable - using static keypoints")
        
        log(f"Step 3: Running TAPIR tracking ({tracking_resolution} resolution)...")
        
        tracks, confidence = self._run_tapir_tracking(
            images, keypoints_2d, detection_frame, tracking_resolution
        )
        
        if tracks is None:
            log("ERROR: TAPIR tracking failed")
            static_kp = np.tile(keypoints_2d[np.newaxis, :, :], (num_frames, 1, 1))
            output = {
                "keypoints": static_kp,
                "num_frames": num_frames,
                "num_keypoints": num_keypoints,
                "detection_frame": detection_frame,
            }
            conf = torch.ones(num_frames, num_keypoints)
            return (output, conf, bbox, "TAPIR tracking failed - using static keypoints")
        
        log(f"Tracking complete: {tracks.shape}")
        
        # =====================================================================
        # Step 3: Package output
        # =====================================================================
        output = {
            "keypoints": tracks,  # (N, 70, 2)
            "num_frames": num_frames,
            "num_keypoints": num_keypoints,
            "detection_frame": detection_frame,
            "image_size": (W, H),
        }
        
        # Stats
        mean_conf = confidence.mean().item()
        low_conf_frames = (confidence.mean(dim=1) < 0.5).sum().item()
        
        status = (
            f"Tracked {num_keypoints} keypoints across {num_frames} frames\n"
            f"Mean confidence: {mean_conf:.2f}\n"
            f"Low confidence frames: {low_conf_frames}"
        )
        
        log(status.replace('\n', ', '))
        log("=" * 60)
        
        return (output, confidence, bbox, status)

    def _detect_keypoints(
        self,
        images: torch.Tensor,
        sam3d_model: Dict,
        frame_idx: int,
        mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[np.ndarray], List]:
        """Detect 2D keypoints using SAM3D model."""
        
        try:
            # Get the estimator from model dict
            model = sam3d_model.get("model")
            if model is None:
                log("ERROR: No model in sam3d_model dict")
                return None, []
            
            # Get single frame
            frame = images[frame_idx]  # (H, W, C)
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            
            # Ensure uint8
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
            
            # Get mask for this frame if provided
            frame_mask = None
            if mask is not None:
                if len(mask.shape) == 4:
                    frame_mask = mask[frame_idx, 0].cpu().numpy()
                elif len(mask.shape) == 3:
                    frame_mask = mask[frame_idx].cpu().numpy()
                else:
                    frame_mask = mask.cpu().numpy()
            
            # Check if model has process_one_image method
            if hasattr(model, 'process_one_image'):
                outputs = model.process_one_image(frame, mask=frame_mask)
            elif hasattr(model, '__call__'):
                outputs = model(frame, mask=frame_mask)
            else:
                log(f"ERROR: Unknown model type: {type(model)}")
                return None, []
            
            # Handle list output (multiple people)
            if isinstance(outputs, list):
                if len(outputs) == 0:
                    log("No detections in frame")
                    return None, []
                outputs = outputs[0]  # Take first person
            
            # Extract 2D keypoints
            keypoints_2d = outputs.get("pred_keypoints_2d")
            if keypoints_2d is None:
                log("WARNING: pred_keypoints_2d not in model output")
                log(f"Available keys: {list(outputs.keys())}")
                return None, []
            
            if isinstance(keypoints_2d, torch.Tensor):
                keypoints_2d = keypoints_2d.cpu().numpy()
            
            # Get bbox
            bbox = outputs.get("bbox", [])
            if isinstance(bbox, torch.Tensor):
                bbox = bbox.cpu().numpy().tolist()
            elif isinstance(bbox, np.ndarray):
                bbox = bbox.tolist()
            
            return keypoints_2d, bbox
            
        except Exception as e:
            log(f"ERROR in _detect_keypoints: {e}")
            import traceback
            traceback.print_exc()
            return None, []

    def _load_tapir(self, checkpoint_path: str = "") -> bool:
        """Load TAPIR model."""
        
        if not TAPIR_AVAILABLE:
            log(f"TAPIR module not available (backend: {TAPIR_BACKEND})")
            return False
        
        if self.tapir_model is not None:
            return True
        
        # Find checkpoint
        paths_to_try = []
        
        if checkpoint_path:
            paths_to_try.append(checkpoint_path)
        
        # Standard locations
        base_dirs = [
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."),
            os.getcwd(),
        ]
        
        for base in base_dirs:
            paths_to_try.append(os.path.join(base, "models", "tapir", "bootstapir_checkpoint_v2.pt"))
            paths_to_try.append(os.path.join(base, "ComfyUI", "models", "tapir", "bootstapir_checkpoint_v2.pt"))
        
        checkpoint = None
        for path in paths_to_try:
            if os.path.exists(path):
                checkpoint = path
                break
        
        if checkpoint is None:
            log("ERROR: TAPIR checkpoint not found")
            log(f"Tried: {paths_to_try[:3]}...")
            return False
        
        log(f"Loading TAPIR from: {checkpoint}")
        
        try:
            self.tapir_model = tapir_model.TAPIR(pyramid_level=0)
            self.tapir_model.load_state_dict(torch.load(checkpoint, map_location=self.device))
            self.tapir_model = self.tapir_model.to(self.device)
            self.tapir_model.eval()
            log(f"TAPIR loaded successfully ({TAPIR_BACKEND} backend)")
            return True
        except Exception as e:
            log(f"ERROR loading TAPIR: {e}")
            return False

    def _run_tapir_tracking(
        self,
        images: torch.Tensor,
        keypoints_2d: np.ndarray,
        detection_frame: int,
        resolution: str,
    ) -> Tuple[Optional[np.ndarray], Optional[torch.Tensor]]:
        """Run TAPIR tracking on keypoints."""
        
        num_frames = images.shape[0]
        H, W = images.shape[1], images.shape[2]
        num_keypoints = keypoints_2d.shape[0]
        
        # Scale factor based on resolution setting
        scale_map = {"full": 1.0, "half": 0.5, "quarter": 0.25}
        scale_factor = scale_map.get(resolution, 0.5)
        
        # Also limit by total pixels
        total_pixels = num_frames * H * W
        max_pixels = 30_000_000
        if total_pixels * scale_factor**2 > max_pixels:
            scale_factor = (max_pixels / total_pixels) ** 0.5
            scale_factor = max(0.25, scale_factor)
        
        log(f"Using scale factor: {scale_factor:.2f}")
        
        try:
            # Prepare video: [1, T, H, W, C] normalized to [-1, 1]
            video = images.float()
            if video.max() > 1:
                video = video / 255.0 * 2 - 1
            else:
                video = video * 2 - 1
            
            # Downscale if needed
            if scale_factor < 1.0:
                new_H = int(H * scale_factor)
                new_W = int(W * scale_factor)
                video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
                video = F.interpolate(video, size=(new_H, new_W), mode='bilinear', align_corners=False)
                video = video.permute(0, 2, 3, 1)  # [T, new_H, new_W, C]
                log(f"Resized video to {new_W}x{new_H}")
            else:
                new_H, new_W = H, W
            
            video = video.unsqueeze(0).to(self.device)  # [1, T, H, W, C]
            
            # Prepare query points: [1, N, 3] where each point is [frame_idx, y, x]
            # TAPIR expects (frame, y, x) not (frame, x, y)!
            query_points = np.zeros((num_keypoints, 3), dtype=np.float32)
            query_points[:, 0] = detection_frame  # All from detection frame
            query_points[:, 1] = keypoints_2d[:, 1] * scale_factor  # y
            query_points[:, 2] = keypoints_2d[:, 0] * scale_factor  # x
            
            query_tensor = torch.tensor(query_points, dtype=torch.float32)
            query_tensor = query_tensor.unsqueeze(0).to(self.device)  # [1, N, 3]
            
            log(f"Running TAPIR: {num_frames} frames, {num_keypoints} points...")
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with torch.no_grad():
                outputs = self.tapir_model(video, query_tensor)
            
            # Extract results
            tracks = outputs['tracks'][0].cpu().numpy()  # [N, T, 2] - (x, y)
            occlusions = outputs['occlusion'][0]  # [N, T]
            expected_dist = outputs['expected_dist'][0]  # [N, T]
            
            # Scale tracks back to original resolution
            if scale_factor < 1.0:
                tracks = tracks / scale_factor
            
            # Compute visibility/confidence
            confidence = (1 - torch.sigmoid(occlusions)) * (1 - torch.sigmoid(expected_dist))
            confidence = confidence.cpu()  # [N, T]
            
            # Reshape tracks: [N, T, 2] -> [T, N, 2]
            tracks = np.transpose(tracks, (1, 0, 2))  # [T, N, 2]
            confidence = confidence.transpose(0, 1)  # [T, N]
            
            log(f"Tracking done: tracks {tracks.shape}, confidence {confidence.shape}")
            
            # Cleanup
            del video, query_tensor, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return tracks, confidence
            
        except torch.cuda.OutOfMemoryError:
            log("ERROR: GPU out of memory! Try 'quarter' resolution")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, None
        except Exception as e:
            log(f"ERROR in TAPIR tracking: {e}")
            import traceback
            traceback.print_exc()
            return None, None


NODE_CLASS_MAPPINGS = {
    "SAM3DBody2abc_Keypoint2DTracker": Keypoint2DTracker
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBody2abc_Keypoint2DTracker": "🎯 Keypoint 2D Tracker"
}
