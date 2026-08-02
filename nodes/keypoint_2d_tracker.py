"""
Keypoint 2D Tracker for SAM3DBody2abc
=====================================
Version: 2.0.0

TAPIR-based 2D keypoint tracking. Takes initial keypoints from mesh_sequence
(frame 0) and tracks them across all frames.

This version does NOT load SAM3D - it uses keypoints already detected by
Video Batch Processor, avoiding BF16 conflicts.
"""

import os
import gc
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any


def log(msg):
    print(f"[Keypoint2DTracker] {msg}", flush=True)


# TAPIR Import
TAPIR_AVAILABLE = False
TAPIR_BACKEND = None

try:
    from tapnet.torch import tapir_model
    TAPIR_BACKEND = "tapnet"
    TAPIR_AVAILABLE = True
except ImportError:
    try:
        from tapir.torch import tapir_model
        TAPIR_BACKEND = "tapir"
        TAPIR_AVAILABLE = True
    except ImportError:
        pass


class Keypoint2DTracker:
    """
    Track 2D keypoints using TAPIR.
    
    Takes initial keypoints from mesh_sequence (detected by Video Processor)
    and tracks them across all frames.
    
    Workflow:
        [Video Processor] → mesh_sequence (with pred_keypoints_2d)
              ↓
        [Keypoint2DTracker] → tracked_keypoints_2d
              ↓
        [Video Processor] (2nd pass with tracked keypoints)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mesh_sequence": ("MESH_SEQUENCE",),
            },
            "optional": {
                "detection_frame": ("INT", {
                    "default": 0, 
                    "min": 0,
                    "tooltip": "Frame to get initial keypoints from (usually 0)"
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

    RETURN_TYPES = ("KEYPOINTS_2D", "TENSOR", "STRING")
    RETURN_NAMES = ("tracked_keypoints_2d", "tracking_confidence", "status")
    FUNCTION = "process"
    CATEGORY = "SAM3DBody2abc/Tracking"

    def __init__(self):
        self.tapir_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def process(
        self,
        images: torch.Tensor,
        mesh_sequence: Dict,
        detection_frame: int = 0,
        tapir_checkpoint: str = "",
        tracking_resolution: str = "half",
    ) -> Tuple[Dict, torch.Tensor, str]:
        
        log("=" * 60)
        log("Keypoint 2D Tracker v2.0 (TAPIR-only)")
        log("=" * 60)
        
        # Get dimensions
        num_frames = images.shape[0]
        H, W = images.shape[1], images.shape[2]
        log(f"Input: {num_frames} frames, {W}x{H}")
        
        # =====================================================================
        # Step 1: Extract 2D keypoints from mesh_sequence
        # =====================================================================
        log(f"Step 1: Extracting keypoints from mesh_sequence frame {detection_frame}...")
        
        keypoints_2d = self._extract_keypoints_from_mesh(mesh_sequence, detection_frame)
        
        if keypoints_2d is None:
            log("ERROR: Could not extract keypoints from mesh_sequence!")
            log("Make sure Video Processor runs BEFORE this node.")
            empty_kp = {
                "keypoints": np.zeros((num_frames, 70, 2), dtype=np.float32),
                "num_frames": num_frames,
                "num_keypoints": 70,
                "detection_frame": detection_frame,
            }
            empty_conf = torch.zeros(num_frames, 70)
            return (empty_kp, empty_conf, "Failed: No keypoints in mesh_sequence")
        
        num_keypoints = keypoints_2d.shape[0]
        log(f"Extracted {num_keypoints} keypoints from frame {detection_frame}")
        log(f"Keypoint range: x=[{keypoints_2d[:,0].min():.1f}, {keypoints_2d[:,0].max():.1f}], "
            f"y=[{keypoints_2d[:,1].min():.1f}, {keypoints_2d[:,1].max():.1f}]")
        
        # =====================================================================
        # Step 2: Load TAPIR and track
        # =====================================================================
        log(f"Step 2: Loading TAPIR model...")
        
        if not self._load_tapir(tapir_checkpoint):
            log("ERROR: TAPIR not available, using static keypoints")
            static_kp = np.tile(keypoints_2d[np.newaxis, :, :], (num_frames, 1, 1))
            output = {
                "keypoints": static_kp,
                "num_frames": num_frames,
                "num_keypoints": num_keypoints,
                "detection_frame": detection_frame,
            }
            confidence = torch.ones(num_frames, num_keypoints)
            return (output, confidence, "TAPIR unavailable - using static keypoints")
        
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
            return (output, conf, "TAPIR tracking failed - using static keypoints")
        
        log(f"Tracking complete: {tracks.shape}")
        
        # =====================================================================
        # Step 3: Package output
        # =====================================================================
        output = {
            "keypoints": tracks,  # (N, num_keypoints, 2)
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
        
        return (output, confidence, status)

    def _extract_keypoints_from_mesh(
        self,
        mesh_sequence: Dict,
        frame_idx: int,
    ) -> Optional[np.ndarray]:
        """Extract 2D keypoints from mesh_sequence."""
        
        try:
            frames = mesh_sequence.get("frames", {})
            
            if not frames:
                log("mesh_sequence has no frames")
                return None
            
            # Get available frame indices
            frame_keys = sorted(frames.keys())
            log(f"Available frames in mesh_sequence: {len(frame_keys)}")
            
            if frame_idx not in frames:
                # Try to find closest frame
                if frame_keys:
                    frame_idx = frame_keys[0]
                    log(f"Frame {frame_idx} not found, using first available: {frame_idx}")
                else:
                    return None
            
            frame_data = frames[frame_idx]
            
            # Get pred_keypoints_2d
            keypoints_2d = frame_data.get("pred_keypoints_2d")
            
            if keypoints_2d is None:
                log(f"No pred_keypoints_2d in frame {frame_idx}")
                log(f"Available keys: {list(frame_data.keys())}")
                return None
            
            if isinstance(keypoints_2d, torch.Tensor):
                keypoints_2d = keypoints_2d.cpu().numpy()
            
            keypoints_2d = np.array(keypoints_2d, dtype=np.float32)
            
            if keypoints_2d.ndim != 2 or keypoints_2d.shape[1] != 2:
                log(f"Invalid keypoints shape: {keypoints_2d.shape}")
                return None
            
            return keypoints_2d
            
        except Exception as e:
            log(f"Error extracting keypoints: {e}")
            import traceback
            traceback.print_exc()
            return None

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
            "/home/burny/ComfyUI",
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
            self.tapir_model = tapir_model.TAPIR(pyramid_level=1)
            self.tapir_model.load_state_dict(torch.load(checkpoint, map_location=self.device))
            self.tapir_model = self.tapir_model.to(self.device)
            self.tapir_model.eval()
            log(f"TAPIR loaded successfully ({TAPIR_BACKEND} backend)")
            return True
        except Exception as e:
            log(f"ERROR loading TAPIR: {e}")
            import traceback
            traceback.print_exc()
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
