"""
Video Batch Processor for SAM3DBody2abc
Processes video frames using SAM3DBody and accumulates skeleton data.
"""

import os
import cv2
import tempfile
import numpy as np
import torch
from typing import Dict, Tuple, Any, Optional
from scipy.ndimage import gaussian_filter1d


class VideoBatchProcessor:
    """
    Process video frames through SAM3DBody and collect skeleton data.
    
    This node:
    1. Takes video frames (IMAGE batch)
    2. Processes each frame with SAM3DBody
    3. Accumulates skeleton data with optional temporal smoothing
    4. Outputs SKELETON_SEQUENCE ready for FBX export
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("SAM3D_MODEL", {
                    "tooltip": "SAM3DBody model from Load node"
                }),
                "images": ("IMAGE", {
                    "tooltip": "Video frames (batch)"
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional segmentation mask"
                }),
                "bbox_threshold": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "inference_type": (["full", "body"], {
                    "default": "full",
                }),
                "smoothing_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Temporal smoothing (0=none)"
                }),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                }),
                "end_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "-1 for all frames"
                }),
                "skip_frames": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 30,
                }),
            }
        }
    
    RETURN_TYPES = ("SKELETON_SEQUENCE", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("skeleton_sequence", "debug_images", "frame_count", "status")
    FUNCTION = "process_batch"
    CATEGORY = "SAM3DBody2abc"
    
    def _to_numpy(self, data):
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        if isinstance(data, np.ndarray):
            return data.copy()
        return data
    
    def _compute_bbox_from_mask(self, mask_np):
        """Compute bbox from mask."""
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        
        rows = np.any(mask_np > 0.5, axis=1)
        cols = np.any(mask_np > 0.5, axis=0)
        
        if not rows.any() or not cols.any():
            return None
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return np.array([[cmin, rmin, cmax, rmax]], dtype=np.float32)
    
    def _apply_smoothing(self, frames: Dict, strength: float) -> Dict:
        """Apply temporal smoothing to joint positions."""
        if strength <= 0 or len(frames) < 3:
            return frames
        
        sorted_indices = sorted(frames.keys())
        
        # Stack joint positions
        positions_stack = []
        valid_indices = []
        
        for idx in sorted_indices:
            frame = frames[idx]
            pos = frame.get("joint_positions")
            if pos is not None:
                positions_stack.append(pos)
                valid_indices.append(idx)
        
        if len(positions_stack) < 3:
            return frames
        
        # Smooth
        positions_array = np.stack(positions_stack, axis=0)
        sigma = strength * 3
        smoothed = gaussian_filter1d(positions_array, sigma=sigma, axis=0, mode='nearest')
        
        # Update frames
        smoothed_frames = dict(frames)
        for i, idx in enumerate(valid_indices):
            smoothed_frames[idx] = dict(frames[idx])
            smoothed_frames[idx]["joint_positions"] = smoothed[i]
        
        return smoothed_frames
    
    def process_batch(
        self,
        model: Dict,
        images: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        bbox_threshold: float = 0.8,
        inference_type: str = "full",
        smoothing_strength: float = 0.5,
        start_frame: int = 0,
        end_frame: int = -1,
        skip_frames: int = 1,
    ) -> Tuple[Dict, torch.Tensor, int, str]:
        """Process video frames."""
        
        # Import SAM3DBody
        try:
            from sam_3d_body import SAM3DBodyEstimator
        except ImportError:
            return ({}, images[:1], 0, "Error: SAM3DBody not installed")
        
        # Frame range
        total_frames = images.shape[0]
        actual_end = total_frames if end_frame == -1 else min(end_frame + 1, total_frames)
        frame_indices = list(range(start_frame, actual_end, skip_frames))
        
        if not frame_indices:
            return ({}, images[:1], 0, "Error: No frames in range")
        
        print(f"[SAM3DBody2abc] Processing {len(frame_indices)} frames...")
        
        # Extract model
        sam_3d_model = model["model"]
        model_cfg = model["model_cfg"]
        
        # Create estimator
        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=sam_3d_model,
            model_cfg=model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )
        
        # Get joint hierarchy
        joint_parents = None
        joint_names = None
        try:
            if hasattr(sam_3d_model, 'mhr_head') and hasattr(sam_3d_model.mhr_head, 'mhr'):
                mhr = sam_3d_model.mhr_head.mhr
                if hasattr(mhr, 'character_torch') and hasattr(mhr.character_torch, 'skeleton'):
                    skel = mhr.character_torch.skeleton
                    if hasattr(skel, 'joint_parents'):
                        joint_parents = self._to_numpy(skel.joint_parents)
                    if hasattr(skel, 'joint_names'):
                        joint_names = skel.joint_names
        except Exception as e:
            print(f"[SAM3DBody2abc] Could not get joint hierarchy: {e}")
        
        # Process frames
        frames = {}
        debug_images = []
        
        for i, frame_idx in enumerate(frame_indices):
            try:
                # Get frame
                img_tensor = images[frame_idx]
                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                
                if img_np.shape[-1] == 3:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img_np
                
                # Handle mask
                frame_mask = None
                frame_bbox = None
                use_mask = False
                
                if mask is not None and frame_idx < mask.shape[0]:
                    mask_np = mask[frame_idx].cpu().numpy()
                    if mask_np.ndim == 2:
                        frame_mask = mask_np[..., np.newaxis]
                    else:
                        frame_mask = mask_np
                    
                    frame_bbox = self._compute_bbox_from_mask(mask_np)
                    if frame_bbox is not None:
                        use_mask = True
                    else:
                        frame_mask = None
                
                # Save temp image
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    cv2.imwrite(tmp.name, img_bgr)
                    tmp_path = tmp.name
                
                try:
                    # Process
                    outputs = estimator.process_one_image(
                        tmp_path,
                        bboxes=frame_bbox,
                        masks=frame_mask,
                        use_mask=use_mask,
                        inference_type=inference_type,
                    )
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                
                if outputs and len(outputs) > 0:
                    output = outputs[0]
                    
                    # Store skeleton data
                    frames[frame_idx] = {
                        "joint_positions": self._to_numpy(output.get("pred_joint_coords")),
                        "joint_rotations": self._to_numpy(output.get("pred_global_rots")),
                        "camera": self._to_numpy(output.get("pred_cam_t")),
                        "focal_length": output.get("focal_length"),
                        "global_rot": self._to_numpy(output.get("global_rot")),
                    }
                    
                    debug_images.append(img_tensor)
                else:
                    frames[frame_idx] = {"joint_positions": None}
                    debug_images.append(img_tensor)
                
                if (i + 1) % 10 == 0 or (i + 1) == len(frame_indices):
                    print(f"[SAM3DBody2abc] Processed {i + 1}/{len(frame_indices)}")
                    
            except Exception as e:
                print(f"[SAM3DBody2abc] Error frame {frame_idx}: {e}")
                frames[frame_idx] = {"joint_positions": None}
                debug_images.append(images[frame_idx])
        
        # Apply smoothing
        if smoothing_strength > 0:
            print(f"[SAM3DBody2abc] Applying smoothing (strength={smoothing_strength})...")
            frames = self._apply_smoothing(frames, smoothing_strength)
        
        # Build output
        skeleton_sequence = {
            "sequence_id": "batch_process",
            "frames": frames,
            "joint_parents": joint_parents,
            "joint_names": joint_names,
        }
        
        # Stack debug images
        if debug_images:
            debug_batch = torch.stack(debug_images, dim=0)
        else:
            debug_batch = images[:1]
        
        valid_count = sum(1 for f in frames.values() if f.get("joint_positions") is not None)
        status = f"Processed {valid_count}/{len(frame_indices)} frames"
        
        print(f"[SAM3DBody2abc] {status}")
        
        return (skeleton_sequence, debug_batch, len(frame_indices), status)
