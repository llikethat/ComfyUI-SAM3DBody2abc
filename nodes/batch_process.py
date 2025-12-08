"""
Batch Wrapper for SAM3DBody2abc
Processes video frames using SAM3DBody's native Process Image node internally.
"""

import os
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from scipy.ndimage import gaussian_filter1d


class BatchProcess:
    """
    Batch process video frames using SAM3DBody's Process Image internally.
    
    This node:
    1. Takes video frames (IMAGE batch)
    2. Loops through each frame
    3. Calls SAM3DBody Process Image for each
    4. Accumulates results with temporal smoothing
    5. Outputs FRAME_SEQUENCE ready for FBX export
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("SAM3D_MODEL", {
                    "tooltip": "SAM3DBody model from Load SAM3DBody Model node"
                }),
                "images": ("IMAGE", {
                    "tooltip": "Video frames (batch of images)"
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional mask for person segmentation"
                }),
                "bbox_threshold": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "inference_type": (["full", "body"], {
                    "default": "full",
                    "tooltip": "full: body+hands, body: body only (faster)"
                }),
                "smoothing_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Temporal smoothing (0=none, 1=moderate, 2=heavy)"
                }),
                "smoothing_radius": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 15,
                }),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                }),
                "end_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "tooltip": "-1 for all frames"
                }),
                "skip_frames": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 30,
                    "tooltip": "Process every Nth frame"
                }),
            }
        }
    
    RETURN_TYPES = ("FRAME_SEQUENCE", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("frame_sequence", "debug_images", "frame_count", "status")
    FUNCTION = "process_batch"
    CATEGORY = "SAM3DBody2abc"
    
    def _to_numpy(self, data):
        """Convert tensor to numpy."""
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        if isinstance(data, np.ndarray):
            return data.copy()
        return data
    
    def _apply_smoothing(self, frames: Dict, strength: float, radius: int) -> Dict:
        """Apply temporal smoothing to vertices and joints."""
        if strength <= 0 or len(frames) < 3:
            return frames
        
        sorted_indices = sorted(frames.keys())
        
        # Stack vertices
        vertices_stack = []
        joints_stack = []
        valid_indices = []
        
        for idx in sorted_indices:
            frame = frames[idx]
            if frame.get("valid") and frame.get("vertices") is not None:
                vertices_stack.append(frame["vertices"])
                if frame.get("joints") is not None:
                    joints_stack.append(frame["joints"])
                valid_indices.append(idx)
        
        if len(vertices_stack) < 3:
            return frames
        
        # Smooth vertices
        vertices_array = np.stack(vertices_stack, axis=0)
        sigma = strength * radius
        smoothed_vertices = gaussian_filter1d(vertices_array, sigma=sigma, axis=0, mode='nearest')
        
        # Smooth joints
        smoothed_joints = None
        if len(joints_stack) == len(vertices_stack):
            joints_array = np.stack(joints_stack, axis=0)
            smoothed_joints = gaussian_filter1d(joints_array, sigma=sigma, axis=0, mode='nearest')
        
        # Update frames
        smoothed_frames = dict(frames)
        for i, idx in enumerate(valid_indices):
            smoothed_frames[idx] = dict(frames[idx])
            smoothed_frames[idx]["vertices"] = smoothed_vertices[i]
            if smoothed_joints is not None:
                smoothed_frames[idx]["joints"] = smoothed_joints[i]
        
        return smoothed_frames
    
    def process_batch(
        self,
        model: Dict,
        images: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        bbox_threshold: float = 0.8,
        inference_type: str = "full",
        smoothing_strength: float = 0.5,
        smoothing_radius: int = 3,
        start_frame: int = 0,
        end_frame: int = -1,
        skip_frames: int = 1,
    ) -> Tuple[Dict, torch.Tensor, int, str]:
        """
        Process batch of video frames.
        """
        # Import SAM3DBody components
        try:
            from sam_3d_body import SAM3DBodyEstimator
        except ImportError:
            return ({}, images[:1], 0, "Error: SAM3DBody not installed")
        
        # Get frame count
        total_frames = images.shape[0]
        actual_end = total_frames if end_frame == -1 else min(end_frame + 1, total_frames)
        frame_indices = list(range(start_frame, actual_end, skip_frames))
        
        if not frame_indices:
            return ({}, images[:1], 0, "Error: No frames in range")
        
        print(f"[SAM3DBody2abc] Processing {len(frame_indices)} frames...")
        
        # Extract model components
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
        
        # Process frames
        frames = {}
        debug_images = []
        faces = None
        
        import tempfile
        import cv2
        
        for i, frame_idx in enumerate(frame_indices):
            try:
                # Get frame image
                img_tensor = images[frame_idx]
                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                if img_np.shape[-1] == 3:
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img_np
                
                # Get mask if provided
                frame_mask = None
                if mask is not None and frame_idx < mask.shape[0]:
                    mask_np = mask[frame_idx].cpu().numpy()
                    if mask_np.ndim == 2:
                        frame_mask = mask_np[..., np.newaxis]
                    else:
                        frame_mask = mask_np
                
                # Save temp image
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    cv2.imwrite(tmp.name, img_bgr)
                    tmp_path = tmp.name
                
                try:
                    # Process frame
                    outputs = estimator.process_one_image(
                        tmp_path,
                        bboxes=None,
                        masks=frame_mask,
                        use_mask=frame_mask is not None,
                        inference_type=inference_type,
                    )
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                
                if outputs and len(outputs) > 0:
                    output = outputs[0]
                    
                    # Store faces
                    if faces is None:
                        faces = self._to_numpy(estimator.faces)
                    
                    # Extract frame data
                    vertices = self._to_numpy(output.get("pred_vertices"))
                    joints = self._to_numpy(output.get("pred_keypoints_3d")) or \
                             self._to_numpy(output.get("pred_joint_coords"))
                    camera = self._to_numpy(output.get("pred_cam_t"))
                    focal = output.get("focal_length")
                    
                    if focal is not None:
                        if isinstance(focal, (torch.Tensor, np.ndarray)):
                            focal = float(np.array(focal).flatten()[0])
                    
                    frames[frame_idx] = {
                        "vertices": vertices,
                        "joints": joints,
                        "camera": camera,
                        "focal_length": focal,
                        "valid": vertices is not None,
                    }
                    
                    # Create debug image
                    debug_images.append(img_tensor)
                else:
                    frames[frame_idx] = {"valid": False}
                    debug_images.append(img_tensor)
                
                # Progress
                if (i + 1) % 10 == 0 or (i + 1) == len(frame_indices):
                    print(f"[SAM3DBody2abc] Processed {i + 1}/{len(frame_indices)} frames")
                    
            except Exception as e:
                print(f"[SAM3DBody2abc] Error processing frame {frame_idx}: {e}")
                frames[frame_idx] = {"valid": False}
                debug_images.append(images[frame_idx])
        
        # Apply temporal smoothing
        if smoothing_strength > 0:
            print(f"[SAM3DBody2abc] Applying smoothing (strength={smoothing_strength})...")
            frames = self._apply_smoothing(frames, smoothing_strength, smoothing_radius)
        
        # Build output
        frame_sequence = {
            "sequence_id": "batch_process",
            "frames": frames,
            "faces": faces,
            "smoothing_strength": smoothing_strength,
        }
        
        # Stack debug images
        if debug_images:
            debug_batch = torch.stack(debug_images, dim=0)
        else:
            debug_batch = images[:1]
        
        valid_count = sum(1 for f in frames.values() if f.get("valid", False))
        status = f"Processed {valid_count}/{len(frame_indices)} frames"
        
        print(f"[SAM3DBody2abc] {status}")
        
        return (frame_sequence, debug_batch, len(frame_indices), status)
