"""
Video Batch Processor for SAM3DBody2abc
Processes video frames using SAM3DBody and accumulates mesh_data.
"""

import os
import cv2
import tempfile
import numpy as np
import torch
from typing import Dict, Tuple, Any, Optional
from scipy.ndimage import gaussian_filter1d


def to_numpy(data):
    """Convert tensor to numpy."""
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    if isinstance(data, np.ndarray):
        return data.copy()
    return data


class VideoBatchProcessor:
    """
    Process video frames through SAM3DBody and collect mesh_data.
    
    For each frame, calls SAM3DBody internally and accumulates:
    - vertices (for shape keys in animated mesh)
    - joint_coords (for skeleton animation)
    - faces, joint_parents (stored once)
    
    Outputs MESH_SEQUENCE ready for animated FBX export.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("SAM3D_MODEL", {}),
                "images": ("IMAGE", {}),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Per-frame segmentation masks from SAM3 Propagation"
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
                }),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                }),
                "end_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                }),
                "skip_frames": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 30,
                }),
            }
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("mesh_sequence", "debug_images", "frame_count", "status")
    FUNCTION = "process_batch"
    CATEGORY = "SAM3DBody2abc"
    
    def _compute_bbox_from_mask(self, mask_np):
        """Compute bbox [x1,y1,x2,y2] from mask."""
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
        """Apply temporal smoothing to vertices and joints."""
        if strength <= 0 or len(frames) < 3:
            return frames
        
        sorted_indices = sorted(frames.keys())
        
        verts_stack, joints_stack, valid_indices = [], [], []
        
        for idx in sorted_indices:
            frame = frames[idx]
            v = frame.get("vertices")
            j = frame.get("joint_coords")
            if v is not None:
                verts_stack.append(v)
                if j is not None:
                    joints_stack.append(j)
                valid_indices.append(idx)
        
        if len(verts_stack) < 3:
            return frames
        
        sigma = strength * 3
        verts_array = np.stack(verts_stack, axis=0)
        smoothed_verts = gaussian_filter1d(verts_array, sigma=sigma, axis=0, mode='nearest')
        
        smoothed_joints = None
        if len(joints_stack) == len(verts_stack):
            joints_array = np.stack(joints_stack, axis=0)
            smoothed_joints = gaussian_filter1d(joints_array, sigma=sigma, axis=0, mode='nearest')
        
        smoothed_frames = dict(frames)
        for i, idx in enumerate(valid_indices):
            smoothed_frames[idx] = dict(frames[idx])
            smoothed_frames[idx]["vertices"] = smoothed_verts[i]
            if smoothed_joints is not None:
                smoothed_frames[idx]["joint_coords"] = smoothed_joints[i]
        
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
        """Process video frames.
        
        Uses per-frame masks from SAM3 Propagation for character tracking.
        Each frame gets its own mask, allowing accurate tracking as the character moves.
        """
        
        try:
            from sam_3d_body import SAM3DBodyEstimator
        except ImportError:
            return ({}, images[:1], 0, "Error: SAM3DBody not installed")
        
        total_frames = images.shape[0]
        actual_end = total_frames if end_frame == -1 else min(end_frame + 1, total_frames)
        frame_indices = list(range(start_frame, actual_end, skip_frames))
        
        if not frame_indices:
            return ({}, images[:1], 0, "Error: No frames")
        
        print(f"[SAM3DBody2abc] Processing {len(frame_indices)} frames...")
        if mask is not None:
            print(f"[SAM3DBody2abc] Using per-frame masks ({mask.shape[0]} masks available)")
        
        sam_3d_model = model["model"]
        model_cfg = model["model_cfg"]
        
        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=sam_3d_model,
            model_cfg=model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )
        
        # Storage
        frames = {}
        faces = None
        joint_parents = None
        mhr_path = model.get("mhr_path")
        debug_images = []
        
        # Disable autocast for entire processing to avoid BFloat16 sparse matrix errors
        # The MHR model uses sparse operations that don't support BFloat16 on CUDA
        with torch.cuda.amp.autocast(enabled=False):
            for i, frame_idx in enumerate(frame_indices):
                try:
                    img_tensor = images[frame_idx]
                    img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                    
                    if img_np.shape[-1] == 3:
                        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    else:
                        img_bgr = img_np
                    
                    # Get per-frame mask if available
                    frame_mask = None
                    frame_bbox = None
                    use_mask = False
                    
                    if mask is not None and frame_idx < mask.shape[0]:
                        mask_np = mask[frame_idx].cpu().numpy()
                        # SAM3DBody expects 2D mask
                        if mask_np.ndim == 3:
                            mask_np = mask_np[0]
                        
                        frame_bbox = self._compute_bbox_from_mask(mask_np)
                        if frame_bbox is not None:
                            frame_mask = mask_np
                            use_mask = True
                    
                    # Save temp image
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        cv2.imwrite(tmp.name, img_bgr)
                        tmp_path = tmp.name
                    
                    try:
                        with torch.no_grad():
                            outputs = estimator.process_one_image(
                                tmp_path,
                                bboxes=frame_bbox,
                                masks=frame_mask,
                                bbox_thr=bbox_threshold,
                                use_mask=use_mask,
                                inference_type=inference_type,
                            )
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                    
                    if outputs and len(outputs) > 0:
                        output = outputs[0]
                        
                        # Store faces (once)
                        if faces is None:
                            faces = to_numpy(estimator.faces)
                        
                        # Get joint parents (once)
                        if joint_parents is None:
                            try:
                                if hasattr(sam_3d_model, 'mhr_head') and hasattr(sam_3d_model.mhr_head, 'mhr'):
                                    mhr = sam_3d_model.mhr_head.mhr
                                    if hasattr(mhr, 'character_torch') and hasattr(mhr.character_torch, 'skeleton'):
                                        skel = mhr.character_torch.skeleton
                                        if hasattr(skel, 'joint_parents'):
                                            joint_parents = to_numpy(skel.joint_parents)
                            except Exception:
                                pass
                        
                        # Store frame data including camera
                        focal_length = output.get("focal_length")
                        if focal_length is not None and hasattr(focal_length, 'item'):
                            focal_length = float(focal_length.item()) if hasattr(focal_length, 'item') else float(focal_length)
                        
                        frames[frame_idx] = {
                            "vertices": to_numpy(output.get("pred_vertices")),
                            "joint_coords": to_numpy(output.get("pred_joint_coords")),
                            "pred_cam_t": to_numpy(output.get("pred_cam_t")),
                            "focal_length": focal_length,
                            "global_rot": to_numpy(output.get("global_rot")),
                        }
                        
                        debug_images.append(img_tensor)
                    else:
                        frames[frame_idx] = {"vertices": None, "joint_coords": None}
                        debug_images.append(img_tensor)
                    
                    if (i + 1) % 10 == 0 or (i + 1) == len(frame_indices):
                        print(f"[SAM3DBody2abc] Processed {i + 1}/{len(frame_indices)}")
                        
                except Exception as e:
                    print(f"[SAM3DBody2abc] Error frame {frame_idx}: {e}")
                    frames[frame_idx] = {"vertices": None, "joint_coords": None}
                    debug_images.append(images[frame_idx])
        
        # Apply smoothing
        if smoothing_strength > 0:
            print(f"[SAM3DBody2abc] Applying smoothing...")
            frames = self._apply_smoothing(frames, smoothing_strength)
        
        # Build output
        mesh_sequence = {
            "sequence_id": "batch",
            "frames": frames,
            "faces": faces,
            "joint_parents": joint_parents,
            "mhr_path": mhr_path,
        }
        
        if debug_images:
            debug_batch = torch.stack(debug_images, dim=0)
        else:
            debug_batch = images[:1]
        
        valid = sum(1 for f in frames.values() if f.get("vertices") is not None)
        status = f"Processed {valid}/{len(frame_indices)} frames"
        print(f"[SAM3DBody2abc] {status}")
        
        return (mesh_sequence, debug_batch, len(frame_indices), status)
