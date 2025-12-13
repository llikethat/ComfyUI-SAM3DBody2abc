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
                "sam3_masks": ("SAM3_VIDEO_MASKS", {
                    "tooltip": "Masks from SAM3 Propagation"
                }),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.01,
                    "tooltip": "Source video FPS - passed through to export"
                }),
                "object_id": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 10,
                    "tooltip": "Which object ID to track (match obj_id in SAM3 Video Output)"
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
    
    RETURN_TYPES = ("MESH_SEQUENCE", "IMAGE", "INT", "STRING", "FLOAT")
    RETURN_NAMES = ("mesh_sequence", "debug_images", "frame_count", "status", "fps")
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
        sam3_masks: Optional[Any] = None,
        fps: float = 24.0,
        object_id: int = 1,
        bbox_threshold: float = 0.8,
        inference_type: str = "full",
        smoothing_strength: float = 0.5,
        start_frame: int = 0,
        end_frame: int = -1,
        skip_frames: int = 1,
    ) -> Tuple[Dict, torch.Tensor, int, str, float]:
        """Process video frames.
        
        Uses per-frame masks from SAM3 Propagation for character tracking.
        Each frame gets its own mask, allowing accurate tracking as the character moves.
        """
        
        try:
            from sam_3d_body import SAM3DBodyEstimator
        except ImportError:
            return ({}, images[:1], 0, "Error: SAM3DBody not installed", fps)
        
        # Handle sam3_masks input - extract masks for tracking
        active_mask = None
        if sam3_masks is not None:
            print(f"[SAM3DBody2abc] Received sam3_masks, type: {type(sam3_masks).__name__}")
            
            # Debug: inspect the structure (limit output)
            if isinstance(sam3_masks, dict):
                keys = list(sam3_masks.keys())
                print(f"[SAM3DBody2abc] sam3_masks is dict with {len(keys)} keys: {keys[:5]}...{keys[-3:] if len(keys) > 5 else ''}")
                
                # Check if this is a frame-indexed dict (keys are frame indices 0,1,2,...)
                # SAM3 Propagate outputs: {0: mask_frame0, 1: mask_frame1, ...}
                is_frame_indexed = False
                if len(keys) > 1:
                    # Check if keys are consecutive integers starting from 0
                    int_keys = [k for k in keys if isinstance(k, int)]
                    if len(int_keys) == len(keys):
                        sorted_keys = sorted(int_keys)
                        if sorted_keys == list(range(len(sorted_keys))):
                            is_frame_indexed = True
                            print(f"[SAM3DBody2abc] Detected frame-indexed dict format ({len(keys)} frames)")
                
                if is_frame_indexed:
                    # Stack all frames into a single tensor
                    # Each value is shape (1, H, W) or (num_objects, H, W)
                    frame_masks = []
                    sorted_keys = sorted([k for k in keys if isinstance(k, int)])
                    
                    for frame_key in sorted_keys:
                        frame_mask = sam3_masks[frame_key]
                        # Convert to tensor if numpy
                        if isinstance(frame_mask, np.ndarray):
                            frame_mask = torch.from_numpy(frame_mask)
                        elif hasattr(frame_mask, 'cpu'):
                            frame_mask = frame_mask.cpu()
                        frame_masks.append(frame_mask)
                    
                    # Stack: result shape will be (num_frames, num_objects, H, W) or (num_frames, H, W)
                    stacked = torch.stack(frame_masks, dim=0)
                    print(f"[SAM3DBody2abc] Stacked {len(frame_masks)} frames into shape {stacked.shape}")
                    
                    # If shape is (frames, 1, H, W), squeeze the object dim
                    if stacked.ndim == 4 and stacked.shape[1] == 1:
                        stacked = stacked.squeeze(1)  # (frames, H, W)
                        print(f"[SAM3DBody2abc] Squeezed to shape {stacked.shape}")
                    
                    active_mask = stacked
                
                elif 'masks' in sam3_masks:
                    # Nested structure: {'masks': tensor, 'scores': tensor, ...}
                    inner_masks = sam3_masks['masks']
                    if hasattr(inner_masks, 'shape'):
                        print(f"[SAM3DBody2abc] Found nested 'masks' key with shape {inner_masks.shape}")
                        if inner_masks.ndim >= 3:
                            active_mask = inner_masks
                
                # Object-indexed format: {obj_id: mask_tensor}
                elif object_id in sam3_masks:
                    active_mask = sam3_masks[object_id]
                    print(f"[SAM3DBody2abc] Extracted mask for object {object_id} from dict")
                elif str(object_id) in sam3_masks:
                    active_mask = sam3_masks[str(object_id)]
                    print(f"[SAM3DBody2abc] Extracted mask for object '{object_id}' from dict (string key)")
                else:
                    print(f"[SAM3DBody2abc] Warning: Could not determine mask format. Keys sample: {keys[:5]}")
            
            elif isinstance(sam3_masks, (list, tuple)):
                # List format: [mask_for_obj_0, mask_for_obj_1, ...]
                if object_id < len(sam3_masks):
                    active_mask = sam3_masks[object_id]
                    if hasattr(active_mask, 'shape'):
                        print(f"[SAM3DBody2abc] Extracted mask[{object_id}] from list -> shape {active_mask.shape}")
                elif len(sam3_masks) > 0:
                    active_mask = sam3_masks[0]
                    print(f"[SAM3DBody2abc] object_id {object_id} >= len, using first mask")
            elif isinstance(sam3_masks, torch.Tensor):
                print(f"[SAM3DBody2abc] masks tensor shape: {sam3_masks.shape}, ndim: {sam3_masks.ndim}")
                # Try to extract the right object
                if sam3_masks.ndim == 4:
                    # Could be [frames, objects, H, W] or [objects, frames, H, W]
                    if sam3_masks.shape[1] < sam3_masks.shape[0]:
                        # Likely [frames, objects, H, W]
                        if object_id < sam3_masks.shape[1]:
                            active_mask = sam3_masks[:, object_id, :, :]
                            print(f"[SAM3DBody2abc] Extracted mask[:, {object_id}, :, :] -> shape {active_mask.shape}")
                    else:
                        # Likely [objects, frames, H, W]
                        if object_id < sam3_masks.shape[0]:
                            active_mask = sam3_masks[object_id, :, :, :]
                            print(f"[SAM3DBody2abc] Extracted mask[{object_id}, :, :, :] -> shape {active_mask.shape}")
                elif sam3_masks.ndim == 3:
                    # [frames, H, W] - single object already
                    active_mask = sam3_masks
                    print(f"[SAM3DBody2abc] Using masks directly (single object): shape {active_mask.shape}")
            elif hasattr(sam3_masks, 'masks'):
                # Some custom object with .masks attribute
                actual_masks = sam3_masks.masks
                print(f"[SAM3DBody2abc] Found .masks attribute, type: {type(actual_masks).__name__}")
                if isinstance(actual_masks, torch.Tensor) and actual_masks.ndim >= 3:
                    active_mask = actual_masks
        
        if active_mask is not None:
            if hasattr(active_mask, 'shape'):
                print(f"[SAM3DBody2abc] Using mask with shape: {active_mask.shape}")
            else:
                print(f"[SAM3DBody2abc] Using mask of type: {type(active_mask).__name__}")
        else:
            print(f"[SAM3DBody2abc] No mask provided - will use auto-detection")
        
        total_frames = images.shape[0]
        actual_end = total_frames if end_frame == -1 else min(end_frame + 1, total_frames)
        frame_indices = list(range(start_frame, actual_end, skip_frames))
        
        if not frame_indices:
            return ({}, images[:1], 0, "Error: No frames")
        
        print(f"[SAM3DBody2abc] Processing {len(frame_indices)} frames...")
        if active_mask is not None:
            mask_frames = active_mask.shape[0] if hasattr(active_mask, 'shape') and active_mask.ndim >= 1 else 1
            print(f"[SAM3DBody2abc] Using per-frame masks ({mask_frames} masks available)")
        
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
                    
                    if active_mask is not None:
                        # Determine which mask frame to use
                        mask_idx = frame_idx
                        if hasattr(active_mask, 'shape'):
                            if active_mask.ndim >= 1 and mask_idx >= active_mask.shape[0]:
                                mask_idx = active_mask.shape[0] - 1  # Clamp to last available
                        
                        if hasattr(active_mask, 'shape') and active_mask.ndim >= 2:
                            mask_np = active_mask[mask_idx].cpu().numpy() if hasattr(active_mask, 'cpu') else np.array(active_mask[mask_idx])
                        else:
                            mask_np = active_mask.cpu().numpy() if hasattr(active_mask, 'cpu') else np.array(active_mask)
                        
                        # SAM3DBody expects 2D mask
                        if mask_np.ndim == 3:
                            mask_np = mask_np[0] if mask_np.shape[0] == 1 else mask_np[:, :, 0]
                        
                        frame_bbox = self._compute_bbox_from_mask(mask_np)
                        if frame_bbox is not None:
                            frame_mask = mask_np
                            use_mask = True
                            if i == 0:
                                print(f"[SAM3DBody2abc] Frame 0 mask shape: {mask_np.shape}, bbox: {frame_bbox}")
                    
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
                        # If multiple detections and we have a mask, select the one that best matches
                        output = outputs[0]
                        
                        if len(outputs) > 1 and frame_mask is not None:
                            best_overlap = 0
                            best_idx = 0
                            
                            for out_idx, out in enumerate(outputs):
                                out_bbox = out.get("bbox")
                                if out_bbox is not None:
                                    # Compute overlap between detection bbox and mask
                                    bbox_arr = np.array(out_bbox).flatten()
                                    if len(bbox_arr) >= 4:
                                        x1, y1, x2, y2 = int(bbox_arr[0]), int(bbox_arr[1]), int(bbox_arr[2]), int(bbox_arr[3])
                                        # Clamp to image bounds
                                        h_mask, w_mask = frame_mask.shape[:2]
                                        x1, y1 = max(0, x1), max(0, y1)
                                        x2, y2 = min(w_mask, x2), min(h_mask, y2)
                                        
                                        if x2 > x1 and y2 > y1:
                                            # Count mask pixels in bbox region
                                            bbox_mask = frame_mask[y1:y2, x1:x2]
                                            overlap = np.sum(bbox_mask > 0.5)
                                            
                                            if overlap > best_overlap:
                                                best_overlap = overlap
                                                best_idx = out_idx
                            
                            output = outputs[best_idx]
                            if i == 0 and len(outputs) > 1:
                                print(f"[SAM3DBody2abc] Multiple detections ({len(outputs)}), selected #{best_idx} with mask overlap {best_overlap}")
                        
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
                                            print(f"[SAM3DBody2abc] Got joint_parents from mhr.character_torch.skeleton: {joint_parents.shape if hasattr(joint_parents, 'shape') else len(joint_parents)} joints")
                                            print(f"[SAM3DBody2abc] First 10 parent indices: {joint_parents[:10] if joint_parents is not None else 'None'}")
                            except Exception as e:
                                print(f"[SAM3DBody2abc] Error getting joint_parents: {e}")
                        
                        # Store frame data including camera and rotations
                        focal_length = output.get("focal_length")
                        if focal_length is not None and hasattr(focal_length, 'item'):
                            focal_length = float(focal_length.item()) if hasattr(focal_length, 'item') else float(focal_length)
                        
                        # Debug first frame
                        if i == 0:
                            print(f"[SAM3DBody2abc] First frame output keys: {list(output.keys())}")
                            pred_global_rots = output.get("pred_global_rots")
                            if pred_global_rots is not None:
                                if hasattr(pred_global_rots, 'shape'):
                                    print(f"[SAM3DBody2abc] pred_global_rots shape: {pred_global_rots.shape}")
                            else:
                                print(f"[SAM3DBody2abc] WARNING: pred_global_rots is None!")
                            
                            pred_kp_2d = output.get("pred_keypoints_2d")
                            if pred_kp_2d is not None:
                                if hasattr(pred_kp_2d, 'shape'):
                                    print(f"[SAM3DBody2abc] pred_keypoints_2d shape: {pred_kp_2d.shape}")
                                    # Print first few 2D keypoints to verify they're in image space
                                    kp_np = pred_kp_2d.cpu().numpy() if hasattr(pred_kp_2d, 'cpu') else np.array(pred_kp_2d)
                                    print(f"[SAM3DBody2abc] First 5 2D keypoints: {kp_np[:5]}")
                            else:
                                print(f"[SAM3DBody2abc] WARNING: pred_keypoints_2d is None!")
                            
                            bbox = output.get("bbox")
                            if bbox is not None:
                                print(f"[SAM3DBody2abc] Detection bbox: {bbox}")
                        
                        frames[frame_idx] = {
                            "vertices": to_numpy(output.get("pred_vertices")),
                            "joint_coords": to_numpy(output.get("pred_joint_coords")),
                            "joint_rotations": to_numpy(output.get("pred_global_rots")),  # Per-joint rotations!
                            "pred_keypoints_2d": to_numpy(output.get("pred_keypoints_2d")),  # 2D keypoints for overlay
                            "pred_cam_t": to_numpy(output.get("pred_cam_t")),
                            "focal_length": focal_length,
                            "global_rot": to_numpy(output.get("global_rot")),  # Keep for compatibility
                            "bbox": to_numpy(output.get("bbox")),  # Store bbox for debugging
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
        
        # Build output - include fps for downstream nodes
        mesh_sequence = {
            "sequence_id": "batch",
            "frames": frames,
            "faces": faces,
            "joint_parents": joint_parents,
            "mhr_path": mhr_path,
            "fps": fps,  # Pass through for export
        }
        
        if debug_images:
            debug_batch = torch.stack(debug_images, dim=0)
        else:
            debug_batch = images[:1]
        
        valid = sum(1 for f in frames.values() if f.get("vertices") is not None)
        status = f"Processed {valid}/{len(frame_indices)} frames at {fps} fps"
        print(f"[SAM3DBody2abc] {status}")
        
        return (mesh_sequence, debug_batch, len(frame_indices), status, fps)
