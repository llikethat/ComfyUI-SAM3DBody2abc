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

# Import logger
try:
    from ..lib.logger import log, set_module
    set_module("Video Processor")
except ImportError:
    class _FallbackLog:
        def info(self, msg): print(f"[Video Processor] {msg}")
        def debug(self, msg): pass
        def warn(self, msg): print(f"[Video Processor] WARN: {msg}")
        def error(self, msg): print(f"[Video Processor] ERROR: {msg}")
        def progress(self, c, t, task="", interval=10): 
            if c == 0 or c == t - 1 or (c + 1) % interval == 0:
                print(f"[Video Processor] {task}: {c + 1}/{t}")
    log = _FallbackLog()


def to_numpy(data):
    """Convert tensor to numpy."""
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    if isinstance(data, np.ndarray):
        return data.copy()
    return data


def normalize_intrinsics(intrinsics: Optional[Dict], frame_count: int) -> Optional[Dict]:
    """
    Normalize any intrinsics format to a standard internal format.
    
    Supports multiple input formats:
    
    Format 1 - Simple dict (single focal length):
        {"focal_length": 1108.5, "cx": 640, "cy": 360, "width": 1280, "height": 720}
    
    Format 2 - Per-frame (zoom lenses / externally solved):
        {"per_frame": True, "frames": [{"focal_length": 1100}, ...], "width": 1280, "height": 720}
    
    Format 3 - MoGe2 format:
        {"focal_length": 1108.5, "per_frame_focal": [...], "cx": 640, "cy": 360, ...}
    
    Format 4 - Calibration format:
        {"fx": 1108.5, "fy": 1108.5, "cx": 640, "cy": 360, ...}
    
    Returns normalized format:
        {
            "focal_length": float,           # Primary focal length (frame 0 or average)
            "per_frame_focal": [float, ...], # Per-frame focal lengths (same length as frame_count)
            "cx": float,                     # Principal point X
            "cy": float,                     # Principal point Y
            "width": int,                    # Image width
            "height": int,                   # Image height
            "source": str,                   # Source identifier
        }
    """
    if intrinsics is None:
        return None
    
    normalized = {
        "source": "external",
        "cx": None,
        "cy": None,
        "width": None,
        "height": None,
    }
    
    # Extract image dimensions
    normalized["width"] = intrinsics.get("width") or intrinsics.get("image_width")
    normalized["height"] = intrinsics.get("height") or intrinsics.get("image_height")
    
    # Extract principal point (try multiple key names)
    normalized["cx"] = (
        intrinsics.get("cx") or 
        intrinsics.get("principal_point_x") or 
        intrinsics.get("principal_x")
    )
    normalized["cy"] = (
        intrinsics.get("cy") or 
        intrinsics.get("principal_point_y") or 
        intrinsics.get("principal_y")
    )
    
    # Extract focal length (try multiple key names)
    # INTRINSICS format uses "focal_px", CAMERA_INTRINSICS uses "focal_length"
    focal = (
        intrinsics.get("focal_length") or
        intrinsics.get("focal_length_px") or
        intrinsics.get("focal_px") or  # INTRINSICS format from IntrinsicsFromJSON
        intrinsics.get("focal") or
        intrinsics.get("fx")  # Calibration format
    )
    
    # Handle per-frame focal lengths
    per_frame_focal = None
    
    if intrinsics.get("per_frame"):
        # Format 2: Explicit per-frame format
        frames_data = intrinsics.get("frames", [])
        if frames_data:
            per_frame_focal = []
            for f in frames_data:
                f_focal = f.get("focal_length") or f.get("focal") or f.get("fx") or focal
                per_frame_focal.append(float(f_focal) if f_focal else None)
            # Use first frame as primary
            if per_frame_focal and per_frame_focal[0] is not None:
                focal = per_frame_focal[0]
    
    elif "per_frame_focal" in intrinsics:
        # Format 3: MoGe2 style
        per_frame_focal = intrinsics["per_frame_focal"]
        if isinstance(per_frame_focal, (list, np.ndarray)) and len(per_frame_focal) > 0:
            # Use first frame as primary focal if not already set
            if focal is None:
                focal = per_frame_focal[0]
    
    # Convert focal to float
    if focal is not None:
        if hasattr(focal, 'item'):
            focal = focal.item()
        focal = float(focal)
    
    normalized["focal_length"] = focal
    
    # Expand per_frame_focal to match frame_count if needed
    if per_frame_focal is not None:
        per_frame_focal = list(per_frame_focal)
        if len(per_frame_focal) < frame_count:
            # Pad with last value
            last_val = per_frame_focal[-1] if per_frame_focal else focal
            per_frame_focal.extend([last_val] * (frame_count - len(per_frame_focal)))
        elif len(per_frame_focal) > frame_count:
            # Truncate
            per_frame_focal = per_frame_focal[:frame_count]
        normalized["per_frame_focal"] = per_frame_focal
    elif focal is not None:
        # Single focal - expand to all frames
        normalized["per_frame_focal"] = [focal] * frame_count
    
    # Set source identifier
    if "source" in intrinsics:
        normalized["source"] = intrinsics["source"]
    elif "moge" in str(intrinsics.get("type", "")).lower():
        normalized["source"] = "MoGe2"
    elif intrinsics.get("per_frame"):
        normalized["source"] = "external_per_frame"
    else:
        normalized["source"] = "external"
    
    # Log what we got
    log.info(f"Normalized intrinsics from {normalized['source']}:")
    log.info(f"  Focal length: {normalized['focal_length']}")
    if normalized.get('per_frame_focal'):
        focal_range = (min(normalized['per_frame_focal']), max(normalized['per_frame_focal']))
        if focal_range[0] != focal_range[1]:
            log.info(f"  Per-frame focal range: {focal_range[0]:.1f} - {focal_range[1]:.1f}px")
    if normalized['cx'] is not None:
        log.info(f"  Principal point: ({normalized['cx']}, {normalized['cy']})")
    
    return normalized


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
                "masks": ("MASK", {
                    "tooltip": "Masks from SAM3 Video Output or any mask source. Shape: (N, H, W) or (N, 1, H, W)"
                }),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.01,
                    "tooltip": "Source video FPS - passed through to export"
                }),
                "object_id": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "tooltip": "Which object ID to use if masks have multiple objects (usually 0)"
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
                "batch_size": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "tooltip": "Frames to process per batch. Lower = less VRAM, Higher = faster."
                }),
                "external_intrinsics": ("CAMERA_INTRINSICS", {
                    "tooltip": "External camera intrinsics from MoGe2 (overrides SAM3DBody estimation)."
                }),
                "intrinsics_json": ("INTRINSICS", {
                    "tooltip": "Camera intrinsics from JSON file or IntrinsicsEstimator (overrides SAM3DBody estimation)."
                }),
            }
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE", "IMAGE", "INT", "STRING", "FLOAT", "FLOAT")
    RETURN_NAMES = ("mesh_sequence", "debug_images", "frame_count", "status", "fps", "focal_length_px")
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
        masks: Optional[torch.Tensor] = None,
        fps: float = 24.0,
        object_id: int = 0,
        bbox_threshold: float = 0.8,
        inference_type: str = "full",
        smoothing_strength: float = 0.5,
        start_frame: int = 0,
        end_frame: int = -1,
        skip_frames: int = 1,
        batch_size: int = 10,
        external_intrinsics: Optional[Dict] = None,
        intrinsics_json: Optional[Dict] = None,
    ) -> Tuple[Dict, torch.Tensor, int, str, float]:
        """Process video frames.
        
        Uses per-frame masks from SAM3 for character tracking.
        Each frame gets its own mask, allowing accurate tracking as the character moves.
        
        Args:
            masks: Standard ComfyUI MASK tensor with shape (N, H, W) where N = number of frames
        """
        
        try:
            from sam_3d_body import SAM3DBodyEstimator
        except ImportError:
            return ({}, images[:1], 0, "Error: SAM3DBody not installed", fps)
        
        # Handle masks input - standard MASK type is (N, H, W) tensor
        active_mask = None
        if masks is not None:
            log.info(f"Received masks, type: {type(masks).__name__}")
            
            if isinstance(masks, torch.Tensor):
                log.info(f"Mask tensor shape: {masks.shape}, ndim: {masks.ndim}")
                
                if masks.ndim == 3:
                    # Standard MASK format: (N, H, W) - one mask per frame
                    active_mask = masks
                    log.info(f"Using standard MASK format (N, H, W)")
                    
                elif masks.ndim == 4:
                    # Could be (N, 1, H, W) or (N, num_objects, H, W)
                    if masks.shape[1] == 1:
                        # Single object: squeeze to (N, H, W)
                        active_mask = masks.squeeze(1)
                        log.info(f"Squeezed (N, 1, H, W) to (N, H, W)")
                    elif object_id < masks.shape[1]:
                        # Multiple objects: select by object_id
                        active_mask = masks[:, object_id, :, :]
                        log.info(f"Selected object {object_id} from (N, {masks.shape[1]}, H, W)")
                    else:
                        # Fallback to first object
                        active_mask = masks[:, 0, :, :]
                        log.info(f"object_id {object_id} out of range, using object 0")
                        
                elif masks.ndim == 2:
                    # Single mask (H, W) - replicate for all frames
                    num_frames = images.shape[0]
                    active_mask = masks.unsqueeze(0).repeat(num_frames, 1, 1)
                    log.info(f"Replicated single mask to {num_frames} frames")
                    
            elif isinstance(masks, dict):
                # Legacy dict format support
                log.info(f"Received dict masks with keys: {list(masks.keys())[:5]}")
                # Try to extract frame-indexed masks
                keys = [k for k in masks.keys() if isinstance(k, int)]
                if keys:
                    sorted_keys = sorted(keys)
                    frame_masks = []
                    for k in sorted_keys:
                        m = masks[k]
                        if isinstance(m, np.ndarray):
                            m = torch.from_numpy(m)
                        if m.ndim == 3 and m.shape[0] == 1:
                            m = m.squeeze(0)
                        frame_masks.append(m)
                    if frame_masks:
                        active_mask = torch.stack(frame_masks, dim=0)
                        log.info(f"Built mask tensor from dict: {active_mask.shape}")
            
            if active_mask is not None:
                log.info(f"Final active_mask shape: {active_mask.shape}")
        else:
            log.info(f"No mask provided - will use auto-detection")
        
        total_frames = images.shape[0]
        actual_end = total_frames if end_frame == -1 else min(end_frame + 1, total_frames)
        frame_indices = list(range(start_frame, actual_end, skip_frames))
        
        if not frame_indices:
            return ({}, images[:1], 0, "Error: No frames")
        
        log.info(f"Processing {len(frame_indices)} frames...")
        if active_mask is not None:
            mask_frames = active_mask.shape[0] if hasattr(active_mask, 'shape') and active_mask.ndim >= 1 else 1
            log.info(f"Using per-frame masks ({mask_frames} masks available)")
        
        # Normalize external intrinsics if provided
        # Priority: external_intrinsics (CAMERA_INTRINSICS) > intrinsics_json (INTRINSICS) > SAM3DBody
        ext_intrinsics = None
        if external_intrinsics is not None:
            ext_intrinsics = normalize_intrinsics(external_intrinsics, len(frame_indices))
            if ext_intrinsics and ext_intrinsics.get("focal_length"):
                log.info(f"Using CAMERA_INTRINSICS (MoGe2/external) - will override SAM3DBody estimation")
            else:
                log.info(f"CAMERA_INTRINSICS provided but no valid focal length found")
                ext_intrinsics = None
        
        if ext_intrinsics is None and intrinsics_json is not None:
            ext_intrinsics = normalize_intrinsics(intrinsics_json, len(frame_indices))
            if ext_intrinsics and ext_intrinsics.get("focal_length"):
                ext_intrinsics["source"] = "json"
                log.info(f"Using INTRINSICS from JSON - will override SAM3DBody estimation")
            else:
                log.info(f"INTRINSICS from JSON provided but no valid focal length found")
                ext_intrinsics = None
        
        # Load SAM3D model from config dict
        # Expected format from "Load SAM3DBody Model (Direct)" node:
        # {ckpt_path, mhr_path, device, model_path, _loader}
        log.info(f"Model config type: {type(model)}")
        
        if not isinstance(model, dict):
            raise TypeError(
                f"SAM3D_MODEL must be a dict from 'Load SAM3DBody Model (Direct)' node. "
                f"Got: {type(model)}"
            )
        
        log.info(f"Model config keys: {list(model.keys())}")
        
        # Get paths from config
        ckpt_path = model.get("ckpt_path") or model.get("model_path")
        mhr_path = model.get("mhr_path", "")
        device = model.get("device", "cuda")
        
        if not ckpt_path:
            raise ValueError(
                "SAM3D_MODEL missing checkpoint path. "
                "Use the 'Load SAM3DBody Model (Direct)' node to load the model."
            )
        
        log.info(f"Loading SAM-3D-Body model...")
        log.info(f"  Checkpoint: {ckpt_path}")
        log.info(f"  MHR model: {mhr_path}")
        log.info(f"  Device: {device}")
        
        # Import and load the model
        try:
            from sam_3d_body import load_sam_3d_body
        except ImportError as e:
            raise ImportError(
                f"Could not import sam_3d_body. Please install Meta's SAM-3D-Body:\n"
                f"  1. git clone https://github.com/facebookresearch/sam-3d-body\n"
                f"  2. Set SAM3D_PATH environment variable to the cloned directory\n"
                f"  3. Or: pip install -e /path/to/sam-3d-body\n"
                f"Original error: {e}"
            )
        
        # Load the model
        result = load_sam_3d_body(
            checkpoint_path=ckpt_path,
            device=device,
            mhr_path=mhr_path
        )
        
        # Handle different return signatures
        if isinstance(result, tuple):
            sam_3d_model = result[0]
            model_cfg = result[1] if len(result) > 1 else None
            log.info(f"Model loaded (returned {len(result)} values)")
        else:
            sam_3d_model = result
            model_cfg = None
            log.info("Model loaded successfully!")
        
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
                                log.info(f"Frame 0 mask shape: {mask_np.shape}, bbox: {frame_bbox}")
                    
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
                                log.info(f"Multiple detections ({len(outputs)}), selected #{best_idx} with mask overlap {best_overlap}")
                        
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
                                            log.info(f"Got joint_parents from mhr.character_torch.skeleton: {joint_parents.shape if hasattr(joint_parents, 'shape') else len(joint_parents)} joints")
                                            log.info(f"First 10 parent indices: {joint_parents[:10] if joint_parents is not None else 'None'}")
                            except Exception as e:
                                log.info(f"Error getting joint_parents: {e}")
                        
                        # Store frame data including camera and rotations
                        focal_length = output.get("focal_length")
                        if focal_length is not None and hasattr(focal_length, 'item'):
                            focal_length = float(focal_length.item()) if hasattr(focal_length, 'item') else float(focal_length)
                        
                        # Debug first frame
                        if i == 0:
                            log.info(f"First frame output keys: {list(output.keys())}")
                            pred_global_rots = output.get("pred_global_rots")
                            if pred_global_rots is not None:
                                if hasattr(pred_global_rots, 'shape'):
                                    log.info(f"pred_global_rots shape: {pred_global_rots.shape}")
                            else:
                                log.info(f"WARNING: pred_global_rots is None!")
                            
                            pred_kp_2d = output.get("pred_keypoints_2d")
                            if pred_kp_2d is not None:
                                if hasattr(pred_kp_2d, 'shape'):
                                    log.info(f"pred_keypoints_2d shape: {pred_kp_2d.shape}")
                                    # Print first few 2D keypoints to verify they're in image space
                                    kp_np = pred_kp_2d.cpu().numpy() if hasattr(pred_kp_2d, 'cpu') else np.array(pred_kp_2d)
                                    log.info(f"First 5 2D keypoints: {kp_np[:5]}")
                            else:
                                log.info(f"WARNING: pred_keypoints_2d is None!")
                            
                            # Debug pred_keypoints_3d
                            pred_kp_3d = output.get("pred_keypoints_3d")
                            if pred_kp_3d is not None:
                                if hasattr(pred_kp_3d, 'shape'):
                                    log.info(f"pred_keypoints_3d shape: {pred_kp_3d.shape}")
                                else:
                                    log.info(f"pred_keypoints_3d type: {type(pred_kp_3d)}")
                            else:
                                log.info(f"WARNING: pred_keypoints_3d is None! Projection comparison will use joint_coords instead.")
                            
                            bbox = output.get("bbox")
                            if bbox is not None:
                                log.info(f"Detection bbox: {bbox}")
                            
                            # DEBUG: Investigate mesh vs joints 3D center offset
                            pred_verts = output.get("pred_vertices")
                            pred_joints = output.get("pred_joint_coords")
                            pred_cam_t = output.get("pred_cam_t")
                            
                            if pred_verts is not None and pred_joints is not None:
                                verts_np = pred_verts.cpu().numpy() if hasattr(pred_verts, 'cpu') else np.array(pred_verts)
                                joints_np = pred_joints.cpu().numpy() if hasattr(pred_joints, 'cpu') else np.array(pred_joints)
                                
                                # Handle batch dimension if present
                                if verts_np.ndim == 3:
                                    verts_np = verts_np[0]
                                if joints_np.ndim == 3:
                                    joints_np = joints_np[0]
                                
                                # Calculate 3D centers
                                mesh_center_3d = np.mean(verts_np, axis=0)
                                joints_center_3d = np.mean(joints_np, axis=0)
                                pelvis_3d = joints_np[0] if len(joints_np) > 0 else np.zeros(3)  # Joint 0 is typically pelvis
                                
                                # NEW: Calculate vertex bounds to understand coordinate system
                                verts_min = np.min(verts_np, axis=0)
                                verts_max = np.max(verts_np, axis=0)
                                verts_range = verts_max - verts_min
                                
                                log.info(f"Mesh vertices: {verts_np.shape}")
                                log.info(f"Skeleton joints: {joints_np.shape}")
                                log.info(f"Mesh center 3D: X={mesh_center_3d[0]:.4f}, Y={mesh_center_3d[1]:.4f}, Z={mesh_center_3d[2]:.4f}")
                                log.info(f"Mesh bounds min: X={verts_min[0]:.4f}, Y={verts_min[1]:.4f}, Z={verts_min[2]:.4f}")
                                log.info(f"Mesh bounds max: X={verts_max[0]:.4f}, Y={verts_max[1]:.4f}, Z={verts_max[2]:.4f}")
                                log.info(f"Mesh size (range): X={verts_range[0]:.4f}, Y={verts_range[1]:.4f}, Z={verts_range[2]:.4f}")
                                log.info(f"Joints center 3D: X={joints_center_3d[0]:.4f}, Y={joints_center_3d[1]:.4f}, Z={joints_center_3d[2]:.4f}")
                                log.info(f"Pelvis (joint 0) 3D: X={pelvis_3d[0]:.4f}, Y={pelvis_3d[1]:.4f}, Z={pelvis_3d[2]:.4f}")
                                log.info(f"Mesh vs Joints offset: dX={mesh_center_3d[0]-joints_center_3d[0]:.4f}, dY={mesh_center_3d[1]-joints_center_3d[1]:.4f}, dZ={mesh_center_3d[2]-joints_center_3d[2]:.4f}")
                                log.info(f"Mesh vs Pelvis offset: dX={mesh_center_3d[0]-pelvis_3d[0]:.4f}, dY={mesh_center_3d[1]-pelvis_3d[1]:.4f}, dZ={mesh_center_3d[2]-pelvis_3d[2]:.4f}")
                                
                                if pred_cam_t is not None:
                                    cam_t_np = pred_cam_t.cpu().numpy() if hasattr(pred_cam_t, 'cpu') else np.array(pred_cam_t)
                                    cam_t_np = cam_t_np.flatten()
                                    log.info(f"pred_cam_t: tx={cam_t_np[0]:.4f}, ty={cam_t_np[1]:.4f}, tz={cam_t_np[2]:.4f}")
                                    
                                    # Calculate expected 2D position from cam_t
                                    # In weak perspective: x_2d = fx * tx/tz + cx, y_2d = fy * ty/tz + cy
                                    if focal_length is not None and cam_t_np[2] != 0:
                                        img_w, img_h = img_np.shape[1], img_np.shape[0]
                                        expected_x = focal_length * cam_t_np[0] / cam_t_np[2] + img_w/2
                                        expected_y = focal_length * cam_t_np[1] / cam_t_np[2] + img_h/2
                                        log.info(f"Expected 2D position from cam_t: x={expected_x:.1f}, y={expected_y:.1f}")
                                    
                                    # The projection model assumes body at origin, camera at (tx, ty, tz)
                                    # So the body's screen position is determined by tx/tz and ty/tz
                                    # If mesh has an inherent offset from origin, we need to account for it
                                    log.info(f"========================================")
                        
                        # Determine focal length to use: external > SAM3DBody
                        final_focal_length = focal_length
                        final_cx = None
                        final_cy = None
                        intrinsics_source = "SAM3DBody"
                        
                        if ext_intrinsics is not None:
                            # Use external intrinsics
                            per_frame = ext_intrinsics.get("per_frame_focal")
                            if per_frame and i < len(per_frame):
                                final_focal_length = per_frame[i]
                            else:
                                final_focal_length = ext_intrinsics.get("focal_length", focal_length)
                            
                            final_cx = ext_intrinsics.get("cx")
                            final_cy = ext_intrinsics.get("cy")
                            intrinsics_source = ext_intrinsics.get("source", "external")
                            
                            if i == 0:
                                log.info(f"Using {intrinsics_source} intrinsics: focal={final_focal_length:.1f}px")
                                if final_cx is not None:
                                    log.info(f"  Principal point: ({final_cx:.1f}, {final_cy:.1f})")
                                if focal_length is not None:
                                    diff = final_focal_length - focal_length
                                    log.info(f"  SAM3DBody focal was: {focal_length:.1f}px (diff: {diff:+.1f}px)")
                        
                        frames[frame_idx] = {
                            "vertices": to_numpy(output.get("pred_vertices")),
                            "joint_coords": to_numpy(output.get("pred_joint_coords")),
                            "joint_rotations": to_numpy(output.get("pred_global_rots")),  # Per-joint rotations!
                            "pred_keypoints_2d": to_numpy(output.get("pred_keypoints_2d")),  # 2D keypoints for overlay
                            "pred_keypoints_3d": to_numpy(output.get("pred_keypoints_3d")),  # 3D keypoints for projection validation
                            "pred_cam_t": to_numpy(output.get("pred_cam_t")),
                            "focal_length": final_focal_length,
                            "focal_length_sam3d": focal_length,  # Keep original for comparison
                            "cx": final_cx,
                            "cy": final_cy,
                            "intrinsics_source": intrinsics_source,
                            "global_rot": to_numpy(output.get("global_rot")),  # Keep for compatibility
                            "bbox": to_numpy(output.get("bbox")),  # Store bbox for debugging
                            "image_size": (img_np.shape[1], img_np.shape[0]),  # (width, height) for alignment
                        }
                        
                        debug_images.append(img_tensor)
                    else:
                        frames[frame_idx] = {"vertices": None, "joint_coords": None}
                        debug_images.append(img_tensor)
                    
                    # Progress logging using batch_size as interval
                    if (i + 1) % batch_size == 0 or (i + 1) == len(frame_indices):
                        log.info(f"Processed {i + 1}/{len(frame_indices)}")
                        
                except Exception as e:
                    log.info(f"Error frame {frame_idx}: {e}")
                    frames[frame_idx] = {"vertices": None, "joint_coords": None}
                    debug_images.append(images[frame_idx])
        
        # Apply smoothing
        if smoothing_strength > 0:
            log.info(f"Applying smoothing...")
            frames = self._apply_smoothing(frames, smoothing_strength)
        
        # Compute average focal length from all frames
        focal_lengths = []
        focal_lengths_sam3d = []
        for f in frames.values():
            if f.get("focal_length") is not None:
                focal_lengths.append(f["focal_length"])
            if f.get("focal_length_sam3d") is not None:
                focal_lengths_sam3d.append(f["focal_length_sam3d"])
        
        avg_focal_length = sum(focal_lengths) / len(focal_lengths) if focal_lengths else 1000.0
        avg_focal_sam3d = sum(focal_lengths_sam3d) / len(focal_lengths_sam3d) if focal_lengths_sam3d else None
        
        # Determine intrinsics source
        intrinsics_source = "SAM3DBody"
        if ext_intrinsics is not None:
            intrinsics_source = ext_intrinsics.get("source", "external")
            log.info(f"Final focal length: {avg_focal_length:.1f}px (source: {intrinsics_source})")
            if avg_focal_sam3d is not None:
                diff = avg_focal_length - avg_focal_sam3d
                log.info(f"  SAM3DBody avg was: {avg_focal_sam3d:.1f}px (diff: {diff:+.1f}px)")
        else:
            log.info(f"Focal length: {avg_focal_length:.1f}px (from {len(focal_lengths)} frames)")
        
        # Build output - include fps for downstream nodes
        mesh_sequence = {
            "sequence_id": "batch",
            "frames": frames,
            "faces": faces,
            "joint_parents": joint_parents,
            "mhr_path": mhr_path,
            "fps": fps,  # Pass through for export
            "focal_length_px": avg_focal_length,  # Add for camera solver
            "focal_length_sam3d": avg_focal_sam3d,  # Original SAM3DBody estimation
            "intrinsics_source": intrinsics_source,
            "external_intrinsics": ext_intrinsics,  # Full external intrinsics if provided
        }
        
        if debug_images:
            debug_batch = torch.stack(debug_images, dim=0)
        else:
            debug_batch = images[:1]
        
        valid = sum(1 for f in frames.values() if f.get("vertices") is not None)
        status = f"Processed {valid}/{len(frame_indices)} frames at {fps} fps"
        log.info(f"{status}")
        
        return (mesh_sequence, debug_batch, len(frame_indices), status, fps, avg_focal_length)
