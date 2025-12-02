"""
Video Batch Processor for SAM3DBody
Processes video frames or image sequences through SAM3DBody model.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import folder_paths


class SAM3DBodyBatchProcessor:
    """
    Process video frames through SAM3DBody for 3D mesh recovery.
    Works with VHS Load Video node output and SAM3DBody model.
    
    This node automates what would otherwise require manual per-frame processing.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),  # [N, H, W, C] batch from VHS Load Video
                "sam3dbody_model": ("SAM3DBODY_MODEL",),  # From Load SAM 3D Body Model
            },
            "optional": {
                "masks": ("MASK",),  # Optional per-frame masks
                "detection_mode": (["full", "body", "hand"], {
                    "default": "full",
                    "tooltip": "full=entire body, body=torso/limbs, hand=hand detail"
                }),
                "det_thresh": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Detection confidence threshold"
                }),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000
                }),
                "end_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 100000,
                    "tooltip": "-1 processes all frames"
                }),
                "skip_frames": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 30,
                    "tooltip": "Process every Nth frame (1=all frames)"
                }),
                "fov": ("FLOAT", {
                    "default": 60.0,
                    "min": 20.0,
                    "max": 120.0,
                    "step": 1.0,
                    "tooltip": "Camera field of view estimate"
                }),
            }
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("mesh_sequence", "overlay_images", "frame_count", "status")
    FUNCTION = "process_batch"
    CATEGORY = "SAM3DBody/Video"
    
    def process_batch(
        self,
        images: torch.Tensor,
        sam3dbody_model: Any,
        masks: Optional[torch.Tensor] = None,
        detection_mode: str = "full",
        det_thresh: float = 0.5,
        start_frame: int = 0,
        end_frame: int = -1,
        skip_frames: int = 1,
        fov: float = 60.0,
    ) -> Tuple[List[Dict], torch.Tensor, int, str]:
        """
        Process video frames through SAM3DBody.
        
        Returns:
            mesh_sequence: List of per-frame mesh data (vertices, faces, joints, pose, shape)
            overlay_images: Visualization with mesh overlay
            frame_count: Number of processed frames
            status: Processing status message
        """
        import comfy.utils
        
        total_frames = images.shape[0]
        actual_end = total_frames if end_frame == -1 else min(end_frame + 1, total_frames)
        frame_indices = list(range(start_frame, actual_end, skip_frames))
        
        if not frame_indices:
            return ([], images, 0, "Error: No frames in range")
        
        mesh_sequence = []
        overlay_list = []
        valid_count = 0
        
        pbar = comfy.utils.ProgressBar(len(frame_indices))
        
        for idx, frame_idx in enumerate(frame_indices):
            try:
                # Get frame
                frame = images[frame_idx]
                frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
                
                # Get mask if provided
                mask_np = None
                if masks is not None and frame_idx < masks.shape[0]:
                    mask_np = (masks[frame_idx].cpu().numpy() * 255).astype(np.uint8)
                
                # Process through SAM3DBody
                result = self._call_sam3dbody(
                    sam3dbody_model, frame_np, mask_np, detection_mode, det_thresh, fov
                )
                
                # Extract mesh data
                mesh_data = {
                    "frame_index": idx,
                    "source_frame": frame_idx,
                    "vertices": result.get("verts") or result.get("vertices"),
                    "faces": result.get("faces"),
                    "joints": result.get("joints") or result.get("J"),
                    "joints_2d": result.get("joints_2d") or result.get("J_2d"),
                    "pose": result.get("pose") or result.get("body_pose"),
                    "betas": result.get("betas") or result.get("shape"),
                    "global_orient": result.get("global_orient"),
                    "transl": result.get("transl") or result.get("translation"),
                    "camera": result.get("camera") or result.get("cam"),
                    "valid": result.get("verts") is not None or result.get("vertices") is not None,
                }
                
                mesh_sequence.append(mesh_data)
                
                if mesh_data["valid"]:
                    valid_count += 1
                
                # Get overlay or create from frame
                overlay = result.get("overlay") or result.get("render")
                if overlay is not None:
                    if isinstance(overlay, np.ndarray):
                        overlay_tensor = torch.from_numpy(overlay.astype(np.float32) / 255.0)
                    else:
                        overlay_tensor = overlay
                    overlay_list.append(overlay_tensor.unsqueeze(0) if overlay_tensor.dim() == 3 else overlay_tensor)
                else:
                    overlay_list.append(images[frame_idx:frame_idx+1])
                    
            except Exception as e:
                print(f"[SAM3DBodyBatchProcessor] Frame {frame_idx} error: {e}")
                mesh_sequence.append({
                    "frame_index": idx,
                    "source_frame": frame_idx,
                    "valid": False,
                    "error": str(e)
                })
                overlay_list.append(images[frame_idx:frame_idx+1])
            
            pbar.update(1)
        
        # Stack overlays
        overlay_images = torch.cat(overlay_list, dim=0) if overlay_list else images[frame_indices]
        
        status = f"Processed {len(frame_indices)} frames, {valid_count} valid meshes"
        
        return (mesh_sequence, overlay_images, len(frame_indices), status)
    
    def _call_sam3dbody(
        self,
        model: Any,
        image: np.ndarray,
        mask: Optional[np.ndarray],
        mode: str,
        det_thresh: float,
        fov: float
    ) -> Dict:
        """Call SAM3DBody model with various interface attempts."""
        
        # Try the standard ComfyUI-SAM3DBody interface
        if hasattr(model, 'process_image'):
            return model.process_image(
                image, mask=mask, mode=mode,
                det_thresh=det_thresh, fov=fov
            )
        
        # Try predictor interface
        if hasattr(model, 'predictor'):
            predictor = model.predictor
            if hasattr(predictor, 'predict'):
                return predictor.predict(image, det_thresh=det_thresh)
        
        # Try direct inference
        if hasattr(model, 'infer'):
            return model.infer(image, det_thresh=det_thresh)
        
        # Try callable
        if callable(model):
            return model(image)
        
        # Try accessing the underlying sam3dbody
        if hasattr(model, 'sam3d_body'):
            sam3d = model.sam3d_body
            if hasattr(sam3d, 'infer'):
                return sam3d.infer(image)
        
        raise RuntimeError("Could not find compatible SAM3DBody interface")


class SAM3DBodySequenceProcess:
    """
    Process image sequence (folder of images) through SAM3DBody.
    Alternative to video input for image sequences.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),  # Can be from Load Images node
                "sam3dbody_model": ("SAM3DBODY_MODEL",),
            },
            "optional": {
                "detection_mode": (["full", "body", "hand"], {"default": "full"}),
                "det_thresh": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
                "fov": ("FLOAT", {"default": 60.0, "min": 20.0, "max": 120.0}),
                "temporal_smooth": ("BOOLEAN", {"default": True}),
                "smooth_window": ("INT", {"default": 3, "min": 1, "max": 11, "step": 2}),
            }
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE", "IMAGE", "INT")
    RETURN_NAMES = ("mesh_sequence", "overlays", "frame_count")
    FUNCTION = "process_sequence"
    CATEGORY = "SAM3DBody/Video"
    
    def process_sequence(
        self,
        images: torch.Tensor,
        sam3dbody_model: Any,
        detection_mode: str = "full",
        det_thresh: float = 0.5,
        fov: float = 60.0,
        temporal_smooth: bool = True,
        smooth_window: int = 3,
    ) -> Tuple[List[Dict], torch.Tensor, int]:
        """Process image sequence and optionally apply temporal smoothing."""
        
        # Use batch processor
        processor = SAM3DBodyBatchProcessor()
        mesh_sequence, overlays, count, _ = processor.process_batch(
            images=images,
            sam3dbody_model=sam3dbody_model,
            detection_mode=detection_mode,
            det_thresh=det_thresh,
            fov=fov,
        )
        
        # Apply temporal smoothing if requested
        if temporal_smooth and len(mesh_sequence) > 1:
            mesh_sequence = self._apply_temporal_smoothing(mesh_sequence, smooth_window)
        
        return (mesh_sequence, overlays, count)
    
    def _apply_temporal_smoothing(
        self,
        sequence: List[Dict],
        window: int
    ) -> List[Dict]:
        """Apply temporal smoothing to reduce jitter between frames."""
        
        half_window = window // 2
        smoothed = []
        
        for i, frame in enumerate(sequence):
            if not frame.get("valid"):
                smoothed.append(frame)
                continue
            
            # Get window of frames
            start = max(0, i - half_window)
            end = min(len(sequence), i + half_window + 1)
            
            valid_frames = [
                sequence[j] for j in range(start, end)
                if sequence[j].get("valid") and sequence[j].get("vertices") is not None
            ]
            
            if len(valid_frames) <= 1:
                smoothed.append(frame)
                continue
            
            # Average vertices
            new_frame = frame.copy()
            vertices_stack = np.stack([np.array(f["vertices"]) for f in valid_frames])
            new_frame["vertices"] = np.mean(vertices_stack, axis=0)
            
            # Average joints if present
            if frame.get("joints") is not None:
                joints_stack = np.stack([
                    np.array(f["joints"]) for f in valid_frames
                    if f.get("joints") is not None
                ])
                if len(joints_stack) > 1:
                    new_frame["joints"] = np.mean(joints_stack, axis=0)
            
            smoothed.append(new_frame)
        
        return smoothed
