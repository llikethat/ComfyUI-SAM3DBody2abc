"""
Video Batch Processor for SAM3DBody
Based on SAM3DBodyProcess from ComfyUI-SAM3DBody/nodes/process.py

Processes video frames through SAM3DBody for 3D mesh recovery.
"""

import os
import tempfile
import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional


class SAM3DBodyBatchProcessor:
    """
    Process video frames through SAM3DBody for 3D mesh recovery.
    
    This is a batch version of SAM3DBodyProcess that processes multiple frames
    and collects results into a mesh sequence for animated export.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAM3D_MODEL", {
                    "tooltip": "Loaded SAM 3D Body model from Load node"
                }),
                "images": ("IMAGE", {
                    "tooltip": "Input images (batch from video or image sequence)"
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional segmentation mask to guide reconstruction"
                }),
                "bbox_threshold": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Confidence threshold for human detection bounding boxes"
                }),
                "inference_type": (["full", "body", "hand"], {
                    "default": "full",
                    "tooltip": "full: body+hand decoders, body: body decoder only, hand: hand decoder only"
                }),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "tooltip": "First frame to process"
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
            }
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE", "IMAGE", "INT", "STRING", "FLOAT")
    RETURN_NAMES = ("mesh_sequence", "images", "frame_count", "status", "focal_length")
    FUNCTION = "process_batch"
    CATEGORY = "SAM3DBody2abc/Video"
    
    def _comfy_image_to_numpy(self, image):
        """Convert ComfyUI image tensor to numpy BGR format."""
        # ComfyUI images are [B, H, W, C] float32 0-1 RGB
        if isinstance(image, torch.Tensor):
            img = image.cpu().numpy()
        else:
            img = np.array(image)
        
        # Handle batch dimension
        if img.ndim == 4:
            img = img[0]  # Take first image from batch
        
        # Convert to uint8
        img = (img * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img_bgr
    
    # SMPL 24 joint names for simplified skeleton export
    SMPL_JOINT_NAMES = [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
        "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
        "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand"
    ]
    
    # SMPL skeleton hierarchy (parent indices, -1 = root)
    SMPL_JOINT_PARENTS = [
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
        9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
    ]
    
    def _compute_bbox_from_mask(self, mask):
        """Compute bounding box from binary mask."""
        rows = np.any(mask > 0.5, axis=1)
        cols = np.any(mask > 0.5, axis=0)
        
        if not rows.any() or not cols.any():
            return None
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return np.array([[cmin, rmin, cmax, rmax]], dtype=np.float32)
    
    def _extract_mhr_joint_parents(self, sam_3d_model):
        """Extract joint parent hierarchy from MHR model (127 joints)."""
        joint_parents = None
        
        try:
            # Method 1: From the model directly
            if hasattr(sam_3d_model, 'mhr_head') and hasattr(sam_3d_model.mhr_head, 'mhr'):
                mhr = sam_3d_model.mhr_head.mhr
                if hasattr(mhr, 'character_torch') and hasattr(mhr.character_torch, 'skeleton'):
                    skeleton_obj = mhr.character_torch.skeleton
                    if hasattr(skeleton_obj, 'joint_parents'):
                        parent_tensor = skeleton_obj.joint_parents
                        if isinstance(parent_tensor, torch.Tensor):
                            joint_parents = parent_tensor.cpu().numpy().astype(int).tolist()
                            print(f"[SAM3DBody2abc] Extracted MHR joint hierarchy: {len(joint_parents)} joints")
                            root_idx = joint_parents.index(-1) if -1 in joint_parents else 0
                            print(f"[SAM3DBody2abc] Root joint at index: {root_idx}")
                            return joint_parents
        except Exception as e:
            print(f"[SAM3DBody2abc] Method 1 (direct model access) failed: {e}")
        
        try:
            # Method 2: Load from MHR model file in HuggingFace cache
            import glob
            
            hf_cache_base = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--sam-3d-body-dinov3")
            if os.path.exists(hf_cache_base):
                pattern = os.path.join(hf_cache_base, "snapshots", "*", "assets", "mhr_model.pt")
                matches = glob.glob(pattern)
                if matches:
                    matches.sort(key=os.path.getmtime, reverse=True)
                    mhr_path = matches[0]
                    print(f"[SAM3DBody2abc] Loading MHR model from: {mhr_path}")
                    mhr_model = torch.jit.load(mhr_path, map_location='cpu')
                    parent_tensor = mhr_model.character_torch.skeleton.joint_parents
                    joint_parents = parent_tensor.cpu().numpy().astype(int).tolist()
                    print(f"[SAM3DBody2abc] Loaded MHR hierarchy: {len(joint_parents)} joints")
                    return joint_parents
        except Exception as e:
            print(f"[SAM3DBody2abc] Method 2 (HF cache) failed: {e}")
        
        print(f"[SAM3DBody2abc] WARNING: Could not extract MHR joint hierarchy, FBX skeleton may be incorrect")
        return None
    
    def process_batch(
        self,
        model,
        images,
        mask=None,
        bbox_threshold: float = 0.8,
        inference_type: str = "full",
        start_frame: int = 0,
        end_frame: int = -1,
        skip_frames: int = 1,
    ):
        """Process video frames through SAM3DBody."""
        import comfy.utils
        
        print(f"[SAM3DBody2abc] Starting batch 3D mesh reconstruction...")
        print(f"[SAM3DBody2abc] Inference type: {inference_type}")
        
        # Calculate frame range
        total_frames = images.shape[0]
        actual_end = total_frames if end_frame == -1 else min(end_frame + 1, total_frames)
        frame_indices = list(range(start_frame, actual_end, skip_frames))
        
        if not frame_indices:
            return ([], images[:1], 0, "Error: No frames in range", 0.0)
        
        print(f"[SAM3DBody2abc] Processing {len(frame_indices)} frames out of {total_frames} total")
        if mask is not None:
            print(f"[SAM3DBody2abc] Using provided mask")
        
        try:
            # Import SAM 3D Body modules
            from sam_3d_body import SAM3DBodyEstimator
            
            # Extract model components
            sam_3d_model = model["model"]
            model_cfg = model["model_cfg"]
            
            # Create estimator (reuse for all frames)
            estimator = SAM3DBodyEstimator(
                sam_3d_body_model=sam_3d_model,
                model_cfg=model_cfg,
                human_detector=None,
                human_segmentor=None,
                fov_estimator=None,
            )
            
            # Extract MHR 127-joint hierarchy from model
            joint_parents = self._extract_mhr_joint_parents(sam_3d_model)
            
            mesh_sequence = []
            valid_count = 0
            first_focal_length = 0.0
            
            # Process mask if provided
            mask_np = None
            if mask is not None:
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = np.array(mask)
                if mask_np.ndim == 3:
                    mask_np = mask_np[0]  # Take first mask if batch
            
            pbar = comfy.utils.ProgressBar(len(frame_indices))
            
            for idx, frame_idx in enumerate(frame_indices):
                try:
                    # Get single frame and convert to BGR numpy
                    frame_tensor = images[frame_idx:frame_idx+1]
                    img_bgr = self._comfy_image_to_numpy(frame_tensor)
                    
                    # Compute bbox from mask if provided
                    bboxes = None
                    if mask_np is not None:
                        bboxes = self._compute_bbox_from_mask(mask_np)
                    
                    # Save image to temporary file (required by SAM3DBodyEstimator)
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        cv2.imwrite(tmp.name, img_bgr)
                        tmp_path = tmp.name
                    
                    try:
                        # Process image
                        outputs = estimator.process_one_image(
                            tmp_path,
                            bboxes=bboxes,
                            masks=mask_np,
                            bbox_thr=bbox_threshold,
                            use_mask=(mask_np is not None),
                            inference_type=inference_type,
                        )
                        
                        # Clean up temp file
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                        raise e
                    
                    # Check if we got valid output
                    if outputs and len(outputs) > 0:
                        output = outputs[0]  # Take first person
                        
                        # Extract vertices
                        vertices = output.get("pred_vertices", None)
                        if vertices is not None:
                            if isinstance(vertices, torch.Tensor):
                                vertices = vertices.cpu().numpy()
                        
                        # Extract joints
                        joints = output.get("pred_joint_coords", None)
                        if joints is None:
                            joints = output.get("pred_keypoints_3d", None)
                        if joints is not None:
                            if isinstance(joints, torch.Tensor):
                                joints = joints.cpu().numpy()
                        
                        # Extract joint rotations for FBX export
                        joint_rotations = output.get("pred_global_rots", None)
                        if joint_rotations is not None:
                            if isinstance(joint_rotations, torch.Tensor):
                                joint_rotations = joint_rotations.cpu().numpy()
                        
                        # Extract camera
                        camera = output.get("pred_cam_t", None)
                        if camera is not None:
                            if isinstance(camera, torch.Tensor):
                                camera = camera.cpu().numpy()
                        
                        # Extract focal length
                        focal_length = output.get("focal_length", None)
                        if focal_length is not None:
                            if isinstance(focal_length, torch.Tensor):
                                focal_length = focal_length.cpu().numpy()
                        
                        mesh_data = {
                            "frame_index": idx,
                            "source_frame": frame_idx,
                            "valid": vertices is not None,
                            "vertices": vertices,
                            "faces": estimator.faces if hasattr(estimator, 'faces') else None,
                            "joints": joints,
                            "joint_parents": joint_parents,  # From model (constant)
                            "joint_rotations": joint_rotations,
                            "camera": camera,
                            "focal_length": focal_length,
                            "bbox": output.get("bbox", None),
                            "pose_params": {
                                "body_pose": output.get("body_pose_params", None),
                                "hand_pose": output.get("hand_pose_params", None),
                                "global_rot": output.get("global_rot", None),
                                "shape": output.get("shape_params", None),
                            },
                        }
                        
                        mesh_sequence.append(mesh_data)
                        
                        if vertices is not None:
                            valid_count += 1
                            if valid_count == 1:
                                print(f"[SAM3DBody2abc] First valid mesh: {vertices.shape[0]} vertices")
                                if joints is not None:
                                    print(f"[SAM3DBody2abc] Joint data: {joints.shape[0]} joints")
                                if focal_length is not None:
                                    fl_val = float(np.array(focal_length).flatten()[0]) if hasattr(focal_length, '__len__') else float(focal_length)
                                    first_focal_length = fl_val
                                    print(f"[SAM3DBody2abc] Focal length: {fl_val:.2f}")
                    else:
                        mesh_sequence.append({
                            "frame_index": idx,
                            "source_frame": frame_idx,
                            "valid": False,
                        })
                        
                except Exception as e:
                    print(f"[SAM3DBody2abc] Frame {frame_idx} error: {e}")
                    mesh_sequence.append({
                        "frame_index": idx,
                        "source_frame": frame_idx,
                        "valid": False,
                        "error": str(e)
                    })
                
                pbar.update(1)
            
            # Return processed frames
            output_images = images[frame_indices]
            
            status = f"Processed {len(frame_indices)} frames, {valid_count} valid meshes"
            print(f"[SAM3DBody2abc] [OK] {status}")
            
            return (mesh_sequence, output_images, len(frame_indices), status, first_focal_length)
            
        except ImportError as e:
            print(f"[SAM3DBody2abc] [ERROR] Failed to import sam_3d_body")
            return ([], images[:1], 0, f"Error: {e}", 0.0)
            
        except Exception as e:
            print(f"[SAM3DBody2abc] [ERROR] Batch processing failed: {e}")
            import traceback
            traceback.print_exc()
            return ([], images[:1], 0, f"Error: {e}", 0.0)


class SAM3DBodySequenceProcess:
    """
    Simplified sequence processor with temporal smoothing.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAM3D_MODEL", {}),
                "images": ("IMAGE", {}),
            },
            "optional": {
                "temporal_smooth": ("BOOLEAN", {"default": True}),
                "smooth_window": ("INT", {"default": 3, "min": 1, "max": 15, "step": 2}),
            }
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE", "STRING")
    RETURN_NAMES = ("mesh_sequence", "status")
    FUNCTION = "process_sequence"
    CATEGORY = "SAM3DBody2abc/Video"
    
    def process_sequence(
        self,
        model,
        images,
        temporal_smooth: bool = True,
        smooth_window: int = 3,
    ):
        """Process sequence with optional temporal smoothing."""
        
        processor = SAM3DBodyBatchProcessor()
        mesh_sequence, output_images, frame_count, status = processor.process_batch(
            model=model,
            images=images,
            bbox_threshold=0.8,
            inference_type="full",
            start_frame=0,
            end_frame=-1,
            skip_frames=1,
        )
        
        if temporal_smooth and len(mesh_sequence) > 2:
            mesh_sequence = self._smooth_sequence(mesh_sequence, smooth_window)
            status += f" (smoothed)"
        
        return (mesh_sequence, status)
    
    def _smooth_sequence(self, sequence: List[Dict], window: int) -> List[Dict]:
        """Apply temporal smoothing to reduce jitter."""
        try:
            from scipy.ndimage import uniform_filter1d
        except ImportError:
            print("[SAM3DBody2abc] scipy not available, skipping smoothing")
            return sequence
        
        valid_indices = [i for i, m in enumerate(sequence) if m.get("valid", False)]
        
        if len(valid_indices) < 3:
            return sequence
        
        # Smooth vertices
        if all(sequence[i].get("vertices") is not None for i in valid_indices):
            try:
                verts = np.stack([sequence[i]["vertices"] for i in valid_indices])
                smoothed = uniform_filter1d(verts, size=window, axis=0, mode='nearest')
                for idx, fi in enumerate(valid_indices):
                    sequence[fi]["vertices"] = smoothed[idx]
            except Exception as e:
                print(f"[SAM3DBody2abc] Could not smooth vertices: {e}")
        
        # Smooth joints
        if all(sequence[i].get("joints") is not None for i in valid_indices):
            try:
                joints = np.stack([sequence[i]["joints"] for i in valid_indices])
                smoothed = uniform_filter1d(joints, size=window, axis=0, mode='nearest')
                for idx, fi in enumerate(valid_indices):
                    sequence[fi]["joints"] = smoothed[idx]
            except Exception as e:
                print(f"[SAM3DBody2abc] Could not smooth joints: {e}")
        
        return sequence
