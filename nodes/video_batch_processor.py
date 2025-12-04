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
                "fov": ("FLOAT", {
                    "default": 55.0,
                    "min": 0.0,
                    "max": 150.0,
                    "step": 1.0,
                    "tooltip": "Camera FOV in degrees. Set to 0 to use focal length instead."
                }),
                "focal_length_mm": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 1.0,
                    "tooltip": "Lens focal length in mm (e.g. 50mm, 35mm, 85mm). Requires sensor_width_mm."
                }),
                "sensor_width_mm": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Sensor width in mm. Full-frame=36, APS-C=23.5, MFT=17.3, Super35=24.9"
                }),
                "focal_length_px": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10000.0,
                    "step": 1.0,
                    "tooltip": "Focal length in PIXELS (advanced). Overrides mm calculation if >0."
                }),
                "auto_calibrate": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Auto-estimate FOV using GeoCalib (requires geocalib package). Overrides all other options."
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
        
        # Method 3: Use anatomical fallback (hardcoded MHR hierarchy)
        print(f"[SAM3DBody2abc] Using anatomical MHR joint hierarchy fallback")
        return self._get_mhr_joint_parents_fallback(127)
    
    def _get_mhr_joint_parents_fallback(self, num_joints: int):
        """
        Anatomically-correct MHR joint hierarchy fallback.
        Uses left_hip (9) as root connecting upper and lower body.
        """
        mhr70_parents = [
            69, 0, 0, 1, 2,           # 0-4: head
            69, 69, 5, 6,             # 5-8: shoulders, elbows
            -1, 9, 9, 10,             # 9-12: hips (root), knees
            11, 12,                   # 13-14: ankles
            13, 13, 13,               # 15-17: left foot
            14, 14, 14,               # 18-20: right foot
            # Right hand (21-41)
            22, 23, 24, 41,           # thumb
            26, 27, 28, 41,           # index
            30, 31, 32, 41,           # middle
            34, 35, 36, 41,           # ring
            38, 39, 40, 41, 8,        # pinky + wrist
            # Left hand (42-62)
            43, 44, 45, 62,           # thumb
            47, 48, 49, 62,           # index
            51, 52, 53, 62,           # middle
            55, 56, 57, 62,           # ring
            59, 60, 61, 62, 7,        # pinky + wrist
            # Additional (63-69)
            7, 8, 7, 8, 5, 6, 9,      # olecranon, cubital, acromion, neck
        ]
        
        if num_joints <= 70:
            return mhr70_parents[:num_joints]
        
        # Extend for 127 joints (additional face keypoints -> nose)
        joint_parents = mhr70_parents.copy()
        for _ in range(70, num_joints):
            joint_parents.append(0)
        
        return joint_parents
    
    def _fov_to_focal_length(self, fov_degrees: float, img_size: int) -> float:
        """
        Convert field of view (degrees) to focal length (pixels).
        
        Formula: focal_length = img_size / (2 * tan(fov/2))
        
        Common FOV values:
        - Smartphone: 50-65°
        - Webcam: 55-70°
        - GoPro/Action cam: 70-120°
        - DSLR 50mm: 40-50°
        - DSLR 35mm: 55-65°
        """
        fov_radians = np.radians(fov_degrees)
        focal_length = img_size / (2 * np.tan(fov_radians / 2))
        return float(focal_length)
    
    def _focal_length_to_fov(self, focal_length: float, img_size: int) -> float:
        """
        Convert focal length (pixels) to field of view (degrees).
        
        Formula: fov = 2 * atan(img_size / (2 * focal_length))
        
        Note: focal_length is in PIXELS, not mm!
        The relationship to physical focal length is:
          focal_px = focal_mm * (sensor_width_px / sensor_width_mm)
        """
        fov_radians = 2 * np.arctan(img_size / (2 * focal_length))
        fov_degrees = np.degrees(fov_radians)
        return float(fov_degrees)
    
    def _focal_mm_to_px(self, focal_mm: float, sensor_width_mm: float, image_width_px: int) -> float:
        """
        Convert physical focal length (mm) to pixel focal length.
        
        Formula: focal_px = focal_mm * (image_width_px / sensor_width_mm)
        
        Common sensor widths:
        - Full Frame (35mm): 36.0 mm
        - APS-C Canon: 22.3 mm
        - APS-C Nikon/Sony: 23.5 mm
        - APS-C Fuji: 23.5 mm
        - Micro Four Thirds: 17.3 mm
        - 1" sensor: 13.2 mm
        - Super 35 (cinema): 24.89 mm
        - RED Komodo: 27.03 mm
        - ARRI Alexa: 28.17 mm
        """
        focal_px = focal_mm * (image_width_px / sensor_width_mm)
        return float(focal_px)
    
    def _auto_calibrate_fov(self, images: torch.Tensor, frame_indices: List[int]) -> Tuple[float, float]:
        """
        Auto-estimate FOV using GeoCalib model.
        
        Uses first frame to estimate FOV for the entire video.
        Returns: (focal_length, fov_degrees)
        """
        try:
            from geocalib import GeoCalib
        except ImportError:
            raise ImportError(
                "GeoCalib not installed. Install with:\n"
                "  pip install -e 'git+https://github.com/cvg/GeoCalib#egg=geocalib'\n"
                "Or use torch.hub:\n"
                "  model = torch.hub.load('cvg/GeoCalib', 'GeoCalib', trust_repo=True)"
            )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SAM3DBody2abc] Loading GeoCalib model on {device}...")
        
        # Load GeoCalib model
        geocalib_model = GeoCalib().to(device)
        
        img_h, img_w = images.shape[1], images.shape[2]
        img_size = max(img_h, img_w)
        
        # Use only first frame for simplicity
        idx = frame_indices[0]
        
        # Convert ComfyUI image [H, W, C] to GeoCalib format [C, H, W]
        img = images[idx].cpu()  # [H, W, C]
        img = img.permute(2, 0, 1)  # [C, H, W]
        img = img.to(device)
        
        print(f"[SAM3DBody2abc] Calibrating frame {idx} with GeoCalib...")
        
        # Calibrate single image
        result = geocalib_model.calibrate(img)
        
        # Extract focal length from camera object
        # GeoCalib Camera stores: {w, h, fx, fy, cx, cy, ...} as a tensor
        camera = result["camera"]
        focal_length = None
        
        print(f"[SAM3DBody2abc] Camera type: {type(camera)}")
        
        # Try to access the underlying data tensor
        try:
            # GeoCalib Camera is a TensorWrapper - access underlying data
            if hasattr(camera, '_data'):
                # _data shape is (..., 6/7/8) with [w, h, fx, fy, cx, cy, ...]
                data = camera._data
                if isinstance(data, torch.Tensor):
                    # fx is at index 2
                    focal_length = float(data[..., 2].flatten()[0].cpu().numpy())
                    print(f"[SAM3DBody2abc] Got focal from camera._data[2]: {focal_length:.2f}")
        except Exception as e:
            print(f"[SAM3DBody2abc] camera._data failed: {e}")
        
        if focal_length is None:
            try:
                # Try direct tensor access (Camera might subclass Tensor)
                if isinstance(camera, torch.Tensor):
                    focal_length = float(camera[..., 2].flatten()[0].cpu().numpy())
                    print(f"[SAM3DBody2abc] Got focal from camera tensor[2]: {focal_length:.2f}")
            except Exception as e:
                print(f"[SAM3DBody2abc] camera tensor access failed: {e}")
        
        if focal_length is None:
            try:
                # Try fx property
                if hasattr(camera, 'fx'):
                    fx_val = camera.fx
                    if isinstance(fx_val, torch.Tensor):
                        focal_length = float(fx_val.flatten()[0].cpu().numpy())
                    else:
                        focal_length = float(fx_val)
                    print(f"[SAM3DBody2abc] Got focal from camera.fx: {focal_length:.2f}")
            except Exception as e:
                print(f"[SAM3DBody2abc] camera.fx failed: {e}")
        
        if focal_length is None:
            try:
                # Try f property (normalized)
                if hasattr(camera, 'f'):
                    f_val = camera.f
                    if isinstance(f_val, torch.Tensor):
                        f_normalized = float(f_val.flatten()[0].cpu().numpy())
                        # Check if it's normalized (< 10) or absolute
                        if f_normalized < 10:
                            focal_length = f_normalized * img_size
                            print(f"[SAM3DBody2abc] Got focal from camera.f (normalized): {f_normalized} * {img_size} = {focal_length:.2f}")
                        else:
                            focal_length = f_normalized
                            print(f"[SAM3DBody2abc] Got focal from camera.f (absolute): {focal_length:.2f}")
            except Exception as e:
                print(f"[SAM3DBody2abc] camera.f failed: {e}")
        
        if focal_length is None:
            # Last resort: inspect the object
            print(f"[SAM3DBody2abc] Camera object attributes: {[a for a in dir(camera) if not a.startswith('_')]}")
            if hasattr(camera, '__dict__'):
                print(f"[SAM3DBody2abc] Camera __dict__: {camera.__dict__}")
            # Try to print it directly
            print(f"[SAM3DBody2abc] Camera repr: {repr(camera)}")
            raise ValueError("Could not extract focal length from GeoCalib result")
        
        # Calculate FOV from focal length
        fov_degrees = 2 * np.degrees(np.arctan(img_size / (2 * focal_length)))
        
        print(f"[SAM3DBody2abc] GeoCalib result: focal={focal_length:.2f}, fov={fov_degrees:.1f}°")
        
        # Also print gravity direction if available
        if "gravity" in result:
            try:
                gravity = result["gravity"]
                if isinstance(gravity, torch.Tensor):
                    gravity = gravity.cpu().numpy()
                print(f"[SAM3DBody2abc] Estimated gravity direction: {gravity}")
            except:
                pass
        
        return focal_length, fov_degrees
    
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
        fov: float = 55.0,
        focal_length_mm: float = 0.0,
        sensor_width_mm: float = 0.0,
        focal_length_px: float = 0.0,
        auto_calibrate: bool = False,
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
        
        # Get image size for focal length calculation
        img_h, img_w = images.shape[1], images.shape[2]
        img_size = max(img_h, img_w)
        
        # Calculate focal length - priority: auto_calibrate > focal_length_px > (mm + sensor) > fov
        custom_focal_length = None
        estimated_fov = fov
        
        if auto_calibrate:
            # Try to use GeoCalib for automatic FOV estimation
            try:
                custom_focal_length, estimated_fov = self._auto_calibrate_fov(images, frame_indices[:min(5, len(frame_indices))])
                print(f"[SAM3DBody2abc] Auto-calibrated: focal={custom_focal_length:.2f}px, FOV={estimated_fov:.1f}°")
            except Exception as e:
                print(f"[SAM3DBody2abc] Auto-calibration failed: {e}")
                auto_calibrate = False
        
        if not auto_calibrate:
            if focal_length_px > 0:
                # Direct pixel focal length (highest priority manual option)
                custom_focal_length = focal_length_px
                estimated_fov = self._focal_length_to_fov(focal_length_px, img_size)
                print(f"[SAM3DBody2abc] Using focal_length_px: {focal_length_px:.2f}px → FOV={estimated_fov:.1f}°")
            elif focal_length_mm > 0 and sensor_width_mm > 0:
                # Convert mm to pixels using sensor size
                custom_focal_length = self._focal_mm_to_px(focal_length_mm, sensor_width_mm, img_w)
                estimated_fov = self._focal_length_to_fov(custom_focal_length, img_size)
                print(f"[SAM3DBody2abc] Using {focal_length_mm:.1f}mm lens on {sensor_width_mm:.1f}mm sensor")
                print(f"[SAM3DBody2abc] Calculated: focal={custom_focal_length:.2f}px, FOV={estimated_fov:.1f}°")
            elif fov > 0:
                # Use FOV to calculate focal length
                custom_focal_length = self._fov_to_focal_length(fov, img_size)
                estimated_fov = fov
                print(f"[SAM3DBody2abc] Using FOV: {fov:.1f}° → focal={custom_focal_length:.2f}px")
            else:
                # Default FOV
                custom_focal_length = self._fov_to_focal_length(55.0, img_size)
                estimated_fov = 55.0
                print(f"[SAM3DBody2abc] Using default: FOV=55° → focal={custom_focal_length:.2f}px")
        
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
            # Use custom focal length if we calculated one, otherwise 0 to be updated later
            first_focal_length = custom_focal_length if custom_focal_length else 0.0
            
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
                        
                        # Extract focal length - prefer our custom FOV-based calculation
                        focal_length = output.get("focal_length", None)
                        if focal_length is not None:
                            if isinstance(focal_length, torch.Tensor):
                                focal_length = focal_length.cpu().numpy()
                        
                        # Override with custom focal length if we calculated one from FOV
                        if custom_focal_length is not None:
                            focal_length = custom_focal_length
                        
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
