"""
Video Batch Processor for SAM3DBody
Processes video frames through SAM3DBody by calling its native ProcessImage node.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import sys
import os


def find_sam3dbody_node():
    """
    Find and return the ProcessImage node class from ComfyUI-SAM3DBody.
    """
    # Method 1: Try to get from ComfyUI's node mappings
    try:
        import execution
        if hasattr(execution, 'nodes'):
            # Look for SAM3DBody nodes
            for name, cls in execution.nodes.NODE_CLASS_MAPPINGS.items():
                if "ProcessImage" in name and "SAM3D" in name:
                    return cls()
    except:
        pass
    
    # Method 2: Import directly from the module
    try:
        import folder_paths
        custom_nodes = os.path.join(folder_paths.base_path, "custom_nodes")
        
        for folder in ["ComfyUI-SAM3DBody", "ComfyUI_SAM3DBody"]:
            path = os.path.join(custom_nodes, folder)
            if os.path.exists(path) and path not in sys.path:
                sys.path.insert(0, path)
        
        # Try import
        try:
            from nodes import ProcessImage
            return ProcessImage()
        except:
            pass
        
        try:
            from nodes import SAM3DBodyProcessImage  
            return SAM3DBodyProcessImage()
        except:
            pass
            
    except:
        pass
    
    return None


class SAM3DBodyBatchProcessor:
    """
    Process video frames through SAM3DBody for 3D mesh recovery.
    
    This node calls the existing SAM3DBody ProcessImage node internally
    for each frame, collecting the results into a mesh sequence.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),  # [N, H, W, C] batch from VHS Load Video
                "sam3dbody_model": ("SAM3D_MODEL",),  # From Load SAM 3D Body Model
            },
            "optional": {
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
            }
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE", "INT", "STRING")
    RETURN_NAMES = ("mesh_sequence", "frame_count", "status")
    FUNCTION = "process_batch"
    CATEGORY = "SAM3DBody2abc/Video"
    
    def process_batch(
        self,
        images: torch.Tensor,
        sam3dbody_model: Any,
        start_frame: int = 0,
        end_frame: int = -1,
        skip_frames: int = 1,
    ) -> Tuple[List[Dict], int, str]:
        """Process video frames through SAM3DBody."""
        import comfy.utils
        
        total_frames = images.shape[0]
        actual_end = total_frames if end_frame == -1 else min(end_frame + 1, total_frames)
        frame_indices = list(range(start_frame, actual_end, skip_frames))
        
        if not frame_indices:
            return ([], 0, "Error: No frames in range")
        
        mesh_sequence = []
        valid_count = 0
        errors = []
        
        # Debug: Print model info
        print(f"[SAM3DBody2abc] Model type: {type(sam3dbody_model)}")
        if hasattr(sam3dbody_model, '__dict__'):
            print(f"[SAM3DBody2abc] Model keys: {list(sam3dbody_model.__dict__.keys())[:20]}")
        elif isinstance(sam3dbody_model, dict):
            print(f"[SAM3DBody2abc] Model dict keys: {list(sam3dbody_model.keys())[:20]}")
        
        # Try to find the pipeline/predictor
        pipeline = self._get_pipeline(sam3dbody_model)
        if pipeline is None:
            return ([], 0, f"Error: Could not extract pipeline from model. Type: {type(sam3dbody_model)}")
        
        print(f"[SAM3DBody2abc] Pipeline type: {type(pipeline)}")
        
        pbar = comfy.utils.ProgressBar(len(frame_indices))
        
        for idx, frame_idx in enumerate(frame_indices):
            try:
                # Get single frame
                frame = images[frame_idx]
                
                # Convert to format expected by SAM3DBody
                # ComfyUI images are [H, W, C] float 0-1
                # SAM3DBody likely expects [H, W, C] uint8 0-255 or PIL Image
                frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
                
                # Try inference
                result = self._run_inference(pipeline, frame_np)
                
                if result is not None:
                    mesh_data = self._extract_mesh_data(result)
                    mesh_data["frame_index"] = idx
                    mesh_data["source_frame"] = frame_idx
                    mesh_data["valid"] = mesh_data.get("vertices") is not None
                    mesh_sequence.append(mesh_data)
                    
                    if mesh_data["valid"]:
                        valid_count += 1
                else:
                    mesh_sequence.append({
                        "frame_index": idx,
                        "source_frame": frame_idx,
                        "valid": False,
                    })
                    
            except Exception as e:
                errors.append(f"Frame {frame_idx}: {str(e)[:50]}")
                mesh_sequence.append({
                    "frame_index": idx,
                    "source_frame": frame_idx,
                    "valid": False,
                    "error": str(e)
                })
            
            pbar.update(1)
        
        status = f"Processed {len(frame_indices)} frames, {valid_count} valid meshes"
        if errors:
            status += f". Errors: {len(errors)}"
        
        return (mesh_sequence, len(frame_indices), status)
    
    def _get_pipeline(self, model: Any) -> Any:
        """Extract the inference pipeline from the model object."""
        
        # If it's already a callable pipeline
        if hasattr(model, 'predict') or hasattr(model, 'infer') or hasattr(model, '__call__'):
            return model
        
        # If it's a dict, look for pipeline
        if isinstance(model, dict):
            for key in ['pipeline', 'model', 'predictor', 'sam3d', 'sam3dbody']:
                if key in model:
                    return model[key]
            # Return first value if it's a single-item dict
            if len(model) == 1:
                return list(model.values())[0]
        
        # If it has attributes
        if hasattr(model, '__dict__'):
            for attr in ['pipeline', 'model', 'predictor', 'sam3d', 'sam3dbody', '_model', '_pipeline']:
                if hasattr(model, attr):
                    return getattr(model, attr)
        
        # Last resort: return as-is
        return model
    
    def _run_inference(self, pipeline: Any, image: np.ndarray) -> Optional[Dict]:
        """Run inference with various method attempts."""
        
        methods = [
            ('predict', lambda p, img: p.predict(img)),
            ('infer', lambda p, img: p.infer(img)),
            ('__call__', lambda p, img: p(img)),
            ('forward', lambda p, img: p.forward(img)),
            ('process', lambda p, img: p.process(img)),
            ('run', lambda p, img: p.run(img)),
            ('reconstruct', lambda p, img: p.reconstruct(img)),
        ]
        
        # Also try with PIL Image
        try:
            from PIL import Image
            pil_image = Image.fromarray(image)
        except:
            pil_image = None
        
        for method_name, method_func in methods:
            if hasattr(pipeline, method_name):
                # Try with numpy array
                try:
                    result = method_func(pipeline, image)
                    if result is not None:
                        return result
                except Exception as e:
                    pass
                
                # Try with PIL image
                if pil_image is not None:
                    try:
                        result = method_func(pipeline, pil_image)
                        if result is not None:
                            return result
                    except:
                        pass
                
                # Try with torch tensor
                try:
                    img_tensor = torch.from_numpy(image).float() / 255.0
                    result = method_func(pipeline, img_tensor)
                    if result is not None:
                        return result
                except:
                    pass
                
                # Try with batched tensor [1, H, W, C]
                try:
                    img_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
                    result = method_func(pipeline, img_tensor)
                    if result is not None:
                        return result
                except:
                    pass
        
        return None
    
    def _extract_mesh_data(self, result: Any) -> Dict:
        """Extract mesh data from inference result."""
        mesh_data = {}
        
        # Handle different result types
        if isinstance(result, tuple):
            # Might be (mesh_dict, image, mask) from ComfyUI node
            result = result[0] if result else {}
        
        if isinstance(result, dict):
            source = result
        elif hasattr(result, '__dict__'):
            source = result.__dict__
        else:
            return mesh_data
        
        # Extract fields with various key names
        key_mappings = {
            'vertices': ['verts', 'vertices', 'v', 'mesh_verts', 'pred_verts'],
            'faces': ['faces', 'f', 'mesh_faces', 'triangles'],
            'joints': ['joints', 'J', 'joints_3d', 'skeleton', 'pred_joints'],
            'joints_2d': ['joints_2d', 'J_2d', 'keypoints', 'keypoints_2d'],
            'pose': ['pose', 'body_pose', 'theta', 'pose_params'],
            'betas': ['betas', 'shape', 'shape_params', 'beta'],
            'global_orient': ['global_orient', 'global_rotation', 'root_orient'],
            'transl': ['transl', 'translation', 'trans', 'root_trans'],
            'camera': ['camera', 'cam', 'camera_params', 'cam_params', 'pred_cam'],
        }
        
        for target_key, source_keys in key_mappings.items():
            for src_key in source_keys:
                if src_key in source and source[src_key] is not None:
                    val = source[src_key]
                    if isinstance(val, torch.Tensor):
                        mesh_data[target_key] = val.detach().cpu().numpy()
                    elif isinstance(val, np.ndarray):
                        mesh_data[target_key] = val
                    elif isinstance(val, (list, tuple)):
                        mesh_data[target_key] = np.array(val)
                    else:
                        mesh_data[target_key] = val
                    break
        
        return mesh_data


class SAM3DBodySequenceProcess:
    """
    Simplified sequence processor with temporal smoothing.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
                "sam3dbody_model": ("SAM3D_MODEL",),
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
        images: torch.Tensor,
        sam3dbody_model: Any,
        temporal_smooth: bool = True,
        smooth_window: int = 3,
    ) -> Tuple[List[Dict], str]:
        """Process sequence with optional temporal smoothing."""
        
        processor = SAM3DBodyBatchProcessor()
        mesh_sequence, frame_count, status = processor.process_batch(
            images=images,
            sam3dbody_model=sam3dbody_model,
            start_frame=0,
            end_frame=-1,
            skip_frames=1,
        )
        
        if temporal_smooth and len(mesh_sequence) > 2:
            mesh_sequence = self._smooth_sequence(mesh_sequence, smooth_window)
            status += f" (smoothed)"
        
        return (mesh_sequence, status)
    
    def _smooth_sequence(self, sequence: List[Dict], window: int) -> List[Dict]:
        """Apply temporal smoothing."""
        try:
            from scipy.ndimage import uniform_filter1d
        except ImportError:
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
            except:
                pass
        
        # Smooth joints
        if all(sequence[i].get("joints") is not None for i in valid_indices):
            try:
                joints = np.stack([sequence[i]["joints"] for i in valid_indices])
                smoothed = uniform_filter1d(joints, size=window, axis=0, mode='nearest')
                for idx, fi in enumerate(valid_indices):
                    sequence[fi]["joints"] = smoothed[idx]
            except:
                pass
        
        return sequence


class SAM3DBodyModelDebug:
    """
    Debug node to inspect the SAM3D_MODEL structure.
    Use this to see what methods and attributes the model has.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "sam3dbody_model": ("SAM3D_MODEL",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("debug_info",)
    FUNCTION = "debug_model"
    CATEGORY = "SAM3DBody2abc/Debug"
    
    def debug_model(self, sam3dbody_model: Any) -> Tuple[str]:
        """Print detailed info about the model structure."""
        info_lines = []
        
        info_lines.append(f"=== SAM3D_MODEL Debug Info ===")
        info_lines.append(f"Type: {type(sam3dbody_model)}")
        info_lines.append(f"Repr: {repr(sam3dbody_model)[:200]}")
        
        # If dict
        if isinstance(sam3dbody_model, dict):
            info_lines.append(f"\nDict keys: {list(sam3dbody_model.keys())}")
            for k, v in sam3dbody_model.items():
                info_lines.append(f"  {k}: {type(v).__name__}")
                if hasattr(v, '__dict__'):
                    info_lines.append(f"    attrs: {list(v.__dict__.keys())[:10]}")
        
        # If object with __dict__
        if hasattr(sam3dbody_model, '__dict__'):
            info_lines.append(f"\nObject attributes: {list(sam3dbody_model.__dict__.keys())[:30]}")
        
        # Callable methods
        methods = [m for m in dir(sam3dbody_model) if not m.startswith('_') and callable(getattr(sam3dbody_model, m, None))]
        info_lines.append(f"\nCallable methods: {methods[:20]}")
        
        # Check common inference methods
        for method in ['predict', 'infer', '__call__', 'forward', 'process', 'run']:
            has_method = hasattr(sam3dbody_model, method)
            info_lines.append(f"Has {method}: {has_method}")
        
        info = "\n".join(info_lines)
        print(info)
        
        return (info,)
