"""
Mesh Sequence Accumulator and Utilities
For collecting mesh data from SAM3DBody output into sequences for animation export.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class MeshSequenceAccumulator:
    """
    Accumulate mesh data from multiple SAM3DBody calls into a sequence.
    Use this when processing frames one at a time through the standard
    SAM3DBody Process Image node.
    """
    
    # Class-level storage
    _sequences: Dict[str, List[Dict]] = {}
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "sam3dbody_mesh": ("SAM3D_MESH",),  # From SAM3DBody Process Image
                "sequence_id": ("STRING", {
                    "default": "animation_001",
                    "multiline": False
                }),
            },
            "optional": {
                "frame_index": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 100000,
                    "tooltip": "-1 for auto-increment"
                }),
                "reset_sequence": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE", "INT", "STRING")
    RETURN_NAMES = ("mesh_sequence", "frame_count", "status")
    FUNCTION = "accumulate"
    CATEGORY = "SAM3DBody2abc/Video"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # Always execute
    
    def accumulate(
        self,
        sam3dbody_mesh: Dict,
        sequence_id: str = "animation_001",
        frame_index: int = -1,
        reset_sequence: bool = False,
    ) -> Tuple[List[Dict], int, str]:
        """
        Add mesh to sequence.
        """
        # Initialize or reset
        if reset_sequence or sequence_id not in self._sequences:
            self._sequences[sequence_id] = []
        
        sequence = self._sequences[sequence_id]
        
        # Determine frame index
        if frame_index == -1:
            actual_index = len(sequence)
        else:
            actual_index = frame_index
        
        # Extract data from SAM3DBody mesh output
        mesh_data = {
            "frame_index": actual_index,
            "vertices": self._extract_array(sam3dbody_mesh, ["verts", "vertices", "v"]),
            "faces": self._extract_array(sam3dbody_mesh, ["faces", "f", "triangles"]),
            "joints": self._extract_array(sam3dbody_mesh, ["joints", "J", "keypoints", "joints_3d"]),
            "joints_2d": self._extract_array(sam3dbody_mesh, ["joints_2d", "J_2d", "keypoints_2d"]),
            "pose": self._extract_array(sam3dbody_mesh, ["pose", "body_pose", "theta"]),
            "betas": self._extract_array(sam3dbody_mesh, ["betas", "shape", "beta"]),
            "global_orient": self._extract_array(sam3dbody_mesh, ["global_orient", "orient"]),
            "transl": self._extract_array(sam3dbody_mesh, ["transl", "translation", "trans"]),
            "camera": sam3dbody_mesh.get("camera") or sam3dbody_mesh.get("cam"),
            "valid": True,
        }
        
        # Check validity
        mesh_data["valid"] = mesh_data["vertices"] is not None
        
        # Add to sequence
        sequence.append(mesh_data)
        
        # Sort by frame index
        sequence.sort(key=lambda x: x["frame_index"])
        
        status = f"Frame {actual_index} added. Total: {len(sequence)} frames"
        
        return (sequence, len(sequence), status)
    
    def _extract_array(self, data: Dict, keys: List[str]) -> Optional[np.ndarray]:
        """Extract array from dict trying multiple key names."""
        for key in keys:
            if key in data and data[key] is not None:
                value = data[key]
                if isinstance(value, torch.Tensor):
                    return value.cpu().numpy()
                elif isinstance(value, np.ndarray):
                    return value
                elif isinstance(value, list):
                    return np.array(value)
        return None
    
    @classmethod
    def get_sequence(cls, sequence_id: str) -> List[Dict]:
        return cls._sequences.get(sequence_id, [])
    
    @classmethod
    def clear_sequence(cls, sequence_id: str):
        if sequence_id in cls._sequences:
            del cls._sequences[sequence_id]
    
    @classmethod
    def clear_all(cls):
        cls._sequences.clear()


class MeshSequenceFromSAM3DBody:
    """
    Convert a single SAM3DBody mesh output to a mesh sequence format.
    Useful for compatibility with export nodes when processing single images.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "sam3dbody_mesh": ("SAM3D_MESH",),
            },
            "optional": {
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 100000}),
            }
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE",)
    RETURN_NAMES = ("mesh_sequence",)
    FUNCTION = "convert"
    CATEGORY = "SAM3DBody2abc/Video"
    
    def convert(
        self,
        sam3dbody_mesh: Dict,
        frame_index: int = 0,
    ) -> Tuple[List[Dict]]:
        """Convert single mesh to sequence format."""
        
        mesh_data = {
            "frame_index": frame_index,
            "vertices": self._get_value(sam3dbody_mesh, "verts", "vertices"),
            "faces": self._get_value(sam3dbody_mesh, "faces"),
            "joints": self._get_value(sam3dbody_mesh, "joints", "J"),
            "pose": self._get_value(sam3dbody_mesh, "pose", "body_pose"),
            "betas": self._get_value(sam3dbody_mesh, "betas", "shape"),
            "valid": True,
        }
        
        mesh_data["valid"] = mesh_data["vertices"] is not None
        
        return ([mesh_data],)
    
    def _get_value(self, data: Dict, *keys) -> Any:
        for key in keys:
            if key in data:
                value = data[key]
                if isinstance(value, torch.Tensor):
                    return value.cpu().numpy()
                return value
        return None


class MeshSequencePreview:
    """
    Preview mesh sequence information.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE",),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("frame_count", "vertex_count", "joint_count", "has_faces", "info")
    FUNCTION = "preview"
    CATEGORY = "SAM3DBody2abc/Video"
    
    def preview(self, mesh_sequence: List[Dict]) -> Tuple[int, int, int, bool, str]:
        """Get sequence information."""
        
        frame_count = len(mesh_sequence)
        valid_count = sum(1 for f in mesh_sequence if f.get("valid"))
        
        # Get counts from first valid frame
        vertex_count = 0
        joint_count = 0
        has_faces = False
        
        for frame in mesh_sequence:
            if frame.get("valid"):
                if frame.get("vertices") is not None:
                    vertex_count = len(frame["vertices"])
                if frame.get("joints") is not None:
                    joint_count = len(frame["joints"])
                if frame.get("faces") is not None:
                    has_faces = True
                break
        
        info = f"Frames: {frame_count} ({valid_count} valid)\n"
        info += f"Vertices: {vertex_count}\n"
        info += f"Joints: {joint_count}\n"
        info += f"Has Faces: {has_faces}"
        
        return (frame_count, vertex_count, joint_count, has_faces, info)


class MeshSequenceSmooth:
    """
    Apply temporal smoothing to reduce jitter in mesh sequences.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE",),
            },
            "optional": {
                "window_size": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 15,
                    "step": 2,
                    "tooltip": "Smoothing window (odd numbers)"
                }),
                "smooth_vertices": ("BOOLEAN", {"default": True}),
                "smooth_joints": ("BOOLEAN", {"default": True}),
                "preserve_extremes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Don't smooth first/last frames"
                }),
            }
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE",)
    RETURN_NAMES = ("smoothed_sequence",)
    FUNCTION = "smooth"
    CATEGORY = "SAM3DBody2abc/Video"
    
    def smooth(
        self,
        mesh_sequence: List[Dict],
        window_size: int = 3,
        smooth_vertices: bool = True,
        smooth_joints: bool = True,
        preserve_extremes: bool = True,
    ) -> Tuple[List[Dict]]:
        """Apply temporal smoothing."""
        
        if len(mesh_sequence) < 3:
            return (mesh_sequence,)
        
        half_window = window_size // 2
        smoothed = []
        
        for i, frame in enumerate(mesh_sequence):
            # Skip invalid frames
            if not frame.get("valid"):
                smoothed.append(frame.copy())
                continue
            
            # Preserve extremes
            if preserve_extremes and (i < half_window or i >= len(mesh_sequence) - half_window):
                smoothed.append(frame.copy())
                continue
            
            # Get window
            start = max(0, i - half_window)
            end = min(len(mesh_sequence), i + half_window + 1)
            
            valid_in_window = [
                mesh_sequence[j] for j in range(start, end)
                if mesh_sequence[j].get("valid")
            ]
            
            if len(valid_in_window) <= 1:
                smoothed.append(frame.copy())
                continue
            
            new_frame = frame.copy()
            
            # Smooth vertices
            if smooth_vertices and frame.get("vertices") is not None:
                vert_stack = np.stack([
                    np.array(f["vertices"]) for f in valid_in_window
                    if f.get("vertices") is not None
                ])
                new_frame["vertices"] = np.mean(vert_stack, axis=0)
            
            # Smooth joints
            if smooth_joints and frame.get("joints") is not None:
                joint_stack = [
                    np.array(f["joints"]) for f in valid_in_window
                    if f.get("joints") is not None
                ]
                if len(joint_stack) > 1:
                    new_frame["joints"] = np.mean(np.stack(joint_stack), axis=0)
            
            smoothed.append(new_frame)
        
        return (smoothed,)


class ClearMeshSequence:
    """
    Clear accumulated mesh sequences from memory.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "sequence_id": ("STRING", {
                    "default": "animation_001",
                    "multiline": False
                }),
            },
            "optional": {
                "clear_all": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "clear"
    CATEGORY = "SAM3DBody2abc/Video"
    
    def clear(
        self,
        sequence_id: str = "animation_001",
        clear_all: bool = False,
    ) -> Tuple[str]:
        """Clear sequence(s)."""
        
        if clear_all:
            MeshSequenceAccumulator.clear_all()
            return ("Cleared all sequences",)
        else:
            MeshSequenceAccumulator.clear_sequence(sequence_id)
            return (f"Cleared sequence: {sequence_id}",)


class MayaCameraScript:
    """
    Generate Maya Python script to create a camera matching the SAM3DBody solve.
    Copy the output text and paste into Maya's Script Editor.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE",),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Connect to get resolution automatically from video frames"
                }),
                "sensor_width_mm": ("FLOAT", {
                    "default": 36.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Sensor width in mm (36=Full Frame, 23.5=APS-C)"
                }),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "Scale factor (1.0=meters, 100=cm)"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("maya_script",)
    FUNCTION = "generate_script"
    CATEGORY = "SAM3DBody2abc/Export"
    
    def generate_script(
        self,
        mesh_sequence: List[Dict],
        images: torch.Tensor = None,
        sensor_width_mm: float = 36.0,
        scale: float = 1.0,
    ) -> Tuple[str]:
        """Generate Maya Python script for camera setup."""
        
        # Get resolution from images if provided
        if images is not None:
            # images shape: [N, H, W, C]
            image_height = images.shape[1]
            image_width = images.shape[2]
        else:
            # Fallback - try to get from mesh_sequence metadata
            image_width = 1920
            image_height = 1080
            for frame in mesh_sequence:
                if frame.get("image_size"):
                    image_width, image_height = frame["image_size"]
                    break
        
        # Get data from first valid frame
        focal_px = None
        cam_t = None
        
        for frame in mesh_sequence:
            if frame.get("valid", False) or frame.get("vertices") is not None:
                focal_px = frame.get("focal_length")
                cam_t = frame.get("camera")
                break
        
        # Convert focal length
        if focal_px is not None:
            if hasattr(focal_px, 'cpu'):
                focal_px = focal_px.cpu().numpy()
            focal_px = float(np.array(focal_px).flatten()[0])
        else:
            focal_px = max(image_width, image_height)
        
        # Convert camera translation
        if cam_t is not None:
            if hasattr(cam_t, 'cpu'):
                cam_t = cam_t.cpu().numpy()
            cam_t = np.array(cam_t).flatten()
        else:
            cam_t = np.array([0.0, 0.3, 2.5])
        
        # Calculate Maya values
        focal_mm = focal_px * sensor_width_mm / image_width
        aspect = image_width / image_height
        sensor_height_mm = sensor_width_mm / aspect
        
        # Film aperture in inches (Maya uses inches)
        h_aperture = sensor_width_mm / 25.4
        v_aperture = sensor_height_mm / 25.4
        
        # Get camera delta for moving shots
        first_cam = cam_t
        last_cam = None
        for frame in reversed(mesh_sequence):
            if frame.get("valid", False) or frame.get("vertices") is not None:
                last_cam_data = frame.get("camera")
                if last_cam_data is not None:
                    if hasattr(last_cam_data, 'cpu'):
                        last_cam_data = last_cam_data.cpu().numpy()
                    last_cam = np.array(last_cam_data).flatten()
                    break
        
        if last_cam is None:
            last_cam = first_cam
        
        cam_delta = last_cam - first_cam
        
        # Generate script
        script = f'''# ============================================
# Maya Camera Script - Generated by SAM3DBody2abc
# ============================================
# Source: {image_width}x{image_height} pixels
# Focal Length: {focal_px:.2f}px → {focal_mm:.2f}mm
# Sensor: {sensor_width_mm:.1f}mm x {sensor_height_mm:.1f}mm
# Scale: {scale}x ({"meters" if scale == 1.0 else "cm" if scale == 100 else "custom"})
# Camera Frame 0: [{first_cam[0]:.4f}, {first_cam[1]:.4f}, {first_cam[2]:.4f}]
# Camera Frame N: [{last_cam[0]:.4f}, {last_cam[1]:.4f}, {last_cam[2]:.4f}]
# Camera Delta: [{cam_delta[0]:.4f}, {cam_delta[1]:.4f}, {cam_delta[2]:.4f}]
# ============================================

import maya.cmds as cmds

# Create camera
cam_transform, cam_shape = cmds.camera(name='sam3d_camera')

# Set focal length
cmds.setAttr(cam_shape + '.focalLength', {focal_mm:.4f})

# Set film back (sensor size)
cmds.setAttr(cam_shape + '.horizontalFilmAperture', {h_aperture:.6f})
cmds.setAttr(cam_shape + '.verticalFilmAperture', {v_aperture:.6f})

# Set film fit to horizontal (match width)
cmds.setAttr(cam_shape + '.filmFit', 1)  # 1 = Horizontal

# Camera at origin (static camera setup)
# The mesh has world_space transform baked in
cmds.setAttr(cam_transform + '.translateX', 0)
cmds.setAttr(cam_transform + '.translateY', 0)
cmds.setAttr(cam_transform + '.translateZ', 0)

# Camera rotation (looking down -Z axis)
cmds.setAttr(cam_transform + '.rotateX', 0)
cmds.setAttr(cam_transform + '.rotateY', 0)
cmds.setAttr(cam_transform + '.rotateZ', 0)

# Set near/far clip planes
cmds.setAttr(cam_shape + '.nearClipPlane', 0.01)
cmds.setAttr(cam_shape + '.farClipPlane', 10000)

# Set render resolution to match source
cmds.setAttr('defaultResolution.width', {image_width})
cmds.setAttr('defaultResolution.height', {image_height})
cmds.setAttr('defaultResolution.deviceAspectRatio', {aspect:.6f})

# Select the camera
cmds.select(cam_transform)
cmds.lookThru(cam_transform)

print("Camera created: " + cam_transform)
print("Focal length: {focal_mm:.2f}mm")
print("Film back: {sensor_width_mm:.1f}mm x {sensor_height_mm:.1f}mm")
'''
        
        return (script,)

