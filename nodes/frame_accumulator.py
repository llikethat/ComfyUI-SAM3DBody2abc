"""
Frame Accumulator for SAM3DBody2abc
Accumulates SAM3D_OUTPUT frames from SAM3DBody Process Image node.
Applies temporal smoothing and exports to JSON for FBX conversion.
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from scipy.ndimage import gaussian_filter1d
import folder_paths


class FrameAccumulator:
    """
    Accumulate frames from SAM3DBody Process Image node.
    
    Workflow:
    1. Connect SAM3DBody Process Image → mesh_data to this node
    2. Process frames one at a time (or batch)
    3. Apply temporal smoothing
    4. Export to JSON for FBX conversion
    """
    
    # Class-level storage for sequences
    _sequences: Dict[str, Dict] = {}
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT",),
                "sequence_id": ("STRING", {
                    "default": "animation",
                    "multiline": False,
                    "tooltip": "Unique ID for this animation sequence"
                }),
                "frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "tooltip": "Current frame number"
                }),
            },
            "optional": {
                "reset": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reset sequence and start fresh"
                }),
                "smoothing_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Temporal smoothing (0=none, 1=moderate, 2=heavy)"
                }),
            }
        }
    
    RETURN_TYPES = ("FRAME_SEQUENCE", "INT", "STRING")
    RETURN_NAMES = ("frame_sequence", "frame_count", "status")
    FUNCTION = "accumulate"
    CATEGORY = "SAM3DBody2abc"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # Always execute
    
    def _to_numpy(self, data):
        """Convert tensor to numpy array."""
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        if isinstance(data, np.ndarray):
            return data.copy()
        if isinstance(data, list):
            return np.array(data)
        return data
    
    def accumulate(
        self,
        mesh_data: Dict,
        sequence_id: str = "animation",
        frame_index: int = 0,
        reset: bool = False,
        smoothing_strength: float = 0.5,
    ) -> Tuple[Dict, int, str]:
        """
        Accumulate a frame into the sequence.
        """
        # Initialize or reset sequence
        if reset or sequence_id not in self._sequences:
            self._sequences[sequence_id] = {
                "frames": {},
                "faces": None,
                "smoothing_strength": smoothing_strength,
                "mhr_path": None,
            }
        
        seq = self._sequences[sequence_id]
        seq["smoothing_strength"] = smoothing_strength
        
        # Extract frame data from SAM3D_OUTPUT
        vertices = self._to_numpy(mesh_data.get("vertices"))
        joints = self._to_numpy(mesh_data.get("joints")) or self._to_numpy(mesh_data.get("joint_coords"))
        camera = self._to_numpy(mesh_data.get("camera"))
        focal_length = mesh_data.get("focal_length")
        
        if focal_length is not None:
            if isinstance(focal_length, (torch.Tensor, np.ndarray)):
                focal_length = float(focal_length.flatten()[0])
        
        # Store faces (same for all frames)
        if seq["faces"] is None and mesh_data.get("faces") is not None:
            seq["faces"] = self._to_numpy(mesh_data.get("faces"))
        
        # Store MHR path
        if seq["mhr_path"] is None:
            seq["mhr_path"] = mesh_data.get("mhr_path")
        
        # Store frame
        frame_data = {
            "vertices": vertices,
            "joints": joints,
            "camera": camera,
            "focal_length": focal_length,
            "valid": vertices is not None,
        }
        
        seq["frames"][frame_index] = frame_data
        
        # Build output
        frame_sequence = {
            "sequence_id": sequence_id,
            "frames": seq["frames"],
            "faces": seq["faces"],
            "smoothing_strength": smoothing_strength,
            "mhr_path": seq["mhr_path"],
        }
        
        frame_count = len(seq["frames"])
        status = f"Frame {frame_index} added. Total: {frame_count} frames"
        
        return (frame_sequence, frame_count, status)
    
    @classmethod
    def get_sequence(cls, sequence_id: str) -> Optional[Dict]:
        return cls._sequences.get(sequence_id)
    
    @classmethod
    def clear_sequence(cls, sequence_id: str):
        if sequence_id in cls._sequences:
            del cls._sequences[sequence_id]
    
    @classmethod
    def clear_all(cls):
        cls._sequences.clear()


class ApplySmoothing:
    """
    Apply temporal smoothing to accumulated frames.
    Call this after all frames are accumulated.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "frame_sequence": ("FRAME_SEQUENCE",),
            },
            "optional": {
                "smoothing_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Override smoothing (0=none, 1=moderate, 2=heavy)"
                }),
                "smoothing_radius": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 15,
                    "tooltip": "Number of neighboring frames to consider"
                }),
            }
        }
    
    RETURN_TYPES = ("FRAME_SEQUENCE", "STRING")
    RETURN_NAMES = ("smoothed_sequence", "status")
    FUNCTION = "apply_smoothing"
    CATEGORY = "SAM3DBody2abc"
    
    def apply_smoothing(
        self,
        frame_sequence: Dict,
        smoothing_strength: float = 0.5,
        smoothing_radius: int = 3,
    ) -> Tuple[Dict, str]:
        """
        Apply Gaussian temporal smoothing to vertices and joints.
        """
        frames = frame_sequence.get("frames", {})
        
        if len(frames) < 3:
            return (frame_sequence, f"Skipped: Only {len(frames)} frames (need 3+)")
        
        if smoothing_strength <= 0:
            return (frame_sequence, "Skipped: smoothing_strength=0")
        
        # Sort frames by index
        sorted_indices = sorted(frames.keys())
        
        # Stack vertices and joints
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
            return (frame_sequence, f"Skipped: Only {len(vertices_stack)} valid frames")
        
        # Convert to arrays (T, V, 3)
        vertices_array = np.stack(vertices_stack, axis=0)
        
        # Apply Gaussian smoothing along time axis
        sigma = smoothing_strength * smoothing_radius
        smoothed_vertices = gaussian_filter1d(vertices_array, sigma=sigma, axis=0, mode='nearest')
        
        # Smooth joints if available
        smoothed_joints = None
        if len(joints_stack) == len(vertices_stack):
            joints_array = np.stack(joints_stack, axis=0)
            smoothed_joints = gaussian_filter1d(joints_array, sigma=sigma, axis=0, mode='nearest')
        
        # Update frames with smoothed data
        smoothed_frames = dict(frames)
        for i, idx in enumerate(valid_indices):
            smoothed_frames[idx] = dict(frames[idx])
            smoothed_frames[idx]["vertices"] = smoothed_vertices[i]
            if smoothed_joints is not None:
                smoothed_frames[idx]["joints"] = smoothed_joints[i]
        
        # Build output
        smoothed_sequence = {
            "sequence_id": frame_sequence.get("sequence_id", "animation"),
            "frames": smoothed_frames,
            "faces": frame_sequence.get("faces"),
            "smoothing_strength": smoothing_strength,
            "mhr_path": frame_sequence.get("mhr_path"),
        }
        
        status = f"Smoothed {len(valid_indices)} frames (sigma={sigma:.2f})"
        
        return (smoothed_sequence, status)


class ExportSequenceJSON:
    """
    Export frame sequence to JSON file for FBX conversion.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "frame_sequence": ("FRAME_SEQUENCE",),
                "filename": ("STRING", {
                    "default": "animation",
                    "multiline": False,
                }),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                }),
            },
            "optional": {
                "output_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Leave empty for ComfyUI output folder"
                }),
                "include_mesh": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include mesh vertices in export"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("json_path", "status", "frame_count")
    FUNCTION = "export_json"
    CATEGORY = "SAM3DBody2abc/Export"
    OUTPUT_NODE = True
    
    def export_json(
        self,
        frame_sequence: Dict,
        filename: str = "animation",
        fps: float = 24.0,
        output_dir: str = "",
        include_mesh: bool = True,
    ) -> Tuple[str, str, int]:
        """
        Export sequence to JSON.
        """
        frames = frame_sequence.get("frames", {})
        faces = frame_sequence.get("faces")
        
        if not frames:
            return ("", "Error: No frames to export", 0)
        
        # Determine output directory
        if not output_dir:
            output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        # Build JSON structure
        sorted_indices = sorted(frames.keys())
        
        export_data = {
            "fps": fps,
            "frame_count": len(sorted_indices),
            "include_mesh": include_mesh,
            "frames": [],
        }
        
        # Add faces if including mesh
        if include_mesh and faces is not None:
            export_data["faces"] = faces.tolist() if isinstance(faces, np.ndarray) else faces
        
        # Add frames
        for idx in sorted_indices:
            frame = frames[idx]
            frame_export = {
                "frame_index": idx,
            }
            
            # Add vertices if including mesh
            if include_mesh and frame.get("vertices") is not None:
                verts = frame["vertices"]
                frame_export["vertices"] = verts.tolist() if isinstance(verts, np.ndarray) else verts
            
            # Always include joints (skeleton)
            if frame.get("joints") is not None:
                joints = frame["joints"]
                frame_export["joints"] = joints.tolist() if isinstance(joints, np.ndarray) else joints
            
            # Include camera
            if frame.get("camera") is not None:
                cam = frame["camera"]
                frame_export["camera"] = cam.tolist() if isinstance(cam, np.ndarray) else cam
            
            if frame.get("focal_length") is not None:
                frame_export["focal_length"] = float(frame["focal_length"])
            
            export_data["frames"].append(frame_export)
        
        # Write JSON
        json_path = os.path.join(output_dir, f"{filename}.json")
        with open(json_path, 'w') as f:
            json.dump(export_data, f)
        
        status = f"Exported {len(sorted_indices)} frames to {filename}.json"
        if not include_mesh:
            status += " (skeleton only)"
        
        print(f"[SAM3DBody2abc] {status}")
        
        return (json_path, status, len(sorted_indices))


class ClearSequences:
    """
    Clear accumulated frame sequences.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "clear_all": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "sequence_id": ("STRING", {
                    "default": "",
                    "tooltip": "Specific sequence to clear (if not clearing all)"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "clear"
    CATEGORY = "SAM3DBody2abc"
    
    def clear(
        self,
        clear_all: bool = True,
        sequence_id: str = "",
    ) -> Tuple[str]:
        """Clear sequences."""
        if clear_all:
            FrameAccumulator.clear_all()
            return ("Cleared all sequences",)
        elif sequence_id:
            FrameAccumulator.clear_sequence(sequence_id)
            return (f"Cleared sequence: {sequence_id}",)
        else:
            return ("Nothing to clear",)
