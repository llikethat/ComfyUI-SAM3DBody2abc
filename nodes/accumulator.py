"""
Mesh Data Accumulator for SAM3DBody2abc
Accumulates SAM3D_OUTPUT (mesh_data) from SAM3DBody Process node.

SAM3DBody Process outputs:
- mesh_data (SAM3D_OUTPUT): vertices, faces, joints, joint_coords, joint_rotations, etc.
- skeleton (SKELETON): joint_positions, joint_rotations, pose_params
- debug_image (IMAGE)

This node accumulates mesh_data across frames for animated FBX export.
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
import folder_paths


def to_list(obj):
    """Convert numpy/torch to list for JSON serialization."""
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_list(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_list(v) for v in obj]
    return obj


def to_numpy(data):
    """Convert tensor to numpy."""
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    if isinstance(data, np.ndarray):
        return data.copy()
    return data


class MeshDataAccumulator:
    """
    Accumulate mesh_data (SAM3D_OUTPUT) from SAM3DBody Process across frames.
    
    Stores per-frame:
    - vertices (for shape keys)
    - joint_coords (127 joints for skeleton animation - positions)
    - joint_rotations (127 joints for skeleton animation - rotation matrices 3x3)
    - pred_cam_t (camera translation for world positioning)
    
    Stores once (from first frame):
    - faces
    - joint_parents (hierarchy)
    - mhr_path (for skinning weights)
    """
    
    # Class-level storage
    _sequences: Dict[str, Dict] = {}
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT", {
                    "tooltip": "mesh_data from SAM3DBody Process node"
                }),
                "sequence_id": ("STRING", {
                    "default": "animation",
                }),
                "frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                }),
            },
            "optional": {
                "reset": ("BOOLEAN", {
                    "default": False,
                }),
            }
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE", "INT", "STRING")
    RETURN_NAMES = ("mesh_sequence", "frame_count", "status")
    FUNCTION = "accumulate"
    CATEGORY = "SAM3DBody2abc"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
    
    def accumulate(
        self,
        mesh_data: Dict,
        sequence_id: str = "animation",
        frame_index: int = 0,
        reset: bool = False,
    ) -> Tuple[Dict, int, str]:
        """Accumulate mesh_data frame."""
        
        # Initialize or reset
        if reset or sequence_id not in self._sequences:
            self._sequences[sequence_id] = {
                "frames": {},
                "faces": None,
                "joint_parents": None,
                "mhr_path": None,
            }
        
        seq = self._sequences[sequence_id]
        
        # Store per-frame data - including joint_rotations now
        frame = {
            "vertices": to_numpy(mesh_data.get("vertices")),
            "joint_coords": to_numpy(mesh_data.get("joint_coords")),
            "joint_rotations": to_numpy(mesh_data.get("joint_rotations")),
            "camera": to_numpy(mesh_data.get("camera")),
            "pred_cam_t": to_numpy(mesh_data.get("camera")),  # Alias for compatibility
            "focal_length": mesh_data.get("focal_length"),
        }
        seq["frames"][frame_index] = frame
        
        # Store shared data (first frame)
        if seq["faces"] is None:
            seq["faces"] = to_numpy(mesh_data.get("faces"))
        
        if seq["mhr_path"] is None:
            seq["mhr_path"] = mesh_data.get("mhr_path")
        
        # Get joint parents from mesh_data
        if seq["joint_parents"] is None:
            joint_rotations = mesh_data.get("joint_rotations")
            if isinstance(joint_rotations, dict) and "joint_parents" in joint_rotations:
                seq["joint_parents"] = to_numpy(joint_rotations["joint_parents"])
            elif mesh_data.get("joint_parents") is not None:
                seq["joint_parents"] = to_numpy(mesh_data.get("joint_parents"))
        
        # Build output
        mesh_sequence = {
            "sequence_id": sequence_id,
            "frames": seq["frames"],
            "faces": seq["faces"],
            "joint_parents": seq["joint_parents"],
            "mhr_path": seq["mhr_path"],
        }
        
        frame_count = len(seq["frames"])
        has_rotations = frame.get("joint_rotations") is not None
        status = f"Frame {frame_index} added (rotations={'yes' if has_rotations else 'no'}). Total: {frame_count}"
        
        return (mesh_sequence, frame_count, status)
    
    @classmethod
    def clear_all(cls):
        cls._sequences.clear()


class ExportMeshSequenceJSON:
    """
    Export accumulated mesh sequence to JSON.
    This JSON can be used for animated FBX export.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE",),
                "filename": ("STRING", {
                    "default": "mesh_animation",
                }),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                }),
            },
            "optional": {
                "include_vertices": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include mesh vertices for shape keys"
                }),
                "include_rotations": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include joint rotations (3x3 matrices)"
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
        mesh_sequence: Dict,
        filename: str = "mesh_animation",
        fps: float = 24.0,
        include_vertices: bool = True,
        include_rotations: bool = True,
    ) -> Tuple[str, str, int]:
        """Export to JSON."""
        
        frames = mesh_sequence.get("frames", {})
        if not frames:
            return ("", "Error: No frames", 0)
        
        output_dir = folder_paths.get_output_directory()
        sorted_indices = sorted(frames.keys())
        
        export_data = {
            "fps": fps,
            "frame_count": len(sorted_indices),
            "include_vertices": include_vertices,
            "include_rotations": include_rotations,
            "faces": to_list(mesh_sequence.get("faces")),
            "joint_parents": to_list(mesh_sequence.get("joint_parents")),
            "mhr_path": mesh_sequence.get("mhr_path"),
            "frames": [],
        }
        
        for idx in sorted_indices:
            frame = frames[idx]
            frame_data = {
                "frame_index": idx,
                "joint_coords": to_list(frame.get("joint_coords")),
            }
            if include_vertices:
                frame_data["vertices"] = to_list(frame.get("vertices"))
            if include_rotations:
                frame_data["joint_rotations"] = to_list(frame.get("joint_rotations"))
            export_data["frames"].append(frame_data)
        
        json_path = os.path.join(output_dir, f"{filename}.json")
        with open(json_path, 'w') as f:
            json.dump(export_data, f)
        
        status = f"Exported {len(sorted_indices)} frames"
        if include_rotations:
            status += " (with rotations)"
        print(f"[SAM3DBody2abc] {status} to {filename}.json")
        
        return (json_path, status, len(sorted_indices))


class ClearAccumulator:
    """Clear accumulated data."""
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "confirm": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "clear"
    CATEGORY = "SAM3DBody2abc"
    
    def clear(self, confirm: bool = True) -> Tuple[str]:
        if confirm:
            MeshDataAccumulator.clear_all()
            return ("Cleared all sequences",)
        return ("Not cleared",)
