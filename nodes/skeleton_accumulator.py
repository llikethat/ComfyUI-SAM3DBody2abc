"""
Skeleton Accumulator for SAM3DBody2abc
Accumulates SKELETON outputs from SAM3DBody Process node over multiple frames.
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
import folder_paths


def convert_to_serializable(obj):
    """Convert numpy/torch types to JSON-serializable Python types."""
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
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(v) for v in obj]
    return obj


class SkeletonAccumulator:
    """
    Accumulate SKELETON data from SAM3DBody Process node across video frames.
    
    Workflow:
    1. Process video frame by frame with SAM3DBody Process
    2. Feed each SKELETON output to this accumulator
    3. When all frames are collected, output goes to Export Animated FBX
    """
    
    # Class-level storage
    _sequences: Dict[str, Dict] = {}
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "skeleton": ("SKELETON", {
                    "tooltip": "Skeleton from SAM3DBody Process node"
                }),
                "sequence_id": ("STRING", {
                    "default": "animation",
                    "multiline": False,
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
                    "tooltip": "Clear sequence and start fresh"
                }),
            }
        }
    
    RETURN_TYPES = ("SKELETON_SEQUENCE", "INT", "STRING")
    RETURN_NAMES = ("skeleton_sequence", "frame_count", "status")
    FUNCTION = "accumulate"
    CATEGORY = "SAM3DBody2abc"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
    
    def _to_numpy(self, data):
        """Convert tensor to numpy."""
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        if isinstance(data, np.ndarray):
            return data.copy()
        return data
    
    def accumulate(
        self,
        skeleton: Dict,
        sequence_id: str = "animation",
        frame_index: int = 0,
        reset: bool = False,
    ) -> Tuple[Dict, int, str]:
        """Accumulate a skeleton frame."""
        
        # Initialize or reset
        if reset or sequence_id not in self._sequences:
            self._sequences[sequence_id] = {
                "frames": {},
                "joint_parents": None,
                "joint_names": None,
            }
        
        seq = self._sequences[sequence_id]
        
        # Extract and store frame data
        frame_data = {
            "joint_positions": self._to_numpy(skeleton.get("joint_positions")),
            "joint_rotations": self._to_numpy(skeleton.get("joint_rotations")),
            "camera": self._to_numpy(skeleton.get("camera")),
            "focal_length": skeleton.get("focal_length"),
            "global_rot": self._to_numpy(skeleton.get("global_rot")),
        }
        
        seq["frames"][frame_index] = frame_data
        
        # Store joint hierarchy (same for all frames)
        if seq["joint_parents"] is None and skeleton.get("joint_parents") is not None:
            seq["joint_parents"] = self._to_numpy(skeleton.get("joint_parents"))
        
        if seq["joint_names"] is None and skeleton.get("joint_names") is not None:
            seq["joint_names"] = skeleton.get("joint_names")
        
        # Build output
        skeleton_sequence = {
            "sequence_id": sequence_id,
            "frames": seq["frames"],
            "joint_parents": seq["joint_parents"],
            "joint_names": seq["joint_names"],
        }
        
        frame_count = len(seq["frames"])
        status = f"Frame {frame_index} added. Total: {frame_count} frames"
        
        return (skeleton_sequence, frame_count, status)
    
    @classmethod
    def clear_all(cls):
        cls._sequences.clear()


class ExportSkeletonSequenceJSON:
    """
    Export accumulated skeleton sequence to JSON file.
    This JSON can be used for animated FBX export.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "skeleton_sequence": ("SKELETON_SEQUENCE",),
                "filename": ("STRING", {
                    "default": "skeleton_animation",
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
                    "tooltip": "Leave empty for ComfyUI output folder"
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
        skeleton_sequence: Dict,
        filename: str = "skeleton_animation",
        fps: float = 24.0,
        output_dir: str = "",
    ) -> Tuple[str, str, int]:
        """Export skeleton sequence to JSON."""
        
        frames = skeleton_sequence.get("frames", {})
        
        if not frames:
            return ("", "Error: No frames to export", 0)
        
        # Output directory
        if not output_dir:
            output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        # Sort frames
        sorted_indices = sorted(frames.keys())
        
        # Build export data
        export_data = {
            "fps": fps,
            "frame_count": len(sorted_indices),
            "joint_parents": convert_to_serializable(skeleton_sequence.get("joint_parents")),
            "joint_names": skeleton_sequence.get("joint_names"),
            "frames": [],
        }
        
        for idx in sorted_indices:
            frame = frames[idx]
            frame_export = {
                "frame_index": idx,
                "joint_positions": convert_to_serializable(frame.get("joint_positions")),
                "joint_rotations": convert_to_serializable(frame.get("joint_rotations")),
                "camera": convert_to_serializable(frame.get("camera")),
                "focal_length": convert_to_serializable(frame.get("focal_length")),
                "global_rot": convert_to_serializable(frame.get("global_rot")),
            }
            export_data["frames"].append(frame_export)
        
        # Write JSON
        json_path = os.path.join(output_dir, f"{filename}.json")
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        status = f"Exported {len(sorted_indices)} frames to {filename}.json"
        print(f"[SAM3DBody2abc] {status}")
        
        return (json_path, status, len(sorted_indices))


class ClearAccumulator:
    """Clear accumulated skeleton sequences."""
    
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
            SkeletonAccumulator.clear_all()
            return ("Cleared all sequences",)
        return ("Not cleared",)
