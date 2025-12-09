"""
FBX Export Node for SAM3DBody2abc
Exports skeleton sequence to animated FBX using Blender.

Settings:
- Scale: 1.0 (fixed)
- Up axis: Y (fixed)
"""

import os
import sys
import json
import subprocess
import shutil
import glob
import tempfile
import numpy as np
import torch
from typing import Dict, Tuple, Any, Optional
import folder_paths

# Timeout for Blender
BLENDER_TIMEOUT = 600

# Path to Blender script
_current_dir = os.path.dirname(os.path.abspath(__file__))
_lib_dir = os.path.join(os.path.dirname(_current_dir), "lib")
BLENDER_SCRIPT = os.path.join(_lib_dir, "blender_animated_fbx.py")

# Blender path cache
_BLENDER_PATH = None


def find_blender() -> Optional[str]:
    """Find Blender executable."""
    global _BLENDER_PATH
    
    if _BLENDER_PATH is not None:
        return _BLENDER_PATH
    
    locations = [
        shutil.which("blender"),
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ]
    
    # Check ComfyUI SAM3DBody bundled Blender
    try:
        custom_nodes = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        patterns = [
            os.path.join(custom_nodes, "ComfyUI-SAM3DBody", "lib", "blender", "blender-*-linux-x64", "blender"),
            os.path.join(custom_nodes, "ComfyUI-SAM3DBody", "lib", "blender", "*", "blender"),
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            locations.extend(matches)
    except Exception:
        pass
    
    # Windows
    for ver in ["4.2", "4.1", "4.0", "3.6"]:
        locations.append(f"C:\\Program Files\\Blender Foundation\\Blender {ver}\\blender.exe")
    
    for loc in locations:
        if loc and os.path.exists(loc):
            _BLENDER_PATH = loc
            print(f"[FBX Export] Found Blender: {loc}")
            return loc
    
    return None


def convert_to_serializable(obj):
    """Convert numpy/torch to JSON-serializable types."""
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


class ExportAnimatedFBX:
    """
    Export skeleton sequence to animated FBX.
    
    Uses Blender to create armature with keyframed joint positions.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "skeleton_sequence": ("SKELETON_SEQUENCE",),
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
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("fbx_path", "status", "frame_count")
    FUNCTION = "export_fbx"
    CATEGORY = "SAM3DBody2abc/Export"
    OUTPUT_NODE = True
    
    def export_fbx(
        self,
        skeleton_sequence: Dict,
        filename: str = "animation",
        fps: float = 24.0,
        output_dir: str = "",
    ) -> Tuple[str, str, int]:
        """Export to animated FBX."""
        
        frames = skeleton_sequence.get("frames", {})
        
        if not frames:
            return ("", "Error: No frames", 0)
        
        # Find Blender
        blender_path = find_blender()
        if not blender_path:
            return ("", "Error: Blender not found", 0)
        
        if not os.path.exists(BLENDER_SCRIPT):
            return ("", f"Error: Blender script not found: {BLENDER_SCRIPT}", 0)
        
        # Output directory
        if not output_dir:
            output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        # Sort frames
        sorted_indices = sorted(frames.keys())
        
        # Build JSON for Blender
        export_data = {
            "fps": fps,
            "frame_count": len(sorted_indices),
            "joint_parents": convert_to_serializable(skeleton_sequence.get("joint_parents")),
            "joint_names": skeleton_sequence.get("joint_names"),
            "frames": [],
        }
        
        for idx in sorted_indices:
            frame = frames[idx]
            export_data["frames"].append({
                "frame_index": idx,
                "joint_positions": convert_to_serializable(frame.get("joint_positions")),
                "joint_rotations": convert_to_serializable(frame.get("joint_rotations")),
                "global_rot": convert_to_serializable(frame.get("global_rot")),
            })
        
        # Write temp JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(export_data, f)
            json_path = f.name
        
        fbx_path = os.path.join(output_dir, f"{filename}.fbx")
        
        try:
            # Run Blender
            cmd = [
                blender_path,
                "--background",
                "--python", BLENDER_SCRIPT,
                "--",
                json_path,
                fbx_path,
            ]
            
            print(f"[FBX Export] Exporting {len(sorted_indices)} frames...")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=BLENDER_TIMEOUT,
            )
            
            if result.returncode != 0:
                error = result.stderr[:500] if result.stderr else "Unknown error"
                print(f"[FBX Export] Blender error: {error}")
                return ("", f"Blender error: {error}", 0)
            
            if not os.path.exists(fbx_path):
                return ("", "Error: FBX not created", 0)
            
            status = f"Exported {len(sorted_indices)} frames to {filename}.fbx"
            print(f"[FBX Export] {status}")
            
            return (fbx_path, status, len(sorted_indices))
            
        except subprocess.TimeoutExpired:
            return ("", f"Error: Blender timed out after {BLENDER_TIMEOUT}s", 0)
        except Exception as e:
            return ("", f"Error: {str(e)}", 0)
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)


class ExportAnimatedFBXFromJSON:
    """
    Export animated FBX from a previously saved skeleton JSON file.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "json_path": ("STRING", {
                    "forceInput": True,
                }),
            },
            "optional": {
                "output_dir": ("STRING", {
                    "default": "",
                }),
                "filename": ("STRING", {
                    "default": "animation",
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("fbx_path", "status")
    FUNCTION = "export_fbx"
    CATEGORY = "SAM3DBody2abc/Export"
    OUTPUT_NODE = True
    
    def export_fbx(
        self,
        json_path: str,
        output_dir: str = "",
        filename: str = "animation",
    ) -> Tuple[str, str]:
        """Convert JSON to FBX."""
        
        if not os.path.exists(json_path):
            return ("", f"Error: JSON not found: {json_path}")
        
        blender_path = find_blender()
        if not blender_path:
            return ("", "Error: Blender not found")
        
        if not os.path.exists(BLENDER_SCRIPT):
            return ("", f"Error: Script not found: {BLENDER_SCRIPT}")
        
        if not output_dir:
            output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        fbx_path = os.path.join(output_dir, f"{filename}.fbx")
        
        try:
            cmd = [
                blender_path,
                "--background",
                "--python", BLENDER_SCRIPT,
                "--",
                json_path,
                fbx_path,
            ]
            
            print(f"[FBX Export] Converting JSON to FBX...")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=BLENDER_TIMEOUT,
            )
            
            if result.returncode != 0:
                error = result.stderr[:500] if result.stderr else "Unknown error"
                return ("", f"Blender error: {error}")
            
            if not os.path.exists(fbx_path):
                return ("", "Error: FBX not created")
            
            return (fbx_path, f"Exported to {filename}.fbx")
            
        except subprocess.TimeoutExpired:
            return ("", f"Error: Blender timed out")
        except Exception as e:
            return ("", f"Error: {str(e)}")
