"""
FBX Export for SAM3DBody2abc
Convert JSON sequence to animated FBX file.

Fixed settings:
- Scale: 1.0
- Up axis: Y
- Coordinate system: Blender/Maya compatible

Requires Blender for FBX conversion.
"""

import os
import sys
import json
import subprocess
import shutil
import glob
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import folder_paths

# Timeout for Blender subprocess (10 minutes for long animations)
BLENDER_TIMEOUT = 600

# Get path to Blender script
_current_dir = os.path.dirname(os.path.abspath(__file__))
_lib_dir = os.path.join(os.path.dirname(_current_dir), "lib")
BLENDER_SCRIPT = os.path.join(_lib_dir, "blender_fbx_export.py")

# Global Blender path cache
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
    
    # Check ComfyUI custom_nodes for bundled Blender
    try:
        custom_nodes_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        patterns = [
            os.path.join(custom_nodes_dir, "ComfyUI-SAM3DBody", "lib", "blender", "blender-*-linux-x64", "blender"),
            os.path.join(custom_nodes_dir, "ComfyUI-SAM3DBody", "lib", "blender", "*", "blender"),
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                locations.extend(matches)
                break
    except Exception:
        pass
    
    # Windows paths
    for version in ["4.2", "4.1", "4.0", "3.6"]:
        locations.append(f"C:\\Program Files\\Blender Foundation\\Blender {version}\\blender.exe")
    
    for loc in locations:
        if loc and os.path.exists(loc):
            _BLENDER_PATH = loc
            print(f"[FBX Export] Found Blender: {loc}")
            return loc
    
    return None


class ExportFBX:
    """
    Export JSON sequence to animated FBX file.
    
    Settings:
    - Scale: 1.0 (fixed)
    - Up axis: Y (fixed)
    - Include mesh: optional
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "json_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to animation JSON file"
                }),
            },
            "optional": {
                "output_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Leave empty for ComfyUI output folder"
                }),
                "filename": ("STRING", {
                    "default": "animation",
                    "multiline": False,
                }),
                "include_mesh": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include animated mesh (False = skeleton only)"
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
        include_mesh: bool = True,
    ) -> Tuple[str, str]:
        """
        Convert JSON to FBX using Blender.
        """
        # Check Blender
        blender_path = find_blender()
        if not blender_path:
            return ("", "Error: Blender not found. Install Blender or use ComfyUI-SAM3DBody bundled version.")
        
        # Check JSON file
        if not os.path.exists(json_path):
            return ("", f"Error: JSON file not found: {json_path}")
        
        # Check Blender script
        if not os.path.exists(BLENDER_SCRIPT):
            return ("", f"Error: Blender script not found: {BLENDER_SCRIPT}")
        
        # Determine output directory
        if not output_dir:
            output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        # Output FBX path
        fbx_path = os.path.join(output_dir, f"{filename}.fbx")
        
        # Build Blender command
        cmd = [
            blender_path,
            "--background",
            "--python", BLENDER_SCRIPT,
            "--",
            json_path,
            fbx_path,
            "1" if include_mesh else "0",
        ]
        
        print(f"[FBX Export] Running Blender: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=BLENDER_TIMEOUT,
            )
            
            if result.returncode != 0:
                error_msg = result.stderr[:500] if result.stderr else "Unknown error"
                print(f"[FBX Export] Blender error: {error_msg}")
                return ("", f"Blender error: {error_msg}")
            
            if not os.path.exists(fbx_path):
                return ("", "Error: FBX file was not created")
            
            status = f"Exported: {filename}.fbx"
            if not include_mesh:
                status += " (skeleton only)"
            
            print(f"[FBX Export] {status}")
            
            return (fbx_path, status)
            
        except subprocess.TimeoutExpired:
            return ("", f"Error: Blender timed out after {BLENDER_TIMEOUT}s")
        except Exception as e:
            return ("", f"Error: {str(e)}")


class ExportFBXDirect:
    """
    Export FRAME_SEQUENCE directly to FBX (combines JSON export + FBX conversion).
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
                }),
                "include_mesh": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include animated mesh (False = skeleton only)"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("fbx_path", "status", "frame_count")
    FUNCTION = "export"
    CATEGORY = "SAM3DBody2abc/Export"
    OUTPUT_NODE = True
    
    def export(
        self,
        frame_sequence: Dict,
        filename: str = "animation",
        fps: float = 24.0,
        output_dir: str = "",
        include_mesh: bool = True,
    ) -> Tuple[str, str, int]:
        """
        Export sequence directly to FBX.
        """
        frames = frame_sequence.get("frames", {})
        faces = frame_sequence.get("faces")
        
        if not frames:
            return ("", "Error: No frames to export", 0)
        
        # Check Blender
        blender_path = find_blender()
        if not blender_path:
            return ("", "Error: Blender not found", 0)
        
        # Determine output directory
        if not output_dir:
            output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        # Build JSON for Blender
        sorted_indices = sorted(frames.keys())
        
        export_data = {
            "fps": fps,
            "frame_count": len(sorted_indices),
            "include_mesh": include_mesh,
            "frames": [],
        }
        
        if include_mesh and faces is not None:
            export_data["faces"] = faces.tolist() if isinstance(faces, np.ndarray) else faces
        
        for idx in sorted_indices:
            frame = frames[idx]
            frame_export = {"frame_index": idx}
            
            if include_mesh and frame.get("vertices") is not None:
                verts = frame["vertices"]
                frame_export["vertices"] = verts.tolist() if isinstance(verts, np.ndarray) else verts
            
            if frame.get("joints") is not None:
                joints = frame["joints"]
                frame_export["joints"] = joints.tolist() if isinstance(joints, np.ndarray) else joints
            
            if frame.get("camera") is not None:
                cam = frame["camera"]
                frame_export["camera"] = cam.tolist() if isinstance(cam, np.ndarray) else cam
            
            export_data["frames"].append(frame_export)
        
        # Write temp JSON
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(export_data, f)
            json_path = f.name
        
        try:
            # Output FBX path
            fbx_path = os.path.join(output_dir, f"{filename}.fbx")
            
            # Run Blender
            cmd = [
                blender_path,
                "--background",
                "--python", BLENDER_SCRIPT,
                "--",
                json_path,
                fbx_path,
                "1" if include_mesh else "0",
            ]
            
            print(f"[FBX Export] Exporting {len(sorted_indices)} frames...")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=BLENDER_TIMEOUT,
            )
            
            if result.returncode != 0:
                error_msg = result.stderr[:500] if result.stderr else "Unknown error"
                return ("", f"Blender error: {error_msg}", 0)
            
            if not os.path.exists(fbx_path):
                return ("", "Error: FBX not created", 0)
            
            status = f"Exported {len(sorted_indices)} frames"
            if not include_mesh:
                status += " (skeleton only)"
            
            return (fbx_path, status, len(sorted_indices))
            
        finally:
            # Clean up temp JSON
            if os.path.exists(json_path):
                os.unlink(json_path)
