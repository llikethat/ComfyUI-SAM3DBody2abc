"""
Animated FBX Export for SAM3DBody2abc
Export full video sequences as animated FBX files with rigged skeleton.

This creates a SINGLE FBX file containing:
- Animated mesh (using shape keys or vertex animation)
- Animated skeleton with keyframes for each frame
- Proper skinning weights from MHR model

Requires Blender to be installed for FBX conversion.
"""

import os
import sys
import json
import time
import tempfile
import subprocess
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
import folder_paths

# Import shared find_blender function
from .animated_export import find_blender

# Timeout for Blender subprocess
BLENDER_TIMEOUT = 600  # 10 minutes for long animations

# Get path to our Blender script
_current_dir = os.path.dirname(os.path.abspath(__file__))
_lib_dir = os.path.join(os.path.dirname(_current_dir), "lib")
BLENDER_ANIM_SCRIPT = os.path.join(_lib_dir, "blender_export_animated_fbx.py")


def find_mhr_model_path():
    """Find MHR model path for skinning weights."""
    import glob
    
    # Check ComfyUI models folder
    sam3d_dir = os.path.join(folder_paths.models_dir, "sam3dbody", "assets", "mhr_model.pt")
    if os.path.exists(sam3d_dir):
        return sam3d_dir
    
    # Check HuggingFace cache
    hf_patterns = [
        os.path.expanduser("~/.cache/huggingface/hub/models--facebook--sam-3d-body-*/snapshots/*/assets/mhr_model.pt"),
        os.path.expanduser("~/.cache/huggingface/hub/models--facebook--sam-3d-body-dinov3/snapshots/*/assets/mhr_model.pt"),
    ]
    
    for pattern in hf_patterns:
        matches = glob.glob(pattern)
        if matches:
            return sorted(matches, key=os.path.getmtime, reverse=True)[0]
    
    return None


def extract_skinning_weights(mhr_path: str, num_vertices: int) -> Optional[List]:
    """Extract skinning weights from MHR model."""
    if not mhr_path or not os.path.exists(mhr_path):
        return None
    
    try:
        mhr_model = torch.jit.load(mhr_path, map_location='cpu')
        lbs = mhr_model.character_torch.linear_blend_skinning
        
        vert_indices = lbs.vert_indices_flattened.cpu().numpy().astype(int)
        skin_indices = lbs.skin_indices_flattened.cpu().numpy().astype(int)
        skin_weights = lbs.skin_weights_flattened.cpu().numpy().astype(float)
        
        # Build per-vertex weight list
        vertex_weights = {}
        for i in range(len(vert_indices)):
            vert_idx = int(vert_indices[i])
            bone_idx = int(skin_indices[i])
            weight = float(skin_weights[i])
            
            if vert_idx not in vertex_weights:
                vertex_weights[vert_idx] = []
            vertex_weights[vert_idx].append([bone_idx, weight])
        
        # Convert to list format
        skinning_data = []
        for vert_idx in range(num_vertices):
            if vert_idx in vertex_weights:
                skinning_data.append(vertex_weights[vert_idx])
            else:
                skinning_data.append([])
        
        return skinning_data
        
    except Exception as e:
        print(f"[AnimatedFBX] Could not extract skinning weights: {e}")
        return None


def get_joint_parents(mhr_path: str) -> Optional[List[int]]:
    """Get joint parent hierarchy from MHR model."""
    if not mhr_path or not os.path.exists(mhr_path):
        return None
    
    try:
        mhr_model = torch.jit.load(mhr_path, map_location='cpu')
        parents = mhr_model.character_torch.skeleton.joint_parents.cpu().numpy().astype(int).tolist()
        return parents
    except Exception as e:
        print(f"[AnimatedFBX] Could not extract joint parents: {e}")
        return None


class ExportAnimatedFBX:
    """
    Export mesh sequence to animated FBX format.
    
    Creates a SINGLE FBX file with:
    - Full animation timeline
    - Rigged skeleton with keyframes
    - Skinning weights for proper deformation
    
    Requires Blender to be installed.
    For FBX export without Blender, use BVH export instead.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE",),
                "filename": ("STRING", {
                    "default": "body_animation",
                    "multiline": False
                }),
                "fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 1.0
                }),
            },
            "optional": {
                "output_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Leave empty for ComfyUI output folder"
                }),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.001,
                    "max": 1000.0,
                    "step": 0.01,
                    "tooltip": "Scale factor (1.0=meters)"
                }),
                "include_mesh": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include animated mesh geometry"
                }),
                "include_skeleton": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include animated skeleton"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("file_path", "status", "exported_frames")
    FUNCTION = "export_fbx"
    CATEGORY = "SAM3DBody2abc/Export"
    OUTPUT_NODE = True
    
    def export_fbx(
        self,
        mesh_sequence: List[Dict],
        filename: str = "body_animation",
        fps: float = 30.0,
        output_dir: str = "",
        scale: float = 1.0,
        include_mesh: bool = True,
        include_skeleton: bool = True,
    ) -> Tuple[str, str, int]:
        """Export mesh sequence to animated FBX."""
        
        if not mesh_sequence:
            return ("", "Error: Empty mesh sequence", 0)
        
        # Check Blender
        blender_exe = find_blender()
        if not blender_exe:
            return ("", "Error: Blender not found. Install Blender or set BLENDER_PATH environment variable.", 0)
        
        if not os.path.exists(BLENDER_ANIM_SCRIPT):
            return ("", f"Error: Blender script not found: {BLENDER_ANIM_SCRIPT}", 0)
        
        # Get output directory
        if not output_dir:
            output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter valid frames
        valid_frames = []
        for frame in mesh_sequence:
            if frame.get("valid") and frame.get("vertices") is not None:
                valid_frames.append(frame)
        
        if not valid_frames:
            return ("", "Error: No valid frames in sequence", 0)
        
        print(f"[AnimatedFBX] Processing {len(valid_frames)} frames at {fps} fps")
        
        # Get mesh data from first frame
        first_verts = valid_frames[0]["vertices"]
        if isinstance(first_verts, torch.Tensor):
            first_verts = first_verts.cpu().numpy()
        num_vertices = len(first_verts)
        
        # Get faces (same for all frames)
        faces = valid_frames[0].get("faces")
        if faces is None:
            return ("", "Error: No face data in mesh sequence", 0)
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
        faces = faces.astype(int).tolist()
        
        # Get MHR model for skinning weights
        mhr_path = find_mhr_model_path()
        skinning_weights = None
        joint_parents = None
        
        if mhr_path:
            print(f"[AnimatedFBX] Found MHR model: {mhr_path}")
            skinning_weights = extract_skinning_weights(mhr_path, num_vertices)
            joint_parents = get_joint_parents(mhr_path)
            if skinning_weights:
                print(f"[AnimatedFBX] Extracted skinning weights for {len(skinning_weights)} vertices")
            if joint_parents:
                print(f"[AnimatedFBX] Extracted joint hierarchy ({len(joint_parents)} joints)")
        
        # Build animation data structure
        anim_data = {
            "fps": fps,
            "faces": faces,
            "num_joints": 127,
            "joint_parents": joint_parents,
            "skinning_weights": skinning_weights,
            "include_mesh": include_mesh,
            "include_skeleton": include_skeleton,
            "frames": [],
        }
        
        # Process each frame
        for frame in valid_frames:
            vertices = frame["vertices"]
            if isinstance(vertices, torch.Tensor):
                vertices = vertices.cpu().numpy()
            
            # Apply scale
            vertices = vertices * scale
            
            frame_data = {
                "frame_index": frame.get("frame_index", len(anim_data["frames"])),
                "vertices": vertices.tolist(),
            }
            
            # Add joint positions if available
            joints = frame.get("joints")
            if joints is not None:
                if isinstance(joints, torch.Tensor):
                    joints = joints.cpu().numpy()
                joints = joints * scale
                frame_data["joint_positions"] = joints.tolist()
            
            anim_data["frames"].append(frame_data)
        
        # Write animation JSON to temp file
        temp_dir = folder_paths.get_temp_directory()
        json_path = os.path.join(temp_dir, f"anim_data_{int(time.time())}.json")
        
        with open(json_path, 'w') as f:
            json.dump(anim_data, f)
        
        print(f"[AnimatedFBX] Animation data written to {json_path}")
        
        # Prepare output path
        if not filename.endswith('.fbx'):
            filename = filename + '.fbx'
        output_path = os.path.join(output_dir, filename)
        
        # Handle existing files
        counter = 1
        base_name = filename[:-4]
        while os.path.exists(output_path):
            output_path = os.path.join(output_dir, f"{base_name}_{counter:04d}.fbx")
            counter += 1
        
        # Run Blender export
        print(f"[AnimatedFBX] Running Blender export...")
        cmd = [
            blender_exe,
            "--background",
            "--python", BLENDER_ANIM_SCRIPT,
            "--",
            json_path,
            output_path,
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=BLENDER_TIMEOUT
            )
            
            # Print Blender output for debugging
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if '[AnimatedFBX]' in line or 'Error' in line.lower():
                        print(line)
            
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else "Unknown error"
                return ("", f"Error: Blender export failed: {error_msg[:200]}", 0)
            
            if not os.path.exists(output_path):
                return ("", "Error: Export completed but FBX file not created", 0)
            
            # Clean up temp file
            try:
                os.unlink(json_path)
            except:
                pass
            
            status = f"Exported {len(valid_frames)} frames to FBX"
            print(f"[AnimatedFBX] {status}")
            print(f"[AnimatedFBX] Output: {output_path}")
            
            return (output_path, status, len(valid_frames))
            
        except subprocess.TimeoutExpired:
            return ("", f"Error: Blender export timed out after {BLENDER_TIMEOUT}s", 0)
        except Exception as e:
            return ("", f"Error: {str(e)}", 0)


# Node registration
NODE_CLASS_MAPPINGS = {
    "SAM3DBody2abc_ExportAnimatedFBX": ExportAnimatedFBX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBody2abc_ExportAnimatedFBX": "📦 Export Animated FBX",
}
