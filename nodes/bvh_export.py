"""
BVH Export for SAM3DBody2abc
Export skeleton animation to BVH format - the universal motion capture format.

BVH files work in:
- Maya, Blender, Houdini, Cinema 4D
- Unity, Unreal Engine
- MotionBuilder, iClone
- Any software that supports mocap data
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import folder_paths


# MHR 127-joint to simplified skeleton mapping
# We export a clean 22-joint skeleton that's universally compatible
SKELETON_JOINTS = [
    "Hips",           # 0 - Root
    "Spine",          # 1
    "Spine1",         # 2
    "Spine2",         # 3
    "Neck",           # 4
    "Head",           # 5
    "LeftShoulder",   # 6
    "LeftArm",        # 7
    "LeftForeArm",    # 8
    "LeftHand",       # 9
    "RightShoulder",  # 10
    "RightArm",       # 11
    "RightForeArm",   # 12
    "RightHand",      # 13
    "LeftUpLeg",      # 14
    "LeftLeg",        # 15
    "LeftFoot",       # 16
    "LeftToeBase",    # 17
    "RightUpLeg",     # 18
    "RightLeg",       # 19
    "RightFoot",      # 20
    "RightToeBase",   # 21
]

# Parent indices for skeleton hierarchy (-1 = root)
SKELETON_PARENTS = [
    -1,  # Hips (root)
    0,   # Spine -> Hips
    1,   # Spine1 -> Spine
    2,   # Spine2 -> Spine1
    3,   # Neck -> Spine2
    4,   # Head -> Neck
    3,   # LeftShoulder -> Spine2
    6,   # LeftArm -> LeftShoulder
    7,   # LeftForeArm -> LeftArm
    8,   # LeftHand -> LeftForeArm
    3,   # RightShoulder -> Spine2
    10,  # RightArm -> RightShoulder
    11,  # RightForeArm -> RightArm
    12,  # RightHand -> RightForeArm
    0,   # LeftUpLeg -> Hips
    14,  # LeftLeg -> LeftUpLeg
    15,  # LeftFoot -> LeftLeg
    16,  # LeftToeBase -> LeftFoot
    0,   # RightUpLeg -> Hips
    18,  # RightLeg -> RightUpLeg
    19,  # RightFoot -> RightLeg
    20,  # RightToeBase -> RightFoot
]

# MHR joint indices that map to our simplified skeleton
# MHR has 127 joints, we pick the key ones
MHR_TO_SKELETON = {
    0: 0,    # pelvis -> Hips
    3: 1,    # spine1 -> Spine
    6: 2,    # spine2 -> Spine1
    9: 3,    # spine3 -> Spine2
    12: 4,   # neck -> Neck
    15: 5,   # head -> Head
    13: 6,   # left_collar -> LeftShoulder
    16: 7,   # left_shoulder -> LeftArm
    18: 8,   # left_elbow -> LeftForeArm
    20: 9,   # left_wrist -> LeftHand
    14: 10,  # right_collar -> RightShoulder
    17: 11,  # right_shoulder -> RightArm
    19: 12,  # right_elbow -> RightForeArm
    21: 13,  # right_wrist -> RightHand
    1: 14,   # left_hip -> LeftUpLeg
    4: 15,   # left_knee -> LeftLeg
    7: 16,   # left_ankle -> LeftFoot
    10: 17,  # left_foot -> LeftToeBase
    2: 18,   # right_hip -> RightUpLeg
    5: 19,   # right_knee -> RightLeg
    8: 20,   # right_ankle -> RightFoot
    11: 21,  # right_foot -> RightToeBase
}


def extract_skeleton_joints(joints_127: np.ndarray) -> np.ndarray:
    """Extract 22 key joints from MHR 127-joint data."""
    if joints_127 is None:
        return None
    
    joints_127 = np.array(joints_127)
    
    # Handle different input shapes
    if joints_127.ndim == 1:
        joints_127 = joints_127.reshape(-1, 3)
    
    num_joints = joints_127.shape[0]
    
    # Create output array
    skeleton = np.zeros((22, 3))
    
    for mhr_idx, skel_idx in MHR_TO_SKELETON.items():
        if mhr_idx < num_joints:
            skeleton[skel_idx] = joints_127[mhr_idx]
    
    return skeleton


def compute_bone_rotations(
    skeleton_positions: np.ndarray,
    parent_indices: List[int],
) -> np.ndarray:
    """
    Compute Euler rotations for each bone from positions.
    Returns rotations in degrees (XYZ order).
    """
    num_joints = len(skeleton_positions)
    rotations = np.zeros((num_joints, 3))
    
    for i in range(num_joints):
        parent = parent_indices[i]
        if parent < 0:
            continue
        
        # Get bone direction
        bone_vec = skeleton_positions[i] - skeleton_positions[parent]
        bone_len = np.linalg.norm(bone_vec)
        
        if bone_len < 1e-6:
            continue
        
        bone_dir = bone_vec / bone_len
        
        # Default bone direction is Y-up
        default_dir = np.array([0, 1, 0])
        
        # Compute rotation to align default to actual
        # Using simple Euler angle decomposition
        # This is a simplified approach - proper IK would be better
        
        # Y rotation (around Y axis)
        y_rot = np.arctan2(bone_dir[0], bone_dir[2])
        
        # X rotation (pitch)
        horiz_dist = np.sqrt(bone_dir[0]**2 + bone_dir[2]**2)
        x_rot = -np.arctan2(bone_dir[1], horiz_dist)
        
        rotations[i] = np.degrees([x_rot, y_rot, 0])
    
    return rotations


class ExportBVH:
    """
    Export skeleton animation to BVH format.
    
    BVH is the standard motion capture format supported by virtually
    all 3D software. Unlike FBX, BVH always imports correctly as
    a skeleton hierarchy with animation.
    
    Features:
    - Clean 22-joint skeleton (compatible with most retargeting)
    - Proper hierarchy for Maya, Blender, Unity, etc.
    - World-space or local-space export
    - Automatic scale adjustment
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
                    "default": 100.0,
                    "min": 0.1,
                    "max": 1000.0,
                    "step": 1.0,
                    "tooltip": "Scale factor (100=cm, typical for BVH)"
                }),
                "world_space": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include world position in root motion"
                }),
                "z_up": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use Z-up coordinate system (for Blender)"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("file_path", "status", "exported_frames")
    FUNCTION = "export_bvh"
    CATEGORY = "SAM3DBody2abc/Export"
    OUTPUT_NODE = True
    
    def export_bvh(
        self,
        mesh_sequence: List[Dict],
        filename: str = "body_animation",
        fps: float = 30.0,
        output_dir: str = "",
        scale: float = 100.0,
        world_space: bool = True,
        z_up: bool = False,
    ) -> Tuple[str, str, int]:
        """Export mesh sequence to BVH animation."""
        
        if not mesh_sequence:
            return ("", "Error: Empty mesh sequence", 0)
        
        # Get output directory
        if not output_dir:
            output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect valid frames with joints
        valid_frames = []
        for frame in mesh_sequence:
            if frame.get("valid") and frame.get("joints") is not None:
                valid_frames.append(frame)
        
        if not valid_frames:
            return ("", "Error: No valid frames with joint data", 0)
        
        print(f"[BVH Export] Processing {len(valid_frames)} frames...")
        
        # Extract skeleton from first frame to get T-pose/rest pose
        first_joints = valid_frames[0]["joints"]
        rest_skeleton = extract_skeleton_joints(first_joints)
        
        if rest_skeleton is None:
            return ("", "Error: Could not extract skeleton", 0)
        
        # Apply scale
        rest_skeleton = rest_skeleton * scale
        
        # Compute bone offsets (rest pose bone lengths)
        bone_offsets = np.zeros((22, 3))
        for i, parent in enumerate(SKELETON_PARENTS):
            if parent >= 0:
                bone_offsets[i] = rest_skeleton[i] - rest_skeleton[parent]
            else:
                bone_offsets[i] = rest_skeleton[i]  # Root offset
        
        # Apply coordinate transform if needed
        if z_up:
            # Swap Y and Z
            bone_offsets = bone_offsets[:, [0, 2, 1]]
            bone_offsets[:, 1] *= -1
        
        # Build BVH hierarchy string
        hierarchy = self._build_hierarchy(bone_offsets)
        
        # Build motion data
        frame_time = 1.0 / fps
        motion_data = []
        
        for frame in valid_frames:
            joints = frame["joints"]
            skeleton = extract_skeleton_joints(joints)
            
            if skeleton is None:
                continue
            
            skeleton = skeleton * scale
            
            # Get root position
            root_pos = skeleton[0].copy()
            
            # Apply camera offset for world space
            if world_space:
                cam_t = frame.get("camera")
                if cam_t is not None:
                    if hasattr(cam_t, 'cpu'):
                        cam_t = cam_t.cpu().numpy()
                    cam_t = np.array(cam_t).flatten()
                    # Add camera offset to root
                    root_pos[0] += -cam_t[0] * scale
                    root_pos[1] += cam_t[1] * scale
                    root_pos[2] += cam_t[2] * scale
            
            # Compute rotations
            rotations = compute_bone_rotations(skeleton, SKELETON_PARENTS)
            
            # Apply coordinate transform
            if z_up:
                root_pos = np.array([root_pos[0], root_pos[2], -root_pos[1]])
                # Adjust rotations for Z-up
                rotations = rotations[:, [0, 2, 1]]
            
            # Build frame data: root position + all rotations
            frame_data = [root_pos[0], root_pos[1], root_pos[2]]
            for rot in rotations:
                frame_data.extend([rot[2], rot[0], rot[1]])  # ZXY order for BVH
            
            motion_data.append(frame_data)
        
        # Write BVH file
        output_path = os.path.join(output_dir, f"{filename}.bvh")
        
        # Handle existing files
        counter = 1
        while os.path.exists(output_path):
            output_path = os.path.join(output_dir, f"{filename}_{counter:04d}.bvh")
            counter += 1
        
        with open(output_path, 'w') as f:
            # Write hierarchy
            f.write("HIERARCHY\n")
            f.write(hierarchy)
            
            # Write motion section
            f.write("MOTION\n")
            f.write(f"Frames: {len(motion_data)}\n")
            f.write(f"Frame Time: {frame_time:.6f}\n")
            
            for frame_data in motion_data:
                f.write(" ".join(f"{v:.6f}" for v in frame_data) + "\n")
        
        status = f"Exported {len(motion_data)} frames to BVH"
        print(f"[BVH Export] {status}")
        print(f"[BVH Export] File: {output_path}")
        
        return (output_path, status, len(motion_data))
    
    def _build_hierarchy(self, offsets: np.ndarray) -> str:
        """Build BVH hierarchy section."""
        lines = []
        
        # Build hierarchy recursively
        def write_joint(idx: int, indent: int = 0) -> None:
            name = SKELETON_JOINTS[idx]
            offset = offsets[idx]
            prefix = "  " * indent
            
            # Determine joint type
            children = [i for i, p in enumerate(SKELETON_PARENTS) if p == idx]
            
            if idx == 0:
                lines.append(f"ROOT {name}")
                lines.append("{")
                lines.append(f"  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
                lines.append("  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
            else:
                lines.append(f"{prefix}JOINT {name}")
                lines.append(f"{prefix}{{")
                lines.append(f"{prefix}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
                lines.append(f"{prefix}  CHANNELS 3 Zrotation Xrotation Yrotation")
            
            if children:
                for child_idx in children:
                    write_joint(child_idx, indent + 1)
            else:
                # End site for leaf joints
                lines.append(f"{prefix}  End Site")
                lines.append(f"{prefix}  {{")
                lines.append(f"{prefix}    OFFSET 0.000000 {offset[1] * 0.2:.6f} 0.000000")
                lines.append(f"{prefix}  }}")
            
            lines.append(f"{prefix}}}")
        
        write_joint(0)
        return "\n".join(lines) + "\n"


# Node registration
NODE_CLASS_MAPPINGS = {
    "SAM3DBody2abc_ExportBVH": ExportBVH,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBody2abc_ExportBVH": "📦 Export BVH Skeleton",
}
