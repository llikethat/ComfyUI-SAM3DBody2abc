"""
Blender Script: Export Animated FBX from Skeleton JSON
Creates armature with proper hierarchy and keyframes joint positions.

Usage: blender --background --python blender_animated_fbx.py -- input.json output.fbx

Settings:
- Scale: 1.0
- Up axis: Y
"""

import bpy
import json
import sys
import os
import math
from mathutils import Vector, Matrix, Euler


def clear_scene():
    """Remove all objects."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def create_armature_with_hierarchy(joint_positions, joint_parents, joint_names=None):
    """
    Create armature with proper parent-child hierarchy.
    
    Args:
        joint_positions: First frame joint positions (N, 3)
        joint_parents: Parent index for each joint (-1 for root)
        joint_names: Optional joint names
    """
    num_joints = len(joint_positions)
    
    # Create armature
    bpy.ops.object.armature_add(enter_editmode=True)
    armature = bpy.context.object
    armature.name = "Skeleton"
    arm_data = armature.data
    arm_data.name = "Skeleton_Data"
    
    # Remove default bone
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.armature.delete()
    
    # Generate joint names if not provided
    if joint_names is None:
        joint_names = [f"joint_{i:03d}" for i in range(num_joints)]
    
    # Create bones
    bones = []
    for i in range(num_joints):
        bone_name = joint_names[i] if i < len(joint_names) else f"joint_{i:03d}"
        bone = arm_data.edit_bones.new(bone_name)
        
        pos = joint_positions[i]
        # Flip X for Blender coordinate system
        head = Vector((-pos[0], pos[1], pos[2]))
        bone.head = head
        # Small offset for tail (will be adjusted based on children)
        bone.tail = head + Vector((0, 0.02, 0))
        
        bones.append(bone)
    
    # Set up parent hierarchy
    if joint_parents is not None:
        for i, parent_idx in enumerate(joint_parents):
            if parent_idx >= 0 and parent_idx < len(bones):
                bones[i].parent = bones[parent_idx]
                # Point parent tail toward child
                bones[parent_idx].tail = bones[i].head
    
    # Exit edit mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    print(f"[Blender] Created armature with {num_joints} bones")
    return armature


def animate_skeleton(armature, frames_data, fps):
    """
    Animate skeleton joints using keyframes.
    
    Args:
        armature: Blender armature object
        frames_data: List of frame data with joint_positions
        fps: Frames per second
    """
    # Set scene FPS
    bpy.context.scene.render.fps = int(fps)
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = len(frames_data) - 1
    
    # Get first frame positions as reference (rest pose)
    first_positions = frames_data[0].get("joint_positions", [])
    if not first_positions:
        print("[Blender] Warning: No joint positions in first frame")
        return
    
    # Enter pose mode
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    print(f"[Blender] Animating {len(frames_data)} frames...")
    
    for frame_idx, frame_data in enumerate(frames_data):
        joint_positions = frame_data.get("joint_positions", [])
        if not joint_positions:
            continue
        
        bpy.context.scene.frame_set(frame_idx)
        
        for bone_idx, pose_bone in enumerate(armature.pose.bones):
            if bone_idx >= len(joint_positions):
                break
            
            # Get current position
            pos = joint_positions[bone_idx]
            new_pos = Vector((-pos[0], pos[1], pos[2]))  # Flip X
            
            # Get rest position
            rest_pos = armature.data.bones[bone_idx].head_local
            
            # Calculate offset from rest pose
            offset = new_pos - rest_pos
            
            # Apply location
            pose_bone.location = offset
            pose_bone.keyframe_insert(data_path="location", frame=frame_idx)
        
        if (frame_idx + 1) % 50 == 0 or frame_idx == len(frames_data) - 1:
            print(f"[Blender] Keyframed {frame_idx + 1}/{len(frames_data)} frames")
    
    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    print(f"[Blender] Animation complete: {len(frames_data)} frames")


def export_fbx(output_path):
    """Export scene to FBX."""
    print(f"[Blender] Exporting FBX: {output_path}")
    
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=False,
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options='FBX_SCALE_ALL',
        axis_forward='-Z',
        axis_up='Y',
        object_types={'ARMATURE'},
        use_armature_deform_only=False,
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
    )
    
    print(f"[Blender] FBX exported: {output_path}")


def main():
    # Parse arguments
    argv = sys.argv
    try:
        idx = argv.index("--")
        args = argv[idx + 1:]
    except ValueError:
        print("[Blender] Error: No arguments after '--'")
        sys.exit(1)
    
    if len(args) < 2:
        print("[Blender] Usage: blender --background --python script.py -- input.json output.fbx")
        sys.exit(1)
    
    input_json = args[0]
    output_fbx = args[1]
    
    print(f"[Blender] Input: {input_json}")
    print(f"[Blender] Output: {output_fbx}")
    
    # Load JSON
    if not os.path.exists(input_json):
        print(f"[Blender] Error: File not found: {input_json}")
        sys.exit(1)
    
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    fps = data.get("fps", 24.0)
    frames = data.get("frames", [])
    joint_parents = data.get("joint_parents")
    joint_names = data.get("joint_names")
    
    print(f"[Blender] Loaded {len(frames)} frames at {fps} fps")
    
    if not frames:
        print("[Blender] Error: No frames in JSON")
        sys.exit(1)
    
    # Get first frame joint positions
    first_positions = frames[0].get("joint_positions", [])
    if not first_positions:
        print("[Blender] Error: No joint positions in first frame")
        sys.exit(1)
    
    print(f"[Blender] Joint count: {len(first_positions)}")
    
    # Clear scene
    clear_scene()
    
    # Create armature
    armature = create_armature_with_hierarchy(
        first_positions,
        joint_parents,
        joint_names
    )
    
    # Animate
    if len(frames) > 1:
        animate_skeleton(armature, frames, fps)
    
    # Export
    export_fbx(output_fbx)
    
    print("[Blender] Done!")


if __name__ == "__main__":
    main()
