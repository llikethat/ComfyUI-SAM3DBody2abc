"""
Blender Script: Export Animated FBX from Mesh Sequence JSON
Creates mesh with shape keys + armature with properly connected skeleton hierarchy.

Usage: blender --background --python blender_animated_fbx.py -- input.json output.fbx [up_axis]

Args:
    input.json: JSON with frames data
    output.fbx: Output FBX path
    up_axis: Y, Z, -Y, or -Z (default: Y)
"""

import bpy
import json
import sys
import os
from mathutils import Vector, Matrix
import math


def clear_scene():
    """Remove all objects."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Clean data blocks
    for armature in bpy.data.armatures:
        bpy.data.armatures.remove(armature)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)


def get_transform_for_axis(up_axis):
    """
    Get coordinate transformation based on desired up axis.
    SAM3DBody uses: X-right, Y-up, Z-forward (OpenGL convention)
    
    Returns: (flip_func, axis_forward, axis_up)
    """
    if up_axis == "Y":
        # Y-up (default, matches SAM3DBody)
        # Flip all axes as SAM3DBody does
        return lambda p: (-p[0], -p[1], -p[2]), '-Z', 'Y'
    elif up_axis == "Z":
        # Z-up (Blender default)
        # Rotate 90 degrees around X
        return lambda p: (-p[0], -p[2], p[1]), 'Y', 'Z'
    elif up_axis == "-Y":
        # -Y up (upside down)
        return lambda p: (-p[0], p[1], p[2]), 'Z', '-Y'
    elif up_axis == "-Z":
        # -Z up
        return lambda p: (-p[0], p[2], -p[1]), '-Y', '-Z'
    else:
        # Default to Y-up
        return lambda p: (-p[0], -p[1], -p[2]), '-Z', 'Y'


def build_children_map(joint_parents):
    """Build a map of parent -> list of children indices."""
    children = {}
    for i, parent_idx in enumerate(joint_parents):
        if parent_idx >= 0:
            if parent_idx not in children:
                children[parent_idx] = []
            children[parent_idx].append(i)
    return children


def find_root_joint(joint_parents):
    """Find root joint (parent == -1 or self-referencing)."""
    for i, parent_idx in enumerate(joint_parents):
        if parent_idx < 0 or parent_idx == i:
            return i
    return 0  # Fallback to first joint


def create_mesh_with_shapekeys(first_vertices, faces, all_frames, fps, transform_func):
    """Create mesh with shape keys for animation."""
    mesh = bpy.data.meshes.new("body_mesh")
    
    # Transform vertices
    verts = [transform_func(v) for v in first_vertices]
    
    if faces:
        mesh.from_pydata(verts, [], faces)
    else:
        mesh.from_pydata(verts, [], [])
    
    mesh.update()
    
    obj = bpy.data.objects.new("body", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Set scene settings
    bpy.context.scene.render.fps = int(fps)
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = len(all_frames) - 1
    
    # Add basis shape key
    obj.shape_key_add(name="Basis", from_mix=False)
    
    print(f"[Blender] Creating {len(all_frames)} shape keys...")
    
    for frame_idx, frame_data in enumerate(all_frames):
        frame_verts = frame_data.get("vertices")
        if not frame_verts:
            continue
        
        sk = obj.shape_key_add(name=f"frame_{frame_idx:04d}", from_mix=False)
        
        for j, v in enumerate(frame_verts):
            if j < len(sk.data):
                sk.data[j].co = transform_func(v)
        
        # Keyframe the shape key
        sk.value = 0.0
        sk.keyframe_insert(data_path="value", frame=max(0, frame_idx - 1))
        
        sk.value = 1.0
        sk.keyframe_insert(data_path="value", frame=frame_idx)
        
        sk.value = 0.0
        sk.keyframe_insert(data_path="value", frame=min(len(all_frames) - 1, frame_idx + 1))
        
        if (frame_idx + 1) % 50 == 0:
            print(f"[Blender] Shape keys: {frame_idx + 1}/{len(all_frames)}")
    
    print(f"[Blender] Created mesh with {len(all_frames)} shape keys")
    return obj


def create_armature(first_joints, joint_parents, all_frames, fps, transform_func):
    """
    Create armature with properly connected skeleton hierarchy.
    
    The skeleton hierarchy uses joint_parents to establish parent-child relationships.
    Root joint (typically pelvis/hip) has parent -1.
    Bone tails point toward first child or use small offset if leaf.
    """
    num_joints = len(first_joints)
    
    # Build hierarchy info
    children_map = build_children_map(joint_parents) if joint_parents else {}
    root_idx = find_root_joint(joint_parents) if joint_parents else 0
    
    print(f"[Blender] Root joint: {root_idx}")
    print(f"[Blender] Creating armature with {num_joints} joints...")
    
    # Create armature
    arm_data = bpy.data.armatures.new("Skeleton_Data")
    armature = bpy.data.objects.new("Skeleton", arm_data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Transform all joint positions
    joint_positions = [transform_func(first_joints[i]) for i in range(num_joints)]
    
    # Create all bones first
    bones = []
    for i in range(num_joints):
        bone = arm_data.edit_bones.new(f"joint_{i:03d}")
        pos = joint_positions[i]
        bone.head = Vector(pos)
        # Temporary tail - will adjust after all bones created
        bone.tail = Vector((pos[0], pos[1], pos[2] + 0.02))
        bones.append(bone)
    
    # Set parent hierarchy and adjust bone tails
    if joint_parents and len(joint_parents) == num_joints:
        for i in range(num_joints):
            parent_idx = joint_parents[i]
            
            # Set parent (skip root and invalid parents)
            if parent_idx >= 0 and parent_idx < num_joints and parent_idx != i:
                bones[i].parent = bones[parent_idx]
                bones[i].use_connect = False  # Don't force connection
            
            # Adjust bone tail to point toward first child (better visualization)
            if i in children_map and len(children_map[i]) > 0:
                # Point toward first child
                first_child = children_map[i][0]
                child_pos = joint_positions[first_child]
                direction = Vector(child_pos) - Vector(joint_positions[i])
                if direction.length > 0.001:
                    bones[i].tail = Vector(joint_positions[i]) + direction.normalized() * min(direction.length * 0.8, 0.1)
                else:
                    bones[i].tail = Vector((joint_positions[i][0], joint_positions[i][1], joint_positions[i][2] + 0.02))
            else:
                # Leaf bone - small offset in Z
                bones[i].tail = Vector((joint_positions[i][0], joint_positions[i][1], joint_positions[i][2] + 0.02))
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Animate joints
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    print(f"[Blender] Animating {num_joints} joints over {len(all_frames)} frames...")
    
    # Store rest positions
    rest_positions = [Vector(arm_data.bones[i].head_local) for i in range(num_joints)]
    
    for frame_idx, frame_data in enumerate(all_frames):
        joints = frame_data.get("joint_coords")
        if not joints:
            continue
        
        bpy.context.scene.frame_set(frame_idx)
        
        for bone_idx, pose_bone in enumerate(armature.pose.bones):
            if bone_idx >= len(joints):
                break
            
            pos = joints[bone_idx]
            new_pos = Vector(transform_func(pos))
            
            # Calculate offset from rest pose
            offset = new_pos - rest_positions[bone_idx]
            
            pose_bone.location = offset
            pose_bone.keyframe_insert(data_path="location", frame=frame_idx)
        
        if (frame_idx + 1) % 50 == 0:
            print(f"[Blender] Animated {frame_idx + 1}/{len(all_frames)} frames")
    
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[Blender] Created animated skeleton with proper hierarchy")
    return armature


def export_fbx(output_path, axis_forward, axis_up):
    """Export to FBX with specified orientation."""
    print(f"[Blender] Exporting: {output_path}")
    print(f"[Blender] Orientation: forward={axis_forward}, up={axis_up}")
    
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=False,
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options='FBX_SCALE_ALL',
        axis_forward=axis_forward,
        axis_up=axis_up,
        object_types={'MESH', 'ARMATURE'},
        use_mesh_modifiers=True,
        mesh_smooth_type='FACE',
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
    print(f"[Blender] Export complete")


def main():
    argv = sys.argv
    try:
        idx = argv.index("--")
        args = argv[idx + 1:]
    except ValueError:
        print("[Blender] Error: No arguments")
        sys.exit(1)
    
    if len(args) < 2:
        print("[Blender] Usage: blender --background --python script.py -- input.json output.fbx [up_axis] [include_mesh]")
        sys.exit(1)
    
    input_json = args[0]
    output_fbx = args[1]
    up_axis = args[2] if len(args) > 2 else "Y"
    include_mesh = args[3] == "1" if len(args) > 3 else True
    
    print(f"[Blender] Input: {input_json}")
    print(f"[Blender] Output: {output_fbx}")
    print(f"[Blender] Up axis: {up_axis}")
    print(f"[Blender] Include mesh: {include_mesh}")
    
    if not os.path.exists(input_json):
        print(f"[Blender] Error: File not found: {input_json}")
        sys.exit(1)
    
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    fps = data.get("fps", 24.0)
    frames = data.get("frames", [])
    faces = data.get("faces")
    joint_parents = data.get("joint_parents")
    
    print(f"[Blender] {len(frames)} frames at {fps} fps")
    if joint_parents:
        print(f"[Blender] Joint parents available: {len(joint_parents)} joints")
    
    if not frames:
        print("[Blender] Error: No frames")
        sys.exit(1)
    
    # Get transformation based on up axis
    transform_func, axis_forward, axis_up_export = get_transform_for_axis(up_axis)
    
    first_verts = frames[0].get("vertices")
    first_joints = frames[0].get("joint_coords")
    
    clear_scene()
    
    # Create mesh with shape keys
    if include_mesh and first_verts:
        mesh_obj = create_mesh_with_shapekeys(first_verts, faces, frames, fps, transform_func)
    
    # Create animated skeleton with proper hierarchy
    if first_joints:
        armature = create_armature(first_joints, joint_parents, frames, fps, transform_func)
    
    export_fbx(output_fbx, axis_forward, axis_up_export)
    print("[Blender] Done!")


if __name__ == "__main__":
    main()
