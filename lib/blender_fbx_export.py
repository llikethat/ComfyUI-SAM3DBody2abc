"""
Blender FBX Export Script for SAM3DBody2abc
Converts animation JSON to FBX with mesh and/or skeleton.

Usage: blender --background --python blender_fbx_export.py -- input.json output.fbx include_mesh

Fixed settings:
- Scale: 1.0
- Up axis: Y
"""

import bpy
import json
import sys
import os
import math

def clear_scene():
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def create_mesh_with_animation(data, include_mesh=True):
    """Create animated mesh using shape keys."""
    frames = data.get("frames", [])
    faces = data.get("faces", [])
    fps = data.get("fps", 24.0)
    
    if not frames:
        print("[Blender] No frames to process")
        return None
    
    # Set scene FPS
    bpy.context.scene.render.fps = int(fps)
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = len(frames) - 1
    
    # Get first frame vertices for base mesh
    first_frame = frames[0]
    base_verts = first_frame.get("vertices", [])
    
    if not base_verts or not include_mesh:
        print("[Blender] Skeleton-only mode or no vertices")
        return None
    
    # Create mesh
    mesh = bpy.data.meshes.new("body_mesh")
    
    # Convert vertices - flip X for Blender coordinate system
    verts = [(-v[0], v[1], v[2]) for v in base_verts]
    
    # Create mesh from data
    if faces:
        mesh.from_pydata(verts, [], faces)
    else:
        mesh.from_pydata(verts, [], [])
    
    mesh.update()
    
    # Create object
    obj = bpy.data.objects.new("body", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Add basis shape key
    obj.shape_key_add(name="Basis", from_mix=False)
    
    # Add shape key for each frame
    print(f"[Blender] Creating {len(frames)} shape keys...")
    
    for i, frame in enumerate(frames):
        frame_verts = frame.get("vertices", [])
        if not frame_verts:
            continue
        
        # Create shape key
        sk = obj.shape_key_add(name=f"frame_{i:04d}", from_mix=False)
        
        # Set vertex positions (flip X)
        for j, v in enumerate(frame_verts):
            if j < len(sk.data):
                sk.data[j].co = (-v[0], v[1], v[2])
        
        # Animate shape key
        sk.value = 0.0
        sk.keyframe_insert(data_path="value", frame=max(0, i-1))
        
        sk.value = 1.0
        sk.keyframe_insert(data_path="value", frame=i)
        
        sk.value = 0.0
        sk.keyframe_insert(data_path="value", frame=min(len(frames)-1, i+1))
        
        if (i + 1) % 50 == 0:
            print(f"[Blender] Processed {i + 1}/{len(frames)} frames")
    
    print(f"[Blender] Created mesh with {len(frames)} animated shape keys")
    return obj

def create_skeleton(data):
    """Create animated skeleton from joint positions."""
    frames = data.get("frames", [])
    fps = data.get("fps", 24.0)
    
    if not frames:
        return None
    
    # Check if we have joint data
    first_joints = frames[0].get("joints")
    if not first_joints:
        print("[Blender] No joint data found")
        return None
    
    num_joints = len(first_joints)
    print(f"[Blender] Creating skeleton with {num_joints} joints")
    
    # Create armature
    bpy.ops.object.armature_add(enter_editmode=True)
    armature = bpy.context.object
    armature.name = "skeleton"
    arm_data = armature.data
    arm_data.name = "skeleton_data"
    
    # Remove default bone
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.armature.delete()
    
    # Get first frame joint positions
    first_joints = frames[0]["joints"]
    
    # Create bones for each joint
    for i, joint_pos in enumerate(first_joints):
        bone = arm_data.edit_bones.new(f"joint_{i:03d}")
        # Flip X for Blender coordinate system
        pos = (-joint_pos[0], joint_pos[1], joint_pos[2])
        bone.head = pos
        bone.tail = (pos[0], pos[1], pos[2] + 0.02)  # Small offset for tail
    
    # Exit edit mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Animate joints
    print(f"[Blender] Animating {num_joints} joints over {len(frames)} frames...")
    
    for frame_idx, frame in enumerate(frames):
        joints = frame.get("joints")
        if not joints:
            continue
        
        bpy.context.scene.frame_set(frame_idx)
        
        for bone_idx, joint_pos in enumerate(joints):
            if bone_idx < len(armature.pose.bones):
                bone = armature.pose.bones[f"joint_{bone_idx:03d}"]
                # Set location relative to rest position (flip X)
                rest_pos = arm_data.bones[f"joint_{bone_idx:03d}"].head_local
                new_pos = (-joint_pos[0], joint_pos[1], joint_pos[2])
                bone.location = (
                    new_pos[0] - rest_pos[0],
                    new_pos[1] - rest_pos[1],
                    new_pos[2] - rest_pos[2],
                )
                bone.keyframe_insert(data_path="location", frame=frame_idx)
        
        if (frame_idx + 1) % 50 == 0:
            print(f"[Blender] Animated {frame_idx + 1}/{len(frames)} frames")
    
    print(f"[Blender] Created animated skeleton")
    return armature

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
    
    print(f"[Blender] FBX exported successfully")

def main():
    # Parse arguments
    argv = sys.argv
    
    # Find '--' separator
    try:
        idx = argv.index("--")
        args = argv[idx + 1:]
    except ValueError:
        print("[Blender] Error: No arguments after '--'")
        sys.exit(1)
    
    if len(args) < 2:
        print("[Blender] Usage: blender --background --python script.py -- input.json output.fbx [include_mesh]")
        sys.exit(1)
    
    input_json = args[0]
    output_fbx = args[1]
    include_mesh = args[2] == "1" if len(args) > 2 else True
    
    print(f"[Blender] Input: {input_json}")
    print(f"[Blender] Output: {output_fbx}")
    print(f"[Blender] Include mesh: {include_mesh}")
    
    # Load JSON
    if not os.path.exists(input_json):
        print(f"[Blender] Error: Input file not found: {input_json}")
        sys.exit(1)
    
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    print(f"[Blender] Loaded {data.get('frame_count', 0)} frames at {data.get('fps', 24)} fps")
    
    # Clear scene
    clear_scene()
    
    # Create mesh with animation
    if include_mesh:
        mesh_obj = create_mesh_with_animation(data, include_mesh=True)
    
    # Create skeleton
    skeleton = create_skeleton(data)
    
    # Export FBX
    export_fbx(output_fbx)
    
    print("[Blender] Done!")

if __name__ == "__main__":
    main()
