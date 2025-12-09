"""
Blender Script: Export Animated FBX from Mesh Sequence JSON
Creates mesh with shape keys + armature with keyframed joints.

Usage: blender --background --python blender_animated_fbx.py -- input.json output.fbx

Settings:
- Scale: 1.0
- Up axis: Y
"""

import bpy
import json
import sys
import os
from mathutils import Vector


def clear_scene():
    """Remove all objects."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def create_mesh_with_shapekeys(first_vertices, faces, all_frames, fps):
    """
    Create mesh with shape keys for animation.
    
    Args:
        first_vertices: Base mesh vertices
        faces: Face indices
        all_frames: List of frame data with vertices
        fps: Frames per second
    """
    # Create mesh
    mesh = bpy.data.meshes.new("body_mesh")
    
    # Flip coordinates for Blender (negate all axes as SAM3DBody does)
    verts = [(-v[0], -v[1], -v[2]) for v in first_vertices]
    
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
    
    # Set scene settings
    bpy.context.scene.render.fps = int(fps)
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = len(all_frames) - 1
    
    # Add basis shape key
    obj.shape_key_add(name="Basis", from_mix=False)
    
    print(f"[Blender] Creating {len(all_frames)} shape keys...")
    
    # Create shape key for each frame
    for frame_idx, frame_data in enumerate(all_frames):
        frame_verts = frame_data.get("vertices")
        if not frame_verts:
            continue
        
        sk = obj.shape_key_add(name=f"frame_{frame_idx:04d}", from_mix=False)
        
        # Set vertices (flip coordinates)
        for j, v in enumerate(frame_verts):
            if j < len(sk.data):
                sk.data[j].co = (-v[0], -v[1], -v[2])
        
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


def create_armature(first_joints, joint_parents, all_frames, fps):
    """
    Create armature with animated joints.
    
    Args:
        first_joints: First frame joint positions (127 joints)
        joint_parents: Parent index for each joint
        all_frames: List of frame data with joint_coords
        fps: Frames per second
    """
    num_joints = len(first_joints)
    
    # Create armature
    bpy.ops.object.armature_add(enter_editmode=True)
    armature = bpy.context.object
    armature.name = "Skeleton"
    arm_data = armature.data
    arm_data.name = "Skeleton_Data"
    
    # Remove default bone
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.armature.delete()
    
    # Create bones
    bones = []
    for i in range(num_joints):
        bone = arm_data.edit_bones.new(f"joint_{i:03d}")
        
        pos = first_joints[i]
        # Flip coordinates to match mesh
        head = Vector((-pos[0], -pos[1], -pos[2]))
        bone.head = head
        bone.tail = head + Vector((0, 0.02, 0))
        
        bones.append(bone)
    
    # Set parent hierarchy
    if joint_parents:
        for i, parent_idx in enumerate(joint_parents):
            if parent_idx >= 0 and parent_idx < len(bones):
                bones[i].parent = bones[parent_idx]
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Animate joints
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    print(f"[Blender] Animating {num_joints} joints over {len(all_frames)} frames...")
    
    for frame_idx, frame_data in enumerate(all_frames):
        joints = frame_data.get("joint_coords")
        if not joints:
            continue
        
        bpy.context.scene.frame_set(frame_idx)
        
        for bone_idx, pose_bone in enumerate(armature.pose.bones):
            if bone_idx >= len(joints):
                break
            
            pos = joints[bone_idx]
            new_pos = Vector((-pos[0], -pos[1], -pos[2]))
            
            rest_pos = arm_data.bones[bone_idx].head_local
            offset = new_pos - rest_pos
            
            pose_bone.location = offset
            pose_bone.keyframe_insert(data_path="location", frame=frame_idx)
        
        if (frame_idx + 1) % 50 == 0:
            print(f"[Blender] Animated {frame_idx + 1}/{len(all_frames)} frames")
    
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[Blender] Created animated skeleton")
    return armature


def export_fbx(output_path):
    """Export to FBX."""
    print(f"[Blender] Exporting: {output_path}")
    
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
        print("[Blender] Usage: blender --background --python script.py -- input.json output.fbx")
        sys.exit(1)
    
    input_json = args[0]
    output_fbx = args[1]
    include_mesh = args[2] == "1" if len(args) > 2 else True
    
    print(f"[Blender] Input: {input_json}")
    print(f"[Blender] Output: {output_fbx}")
    print(f"[Blender] Include mesh: {include_mesh}")
    
    if not os.path.exists(input_json):
        print(f"[Blender] Error: File not found")
        sys.exit(1)
    
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    fps = data.get("fps", 24.0)
    frames = data.get("frames", [])
    faces = data.get("faces")
    joint_parents = data.get("joint_parents")
    
    print(f"[Blender] {len(frames)} frames at {fps} fps")
    
    if not frames:
        print("[Blender] Error: No frames")
        sys.exit(1)
    
    # Get first frame data
    first_verts = frames[0].get("vertices")
    first_joints = frames[0].get("joint_coords")
    
    clear_scene()
    
    # Create mesh with shape keys
    if include_mesh and first_verts:
        mesh_obj = create_mesh_with_shapekeys(first_verts, faces, frames, fps)
    
    # Create animated skeleton
    if first_joints:
        armature = create_armature(first_joints, joint_parents, frames, fps)
    
    export_fbx(output_fbx)
    print("[Blender] Done!")


if __name__ == "__main__":
    main()
